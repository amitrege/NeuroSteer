from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .basis import Basis
from .catalog import StackMap, locate_decoder_stack, primary_tensor
from .capture import _select_positions, watch_layers
from .pairs import ReadSpec, WriteSpec
from .rules import ConstantRule, DiagonalGuard, LinearRule, RuleOutput, ThresholdRule
from .trace import Trace, TraceEvent
from .write import apply_write


RuleLike = ConstantRule | ThresholdRule | LinearRule


@dataclass
class Session:
    model: nn.Module
    write_basis: Basis
    write_spec: WriteSpec
    rule: RuleLike
    read_basis: Basis | None = None
    read_spec: ReadSpec | None = None
    mode: str = "static"
    guard: DiagonalGuard | None = None
    trace: Trace | None = None

    def __post_init__(self) -> None:
        self.stack: StackMap = locate_decoder_stack(self.model)
        self.trace = self.trace or Trace()
        self.handles: list[RemovableHandle] = []
        self.forward_step = 0
        self.current_attention_mask: Tensor | None = None
        self.live_coefficients: Tensor | None = None
        self.queued_coefficients: Tensor | None = None
        self.last_rank = self.write_basis.rank
        self._validate_setup()

    def _validate_setup(self) -> None:
        normalized_write_layers = [self.stack.normalize(layer) for layer in self.write_spec.layers]
        for layer in normalized_write_layers:
            if layer not in self.write_basis.layers:
                raise ValueError(f"write basis is missing layer {layer}")
            if self.write_basis.layers[layer].rank != self.last_rank:
                raise ValueError("all write layers must share the same rank")

        if self.mode in {"same_pass", "next_token", "preview"}:
            if self.read_basis is None or self.read_spec is None:
                raise ValueError(f"mode {self.mode!r} requires read_basis and read_spec")
            read_layer = self.stack.normalize(self.read_spec.layer)
            if read_layer not in self.read_basis.layers:
                raise ValueError(f"read basis is missing layer {read_layer}")
            if self.mode == "same_pass":
                earliest_write = min(normalized_write_layers)
                if read_layer > earliest_write:
                    raise ValueError(
                        "same-pass mode requires the read layer to be at or before every write layer"
                    )

    def __enter__(self) -> "Session":
        if self.mode == "preview":
            raise ValueError("preview mode uses preview_call instead of context attachment")
        self.attach()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def attach(self) -> None:
        if self.handles:
            return

        root_pre = self.model.register_forward_pre_hook(self._before_forward, with_kwargs=True)
        self.handles.append(root_pre)

        if self.mode in {"same_pass", "next_token"}:
            read_layer = self.stack.normalize(self.read_spec.layer)
            read_module = self.stack.module(read_layer)
            read_handle = read_module.register_forward_hook(self._after_read)
            self.handles.append(read_handle)

        for layer in self.write_spec.layers:
            normalized = self.stack.normalize(layer)
            write_module = self.stack.module(normalized)
            write_handle = write_module.register_forward_hook(self._after_write)
            self.handles.append(write_handle)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.current_attention_mask = None
        self.live_coefficients = None
        self.queued_coefficients = None

    def _infer_batch_size(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        tensor = kwargs.get("input_ids")
        if tensor is None and args:
            first = args[0]
            if isinstance(first, Tensor):
                tensor = first
        if tensor is None:
            mask = kwargs.get("attention_mask")
            if isinstance(mask, Tensor):
                tensor = mask
        if tensor is None:
            raise ValueError("could not infer batch size from forward inputs")
        return int(tensor.shape[0])

    def _before_forward(
        self, _module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        self.forward_step += 1
        attention_mask = kwargs.get("attention_mask")
        self.current_attention_mask = attention_mask.detach() if isinstance(attention_mask, Tensor) else None
        batch_size = self._infer_batch_size(args, kwargs)

        if self.mode == "static":
            output = self.rule.make(rank=self.last_rank, batch_size=batch_size, device=self.model.device)
            self.live_coefficients = output.coefficients.to(self.model.device)
            self._log_rule(
                layer=self.stack.normalize(self.write_spec.layers[0]),
                layer_label=self.stack.label(self.write_spec.layers[0]),
                output=output,
                distance=None,
                applied=True,
                pass_name="live",
            )
        elif self.mode == "next_token":
            if self.queued_coefficients is None:
                self.live_coefficients = torch.zeros(batch_size, self.last_rank, device=self.model.device)
            else:
                self.live_coefficients = self.queued_coefficients.to(self.model.device)
            self.queued_coefficients = None
        else:
            self.live_coefficients = None
        return None

    def _evaluate_dynamic_rule(self, coords: Tensor) -> tuple[RuleOutput, Tensor, Tensor | None]:
        if isinstance(self.rule, ConstantRule):
            output = self.rule.make(rank=self.last_rank, batch_size=coords.shape[0], device=coords.device)
        else:
            output = self.rule.evaluate(coords, rank=self.last_rank)

        applied = torch.ones_like(output.strength, dtype=torch.bool)
        distance = None
        if self.guard is not None:
            allowed, distance = self.guard.allow(coords)
            applied = allowed
            output = RuleOutput(
                score=output.score,
                strength=output.strength * allowed.to(output.strength.dtype),
                coefficients=output.coefficients * allowed.to(output.coefficients.dtype).unsqueeze(-1),
            )
        return output, applied, distance

    def _after_read(self, module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        hidden = primary_tensor(output)
        layer = self._layer_for_module(module)
        selected = _select_positions(
            hidden=hidden,
            attention_mask=self.current_attention_mask.to(hidden.device) if self.current_attention_mask is not None else None,
            token=self.read_spec.token,
            reduce=self.read_spec.reduce,
        )
        coords = self.read_basis.layers[layer].coordinates(selected)
        output_values, applied, distance = self._evaluate_dynamic_rule(coords)

        coords_mean = coords.mean(dim=0).detach().cpu().tolist()
        distance_value = None if distance is None else float(distance.mean().detach().cpu())
        self.trace.add(
            TraceEvent(
                phase="read",
                pass_name="live",
                step=self.forward_step,
                layer=layer,
                layer_label=self.stack.label(layer),
                value=float(torch.linalg.norm(selected, dim=-1).mean().detach().cpu()),
                coords=[float(x) for x in coords_mean],
                applied=bool(applied.all().detach().cpu()),
                distance=distance_value,
                note="basis coordinates at read point",
            )
        )
        self._log_rule(
            layer=layer,
            layer_label=self.stack.label(layer),
            output=output_values,
            distance=distance_value,
            applied=bool(applied.all().detach().cpu()),
            pass_name="live",
        )

        if self.mode == "same_pass":
            self.live_coefficients = output_values.coefficients.to(hidden.device)
        elif self.mode == "next_token":
            self.queued_coefficients = output_values.coefficients.detach().to(hidden.device)
        return output

    def _after_write(self, module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
        if self.live_coefficients is None:
            return output
        layer = self._layer_for_module(module)
        updated = apply_write(
            output,
            basis=self.write_basis.layers[layer],
            coefficients=self.live_coefficients.to(primary_tensor(output).device),
            token=self.write_spec.token,
            operator=self.write_spec.operator,
            attention_mask=self.current_attention_mask.to(primary_tensor(output).device)
            if self.current_attention_mask is not None
            else None,
        )
        before = primary_tensor(output)
        after = primary_tensor(updated)
        delta_norm = float((after - before).norm(dim=-1).mean().detach().cpu())
        strength = float(torch.linalg.norm(self.live_coefficients, dim=-1).mean().detach().cpu())
        self.trace.add(
            TraceEvent(
                phase="write",
                pass_name="live",
                step=self.forward_step,
                layer=layer,
                layer_label=self.stack.label(layer),
                value=strength,
                coords=[],
                applied=True,
                distance=None,
                note=f"delta_norm={delta_norm:.4f}",
            )
        )
        return updated

    def _layer_for_module(self, module: nn.Module) -> int:
        for index in range(self.stack.count()):
            if self.stack.module(index) is module:
                return index
        raise ValueError("hook fired from unknown layer")

    def _log_rule(
        self,
        *,
        layer: int,
        layer_label: str,
        output: RuleOutput,
        distance: float | None,
        applied: bool,
        pass_name: str,
    ) -> None:
        self.trace.add(
            TraceEvent(
                phase="rule",
                pass_name=pass_name,
                step=self.forward_step,
                layer=layer,
                layer_label=layer_label,
                value=float(output.strength.mean().detach().cpu()),
                coords=[float(x) for x in output.coefficients.mean(dim=0).detach().cpu().tolist()],
                applied=applied,
                distance=distance,
                note="strength chosen by rule",
            )
        )

    @contextmanager
    def _temporary_write(self, coefficients: Tensor, *, attention_mask: Tensor | None) -> Any:
        handles: list[RemovableHandle] = []
        self.current_attention_mask = attention_mask
        self.live_coefficients = coefficients
        for layer in self.write_spec.layers:
            normalized = self.stack.normalize(layer)
            write_module = self.stack.module(normalized)
            handles.append(write_module.register_forward_hook(self._after_write))
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()
            self.current_attention_mask = None
            self.live_coefficients = None

    @torch.no_grad()
    def preview_call(self, *args: Any, **kwargs: Any) -> Any:
        if self.mode != "preview":
            raise ValueError("preview_call is only available when mode='preview'")

        self.forward_step += 1
        attention_mask = kwargs.get("attention_mask")
        self.current_attention_mask = attention_mask.detach() if isinstance(attention_mask, Tensor) else None
        read_layer = self.stack.normalize(self.read_spec.layer)

        with watch_layers(self.model, [read_layer], stack=self.stack, clone=False) as bucket:
            self.model(*args, **kwargs)

        preview_hidden = bucket[read_layer][-1]
        selected = _select_positions(
            hidden=preview_hidden,
            attention_mask=self.current_attention_mask.to(preview_hidden.device)
            if self.current_attention_mask is not None
            else None,
            token=self.read_spec.token,
            reduce=self.read_spec.reduce,
        )
        coords = self.read_basis.layers[read_layer].coordinates(selected)
        output_values, applied, distance = self._evaluate_dynamic_rule(coords)
        coords_mean = coords.mean(dim=0).detach().cpu().tolist()
        distance_value = None if distance is None else float(distance.mean().detach().cpu())

        self.trace.add(
            TraceEvent(
                phase="read",
                pass_name="preview",
                step=self.forward_step,
                layer=read_layer,
                layer_label=self.stack.label(read_layer),
                value=float(torch.linalg.norm(selected, dim=-1).mean().detach().cpu()),
                coords=[float(x) for x in coords_mean],
                applied=bool(applied.all().detach().cpu()),
                distance=distance_value,
                note="preview readout",
            )
        )
        self._log_rule(
            layer=read_layer,
            layer_label=self.stack.label(read_layer),
            output=output_values,
            distance=distance_value,
            applied=bool(applied.all().detach().cpu()),
            pass_name="preview",
        )

        with self._temporary_write(
            output_values.coefficients.to(self.model.device),
            attention_mask=self.current_attention_mask.to(self.model.device)
            if self.current_attention_mask is not None
            else None,
        ):
            return self.model(*args, **kwargs)
