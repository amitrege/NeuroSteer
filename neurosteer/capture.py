from __future__ import annotations

from collections import defaultdict
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from .catalog import StackMap, locate_decoder_stack, primary_tensor
from .pairs import PromptPair, ReadSpec


@contextmanager
def watch_layers(
    model: nn.Module,
    layers: Sequence[int],
    *,
    stack: StackMap | None = None,
    clone: bool = True,
) -> Generator[dict[int, list[Tensor]], None, None]:
    stack = stack or locate_decoder_stack(model)
    bucket: dict[int, list[Tensor]] = defaultdict(list)
    handles: list[RemovableHandle] = []

    def make_hook(layer: int):
        def hook(_module: nn.Module, _args: tuple[Any, ...], output: Any) -> Any:
            hidden = primary_tensor(output)
            bucket[layer].append(hidden.detach().clone() if clone else hidden.detach())
            return output

        return hook

    for layer in layers:
        normalized = stack.normalize(layer)
        handle = stack.module(normalized).register_forward_hook(make_hook(normalized))
        handles.append(handle)

    try:
        yield bucket
    finally:
        for handle in handles:
            handle.remove()


def _select_positions(
    hidden: Tensor,
    attention_mask: Tensor | None,
    token: str | int,
    reduce: str,
) -> Tensor:
    if token == "last":
        if attention_mask is None:
            selected = hidden[:, -1, :]
        else:
            positions = attention_mask.sum(dim=1) - 1
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            selected = hidden[rows, positions.to(hidden.device), :]
    elif isinstance(token, int):
        index = token if token >= 0 else hidden.shape[1] + token
        selected = hidden[:, index, :]
    else:
        raise ValueError(f"unsupported read token setting {token!r}")

    if reduce == "mean":
        return selected.mean(dim=0, keepdim=True)
    if reduce == "none":
        return selected
    raise ValueError(f"unsupported reduce mode {reduce!r}")


@torch.no_grad()
def read_prompt_batch(
    model: nn.Module,
    tokenizer: Any,
    prompts: Sequence[str],
    layers: Sequence[int],
    read_spec: ReadSpec,
    *,
    batch_size: int = 4,
    stack: StackMap | None = None,
) -> dict[int, Tensor]:
    stack = stack or locate_decoder_stack(model)
    normalized_layers = [stack.normalize(layer) for layer in layers]
    chunks = [prompts[start : start + batch_size] for start in range(0, len(prompts), batch_size)]
    collected: dict[int, list[Tensor]] = {layer: [] for layer in normalized_layers}

    for chunk in chunks:
        encoded = tokenizer(chunk, padding=True, return_tensors="pt")
        encoded = {name: value.to(model.device) for name, value in encoded.items()}
        attention_mask = encoded.get("attention_mask")

        with watch_layers(model, normalized_layers, stack=stack, clone=False) as bucket:
            model(**encoded)

        for layer in normalized_layers:
            hidden = bucket[layer][-1]
            selected = _select_positions(
                hidden=hidden,
                attention_mask=attention_mask,
                token=read_spec.token,
                reduce=read_spec.reduce,
            )
            collected[layer].append(selected.cpu())

    return {layer: torch.cat(states, dim=0) for layer, states in collected.items()}


@torch.no_grad()
def read_pair_batch(
    model: nn.Module,
    tokenizer: Any,
    pairs: Sequence[PromptPair],
    layers: Sequence[int],
    read_spec: ReadSpec,
    *,
    batch_size: int = 4,
    stack: StackMap | None = None,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    positives = [pair.positive for pair in pairs]
    negatives = [pair.negative for pair in pairs]
    positive_states = read_prompt_batch(
        model,
        tokenizer,
        positives,
        layers,
        read_spec,
        batch_size=batch_size,
        stack=stack,
    )
    negative_states = read_prompt_batch(
        model,
        tokenizer,
        negatives,
        layers,
        read_spec,
        batch_size=batch_size,
        stack=stack,
    )
    return positive_states, negative_states
