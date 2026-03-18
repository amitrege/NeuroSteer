from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn


@dataclass(frozen=True)
class StackMap:
    layers: nn.ModuleList
    path: str

    def count(self) -> int:
        return len(self.layers)

    def normalize(self, layer: int) -> int:
        if layer < 0:
            layer = self.count() + layer
        if layer < 0 or layer >= self.count():
            raise IndexError(f"layer {layer} is out of range for {self.count()} layers")
        return layer

    def module(self, layer: int) -> nn.Module:
        return self.layers[self.normalize(layer)]

    def label(self, layer: int) -> str:
        return f"{self.path}.{self.normalize(layer)}"


def locate_decoder_stack(model: nn.Module) -> StackMap:
    candidates: list[tuple[str, nn.ModuleList]] = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        if isinstance(layers, nn.ModuleList):
            candidates.append(("model.layers", layers))
    if hasattr(model, "layers"):
        layers = model.layers
        if isinstance(layers, nn.ModuleList):
            candidates.append(("layers", layers))

    if not candidates:
        raise ValueError(
            "expected a Llama-style decoder model with `model.layers` or `layers`"
        )

    if len(candidates) > 1:
        return StackMap(layers=candidates[0][1], path=candidates[0][0])
    return StackMap(layers=candidates[0][1], path=candidates[0][0])


def primary_tensor(output: Any) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple) and output:
        head = output[0]
        if isinstance(head, Tensor):
            return head
    raise TypeError(f"expected a tensor or tensor-first tuple, got {type(output)!r}")


def rebuild_output(original: Any, hidden: Tensor) -> Any:
    if isinstance(original, Tensor):
        return hidden
    if isinstance(original, tuple):
        return (hidden,) + original[1:]
    raise TypeError(f"cannot rebuild output of type {type(original)!r}")
