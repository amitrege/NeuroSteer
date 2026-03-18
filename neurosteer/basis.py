from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor, nn

from .capture import read_pair_batch
from .catalog import StackMap, locate_decoder_stack
from .pairs import PromptPair, ReadSpec


def _normalize_rows(matrix: Tensor) -> Tensor:
    return torch.nn.functional.normalize(matrix, dim=-1)


@dataclass
class LayerBasis:
    components: Tensor
    explained: Tensor = field(default_factory=lambda: torch.empty(0))

    def __post_init__(self) -> None:
        if self.components.ndim != 2:
            raise ValueError("components must have shape [rank, hidden]")

    @property
    def rank(self) -> int:
        return int(self.components.shape[0])

    def coordinates(self, hidden: Tensor) -> Tensor:
        return hidden @ self.components.to(hidden.device, hidden.dtype).T

    def compose(self, coefficients: Tensor) -> Tensor:
        components = self.components.to(coefficients.device, coefficients.dtype)
        if coefficients.ndim == 1:
            coefficients = coefficients.unsqueeze(0)
        return coefficients @ components

    def project(self, hidden: Tensor) -> Tensor:
        return self.compose(self.coordinates(hidden))


@dataclass
class Basis:
    layers: dict[int, LayerBasis]
    method: str

    @property
    def rank(self) -> int:
        if not self.layers:
            raise ValueError("basis is empty")
        first = next(iter(self.layers.values()))
        return first.rank

    def layer(self, layer: int) -> LayerBasis:
        return self.layers[layer]

    @classmethod
    def fit_from_states(
        cls,
        positive_states: dict[int, Tensor],
        negative_states: dict[int, Tensor],
        *,
        method: str,
        rank: int = 1,
    ) -> "Basis":
        fitted: dict[int, LayerBasis] = {}
        for layer, positive in positive_states.items():
            negative = negative_states[layer]
            deltas = positive - negative
            if method == "mean_delta":
                direction = _normalize_rows(deltas.mean(dim=0, keepdim=True))
                fitted[layer] = LayerBasis(components=direction.cpu(), explained=torch.tensor([1.0]))
                continue

            if method not in {"pca_1", "pca_k"}:
                raise ValueError(f"unsupported basis method {method!r}")

            target_rank = 1 if method == "pca_1" else rank
            centered = deltas - deltas.mean(dim=0, keepdim=True)
            _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
            components = _normalize_rows(vh[:target_rank])
            alignment = torch.sign((deltas @ components[0]).mean())
            if alignment < 0:
                components = -components
            total = torch.clamp((singular_values**2).sum(), min=1e-12)
            explained = (singular_values[:target_rank] ** 2) / total
            fitted[layer] = LayerBasis(components=components.cpu(), explained=explained.cpu())
        return cls(layers=fitted, method=method)

    @classmethod
    def fit_from_pairs(
        cls,
        model: nn.Module,
        tokenizer: Any,
        pairs: list[PromptPair],
        *,
        layers: list[int],
        read_spec: ReadSpec,
        method: str,
        rank: int = 1,
        batch_size: int = 4,
        stack: StackMap | None = None,
    ) -> "Basis":
        positive_states, negative_states = read_pair_batch(
            model,
            tokenizer,
            pairs,
            layers,
            read_spec,
            batch_size=batch_size,
            stack=stack or locate_decoder_stack(model),
        )
        return cls.fit_from_states(
            positive_states,
            negative_states,
            method=method,
            rank=rank,
        )


def read_pair_states(
    model: nn.Module,
    tokenizer: Any,
    pairs: list[PromptPair],
    *,
    layers: list[int],
    read_spec: ReadSpec,
    batch_size: int = 4,
) -> tuple[dict[int, Tensor], dict[int, Tensor]]:
    stack = locate_decoder_stack(model)
    return read_pair_batch(
        model,
        tokenizer,
        pairs,
        layers,
        read_spec,
        batch_size=batch_size,
        stack=stack,
    )
