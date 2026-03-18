from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


def _mix_vector(rank: int, mix: Tensor | list[float] | tuple[float, ...] | None, device: torch.device) -> Tensor:
    if mix is None:
        base = torch.zeros(rank, device=device)
        base[0] = 1.0
        return base
    as_tensor = torch.as_tensor(mix, dtype=torch.float32, device=device)
    if as_tensor.numel() != rank:
        raise ValueError(f"mix has {as_tensor.numel()} entries but rank is {rank}")
    return as_tensor


def summarize(coords: Tensor, mode: str, index: int = 0) -> Tensor:
    if mode == "first":
        return coords[:, index]
    if mode == "norm":
        return torch.linalg.norm(coords, dim=-1)
    if mode == "mean":
        return coords.mean(dim=-1)
    raise ValueError(f"unsupported summary mode {mode!r}")


@dataclass
class RuleOutput:
    score: Tensor
    strength: Tensor
    coefficients: Tensor


@dataclass
class ConstantRule:
    value: float | list[float] | tuple[float, ...] | Tensor

    def make(self, *, rank: int, batch_size: int, device: torch.device) -> RuleOutput:
        value = torch.as_tensor(self.value, dtype=torch.float32, device=device)
        if value.ndim == 0:
            coefficients = torch.zeros(batch_size, rank, device=device)
            coefficients[:, 0] = value
            strength = coefficients[:, 0]
        else:
            if value.numel() != rank:
                raise ValueError(f"constant vector has {value.numel()} entries but rank is {rank}")
            coefficients = value.reshape(1, rank).repeat(batch_size, 1)
            strength = torch.linalg.norm(coefficients, dim=-1)
        score = strength.clone()
        return RuleOutput(score=score, strength=strength, coefficients=coefficients)


@dataclass
class ThresholdRule:
    threshold: float
    on_strength: float = 1.0
    off_strength: float = 0.0
    direction: str = "below"
    summary: str = "first"
    index: int = 0
    mix: Tensor | list[float] | tuple[float, ...] | None = None

    def evaluate(self, coords: Tensor, *, rank: int) -> RuleOutput:
        score = summarize(coords, mode=self.summary, index=self.index)
        if self.direction == "below":
            active = score < self.threshold
        elif self.direction == "above":
            active = score > self.threshold
        else:
            raise ValueError(f"unsupported direction {self.direction!r}")

        strength = torch.where(
            active,
            torch.full_like(score, self.on_strength),
            torch.full_like(score, self.off_strength),
        )
        mix = _mix_vector(rank, self.mix, coords.device)
        coefficients = strength.unsqueeze(-1) * mix.reshape(1, rank)
        return RuleOutput(score=score, strength=strength, coefficients=coefficients)


@dataclass
class LinearRule:
    slope: float = 1.0
    bias: float = 0.0
    summary: str = "first"
    index: int = 0
    low: float | None = None
    high: float | None = None
    mix: Tensor | list[float] | tuple[float, ...] | None = None

    def evaluate(self, coords: Tensor, *, rank: int) -> RuleOutput:
        score = summarize(coords, mode=self.summary, index=self.index)
        strength = self.slope * score + self.bias
        if self.low is not None or self.high is not None:
            strength = torch.clamp(
                strength,
                min=self.low if self.low is not None else None,
                max=self.high if self.high is not None else None,
            )
        mix = _mix_vector(rank, self.mix, coords.device)
        coefficients = strength.unsqueeze(-1) * mix.reshape(1, rank)
        return RuleOutput(score=score, strength=strength, coefficients=coefficients)


@dataclass
class DiagonalGuard:
    center: Tensor
    scale: Tensor
    cutoff: float

    @classmethod
    def fit(cls, coords: Tensor, *, cutoff: float = 3.0, eps: float = 1e-5) -> "DiagonalGuard":
        center = coords.mean(dim=0)
        scale = coords.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(center=center, scale=scale, cutoff=cutoff)

    def distance(self, coords: Tensor) -> Tensor:
        normalized = (coords - self.center.to(coords.device)) / self.scale.to(coords.device)
        return torch.linalg.norm(normalized, dim=-1)

    def allow(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        distance = self.distance(coords)
        allowed = distance <= self.cutoff
        return allowed, distance
