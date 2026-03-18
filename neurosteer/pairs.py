from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class PromptPair:
    positive: str
    negative: str


@dataclass(frozen=True)
class ReadSpec:
    layer: int
    token: str | int = "last"
    reduce: str = "none"


@dataclass(frozen=True)
class WriteSpec:
    layers: tuple[int, ...]
    token: str | int | slice = "all"
    operator: str = "add"

    @classmethod
    def from_layers(
        cls, layers: Iterable[int], token: str | int | slice = "all", operator: str = "add"
    ) -> "WriteSpec":
        return cls(layers=tuple(layers), token=token, operator=operator)


def expand_suffix_pairs(
    stems: Sequence[str],
    positive_suffixes: Sequence[str],
    negative_suffixes: Sequence[str],
) -> list[PromptPair]:
    if len(positive_suffixes) != len(negative_suffixes):
        raise ValueError("positive_suffixes and negative_suffixes must have the same length")

    pairs: list[PromptPair] = []
    for stem in stems:
        for pos_suffix, neg_suffix in zip(positive_suffixes, negative_suffixes, strict=True):
            pairs.append(PromptPair(positive=stem + pos_suffix, negative=stem + neg_suffix))
    return pairs
