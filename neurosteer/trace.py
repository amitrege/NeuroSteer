from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TraceEvent:
    phase: str
    pass_name: str
    step: int
    layer: int
    layer_label: str
    value: float | None = None
    coords: list[float] = field(default_factory=list)
    applied: bool = False
    distance: float | None = None
    note: str = ""


@dataclass
class Trace:
    events: list[TraceEvent] = field(default_factory=list)

    def clear(self) -> None:
        self.events.clear()

    def add(self, event: TraceEvent) -> None:
        self.events.append(event)

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "phase": item.phase,
                "pass_name": item.pass_name,
                "step": item.step,
                "layer": item.layer,
                "layer_label": item.layer_label,
                "value": item.value,
                "coords": item.coords,
                "applied": item.applied,
                "distance": item.distance,
                "note": item.note,
            }
            for item in self.events
        ]

    def values(self, phase: str) -> list[float]:
        return [item.value for item in self.events if item.phase == phase and item.value is not None]

    def plot(self, phase: str, title: str | None = None) -> None:
        import matplotlib.pyplot as plt

        ys = self.values(phase)
        xs = list(range(len(ys)))
        plt.figure(figsize=(8, 3))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("trace index")
        plt.ylabel(phase)
        plt.title(title or phase)
        plt.grid(alpha=0.3)
        plt.tight_layout()
