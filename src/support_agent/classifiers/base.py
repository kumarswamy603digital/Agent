"""Shared prediction datatype for all classifiers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Prediction:
    label: str
    proba: Dict[str, float] = field(default_factory=dict)

    @property
    def confidence(self) -> float:
        return self.proba.get(self.label, 0.0) if self.proba else 0.0

    def top(self, k: int = 2) -> List[Tuple[str, float]]:
        return sorted(self.proba.items(), key=lambda kv: kv[1], reverse=True)[:k]

    @property
    def margin(self) -> float:
        t = self.top(2)
        if len(t) < 2:
            return t[0][1] if t else 0.0
        return t[0][1] - t[1][1]
