"""Trivial baseline: always predict the majority class from training data."""

from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from .base import Prediction


class MajorityClassifier:
    name = "majority"

    def __init__(self):
        self.majority = None
        self.labels: List[str] = []

    def fit(self, X: List[str], y: List[str]) -> "MajorityClassifier":
        c = Counter(y)
        self.labels = sorted(c)
        self.majority = c.most_common(1)[0][0]
        return self

    def predict(self, text: str) -> Prediction:
        proba = {lab: (1.0 if lab == self.majority else 0.0) for lab in self.labels}
        return Prediction(label=self.majority, proba=proba)
