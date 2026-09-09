"""Multinomial logistic regression (softmax) — from scratch, standard library only.

Why add a learned model when the rule chain already scores well?
---------------------------------------------------------------
The hand-ordered override chain in `refined.py` reached 94.1% on the dev split but
only 80.2% on held-out test. That 13.9-point gap is the signature of hand-tuning:
an ordered if-else ladder encodes the *exact* dev phrasings rather than the general
pattern. A model that **learns weights over the same signals** should trade a little
dev accuracy for better generalization, because:

  * it weighs evidence continuously instead of first-match-wins,
  * L2 regularization explicitly penalises over-confident reliance on any one cue,
  * conflicting cues are resolved by learned magnitude rather than by my hand-chosen
    ordering, which was itself fitted to dev.

Implementation notes
--------------------
* Full-batch gradient descent with L2. Deterministic (no shuffling, fixed init at
  zero), so results are bit-stable across runs and machines.
* Sparse features: each example is a {feature_name: value} dict; weights are
  {class: {feature: weight}}. This keeps it fast without numpy.
* Trained with soft targets when available (knowledge distillation), which transfers
  more information per example than hard labels.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

from .base import Prediction

Feature = Dict[str, float]


class LogisticRegression:
    name = "logreg"

    def __init__(self, classes: Sequence[str] = (), lr: float = 0.5,
                 epochs: int = 200, l2: float = 1e-4, verbose: bool = False):
        self.classes: List[str] = list(classes)
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.verbose = verbose
        self.w: Dict[str, Dict[str, float]] = {}
        self.b: Dict[str, float] = {}

    # ------------------------------------------------------------------ utils
    def _scores(self, x: Feature) -> Dict[str, float]:
        out = {}
        for c in self.classes:
            wc = self.w[c]
            s = self.b[c]
            for f, v in x.items():
                wv = wc.get(f)
                if wv is not None:
                    s += wv * v
            out[c] = s
        return out

    @staticmethod
    def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
        mx = max(scores.values())
        exps = {c: math.exp(s - mx) for c, s in scores.items()}
        z = sum(exps.values()) or 1.0
        return {c: v / z for c, v in exps.items()}

    # -------------------------------------------------------------------- fit
    def fit(self, X: List[Feature], y: List[str],
            soft_targets: Optional[List[Dict[str, float]]] = None,
            sample_weight: Optional[List[float]] = None) -> "LogisticRegression":
        """Fit by full-batch gradient descent.

        `soft_targets` (optional) lets us distil a teacher's full probability
        distribution instead of only its argmax label.
        """
        if not self.classes:
            self.classes = sorted(set(y))
        self.w = {c: defaultdict(float) for c in self.classes}
        self.b = {c: 0.0 for c in self.classes}

        n = len(X)
        if n == 0:
            return self
        weights = sample_weight or [1.0] * n
        wsum = sum(weights) or 1.0

        # Precompute the target distribution per example.
        targets: List[Dict[str, float]] = []
        for i in range(n):
            if soft_targets is not None:
                t = soft_targets[i]
                # renormalise defensively
                z = sum(t.get(c, 0.0) for c in self.classes) or 1.0
                targets.append({c: t.get(c, 0.0) / z for c in self.classes})
            else:
                targets.append({c: (1.0 if c == y[i] else 0.0) for c in self.classes})

        for epoch in range(self.epochs):
            gw = {c: defaultdict(float) for c in self.classes}
            gb = {c: 0.0 for c in self.classes}
            loss = 0.0
            for i, x in enumerate(X):
                p = self._softmax(self._scores(x))
                t = targets[i]
                sw = weights[i]
                for c in self.classes:
                    diff = (p[c] - t[c]) * sw
                    if diff:
                        gb[c] += diff
                        gc = gw[c]
                        for f, v in x.items():
                            gc[f] += diff * v
                    if t[c] > 0:
                        loss -= sw * t[c] * math.log(max(p[c], 1e-12))
            # gradient step with L2
            for c in self.classes:
                wc = self.w[c]
                gc = gw[c]
                for f, g in gc.items():
                    wc[f] -= self.lr * (g / wsum + self.l2 * wc[f])
                self.b[c] -= self.lr * (gb[c] / wsum)
            if self.verbose and (epoch % 25 == 0 or epoch == self.epochs - 1):
                print(f"  epoch {epoch:4d}  loss={loss / wsum:.4f}")
        # freeze defaultdicts into plain dicts
        self.w = {c: dict(v) for c, v in self.w.items()}
        return self

    # ---------------------------------------------------------------- predict
    def predict_proba(self, x: Feature) -> Dict[str, float]:
        return self._softmax(self._scores(x))

    def predict(self, x: Feature) -> Prediction:
        proba = self.predict_proba(x)
        return Prediction(label=max(proba, key=proba.get), proba=proba)

    def top_features(self, cls: str, k: int = 10):
        """Most positive weights for a class — used for model auditing."""
        return sorted(self.w.get(cls, {}).items(), key=lambda kv: kv[1], reverse=True)[:k]

    # ------------------------------------------------------------- persistence
    def state(self) -> dict:
        return {"classes": self.classes, "lr": self.lr, "epochs": self.epochs,
                "l2": self.l2, "w": self.w, "b": self.b}

    def load_state(self, d: dict) -> "LogisticRegression":
        self.classes = d["classes"]
        self.lr = d["lr"]; self.epochs = d["epochs"]; self.l2 = d["l2"]
        self.w = d["w"]; self.b = d["b"]
        return self
