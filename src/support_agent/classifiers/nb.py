"""Main intent model: Multinomial Naive Bayes over uni+bigram features.

Chosen over a from-scratch logistic regression because:
* It trains in one pass and is trivially reproducible (no SGD seeding issues).
* Its class-conditional log-probs are inspectable, which matters for a system
  we're asking a team to *trust* — every prediction can be explained by the
  top contributing tokens.

Probabilities are produced with a numerically-stable softmax over class
log-scores, then used downstream for the abstain/escalation threshold.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Dict, List

from ..text import featurize
from .base import Prediction


class NaiveBayesClassifier:
    name = "naive_bayes"

    def __init__(self, alpha: float = 0.1, min_df: int = 2, ngram_range=(1, 2)):
        self.alpha = alpha
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.classes: List[str] = []
        self.log_prior: Dict[str, float] = {}
        self.log_likelihood: Dict[str, Dict[str, float]] = {}
        self.default_ll: Dict[str, float] = {}   # unseen-feature log-prob per class
        self.vocab: set = set()

    # ------------------------------------------------------------------ fit
    def fit(self, X: List[str], y: List[str]) -> "NaiveBayesClassifier":
        assert len(X) == len(y)
        n = len(X)
        class_docs: Counter = Counter(y)
        self.classes = sorted(class_docs)

        # document frequency for min_df pruning
        df: Counter = Counter()
        featurized: List[List[str]] = []
        for doc in X:
            feats = featurize(doc, self.ngram_range)
            featurized.append(feats)
            for f in set(feats):
                df[f] += 1
        self.vocab = {f for f, c in df.items() if c >= self.min_df}

        # per-class feature counts
        counts: Dict[str, Counter] = {c: Counter() for c in self.classes}
        totals: Dict[str, int] = {c: 0 for c in self.classes}
        for feats, label in zip(featurized, y):
            cc = counts[label]
            for f in feats:
                if f in self.vocab:
                    cc[f] += 1
                    totals[label] += 1

        V = len(self.vocab)
        self.log_prior = {c: math.log(class_docs[c] / n) for c in self.classes}
        self.log_likelihood = {}
        self.default_ll = {}
        for c in self.classes:
            denom = totals[c] + self.alpha * V
            self.log_likelihood[c] = {
                f: math.log((counts[c][f] + self.alpha) / denom) for f in self.vocab
            }
            self.default_ll[c] = math.log(self.alpha / denom) if denom > 0 else 0.0
        return self

    # -------------------------------------------------------------- predict
    def _log_scores(self, text: str) -> Dict[str, float]:
        feats = [f for f in featurize(text, self.ngram_range) if f in self.vocab]
        scores = {}
        for c in self.classes:
            ll = self.log_likelihood[c]
            default = self.default_ll[c]
            s = self.log_prior[c]
            for f in feats:
                s += ll.get(f, default)
            scores[c] = s
        return scores

    def predict(self, text: str) -> Prediction:
        scores = self._log_scores(text)
        if not scores:
            return Prediction(label=self.classes[0], proba={c: 1.0 / len(self.classes) for c in self.classes})
        mx = max(scores.values())
        exps = {c: math.exp(s - mx) for c, s in scores.items()}
        z = sum(exps.values()) or 1.0
        proba = {c: v / z for c, v in exps.items()}
        label = max(proba, key=proba.get)
        return Prediction(label=label, proba=proba)

    def explain(self, text: str, top_n: int = 6):
        """Return the features that most pushed toward the predicted class."""
        pred = self.predict(text)
        c = pred.label
        ll = self.log_likelihood[c]
        feats = [f for f in featurize(text, self.ngram_range) if f in self.vocab]
        contrib = sorted(((ll.get(f, self.default_ll[c]), f) for f in set(feats)), reverse=True)
        return pred, [f for _, f in contrib[:top_n]]

    # ------------------------------------------------------------ persistence
    def to_json(self) -> str:
        return json.dumps(
            {
                "alpha": self.alpha,
                "min_df": self.min_df,
                "ngram_range": list(self.ngram_range),
                "classes": self.classes,
                "log_prior": self.log_prior,
                "log_likelihood": self.log_likelihood,
                "default_ll": self.default_ll,
                "vocab": sorted(self.vocab),
            }
        )

    @classmethod
    def from_json(cls, s: str) -> "NaiveBayesClassifier":
        d = json.loads(s)
        m = cls(alpha=d["alpha"], min_df=d["min_df"], ngram_range=tuple(d["ngram_range"]))
        m.classes = d["classes"]
        m.log_prior = d["log_prior"]
        m.log_likelihood = d["log_likelihood"]
        m.default_ll = d["default_ll"]
        m.vocab = set(d["vocab"])
        return m
