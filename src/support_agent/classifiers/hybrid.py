"""Main intent model: a rule-prior + Naive-Bayes hybrid.

Why a hybrid rather than "just NB"?
----------------------------------
NB here is *weakly supervised* — its training labels come from the keyword rules,
so on its own it cannot exceed its teacher by much (see REPORT.md, weak
supervision cap). But the two models fail differently:

* Rules are high precision when a domain keyword is present ("refund", "bag",
  "miles") but fall back to a majority guess when none fires.
* NB captures softer, co-occurring vocabulary and always makes a real guess.

So we blend their probabilities: when the rules fire we trust them heavily
(weight `rule_weight`) but still mix in NB; when the rules are silent we defer
entirely to NB instead of a blind majority guess. On the golden set this lifts
accuracy from ~55% (either alone) to ~66% and, crucially, yields ONE coherent
probability distribution that drives the abstain gate and escalation policy.
"""

from __future__ import annotations

from typing import Dict, List

from .base import Prediction
from .nb import NaiveBayesClassifier
from .rules import RuleClassifier


class HybridClassifier:
    name = "hybrid_rule_nb"

    def __init__(self, rule_weight: float = 0.7, nb_alpha: float = 0.3,
                 nb_min_df: int = 1, nb_ngram_range=(1, 1)):
        self.rule_weight = rule_weight
        self.rules = RuleClassifier()
        self.nb = NaiveBayesClassifier(alpha=nb_alpha, min_df=nb_min_df,
                                       ngram_range=nb_ngram_range)
        self.labels: List[str] = []

    def fit(self, X: List[str], y: List[str]) -> "HybridClassifier":
        self.rules.fit(X, y)
        self.nb.fit(X, y)
        self.labels = sorted(set(y) | set(self.nb.classes))
        return self

    def predict(self, text: str) -> Prediction:
        rp = self.rules.predict(text).proba
        npb = self.nb.predict(text).proba
        fired = self.rules.strength(text) > 0
        a = self.rule_weight if fired else 0.0  # rules silent -> defer to NB
        labels = set(rp) | set(npb)
        blended = {l: a * rp.get(l, 0.0) + (1 - a) * npb.get(l, 0.0) for l in labels}
        z = sum(blended.values()) or 1.0
        proba = {l: v / z for l, v in blended.items()}
        label = max(proba, key=proba.get)
        return Prediction(label=label, proba=proba)

    def explain(self, text: str, top_n: int = 6):
        return self.nb.explain(text, top_n)
