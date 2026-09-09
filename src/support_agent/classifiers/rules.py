"""Simple baseline: hand-written keyword rules with a scored fallback.

This is the "reasonable non-ML baseline" a support team might ship in a week.
Each intent has weighted keyword/phrase patterns; the highest-scoring intent
wins. Ties and empty matches fall back to the corpus majority class.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List

from .base import Prediction

# (pattern, weight). Patterns are matched on lowercased raw text.
RULES: Dict[str, List[tuple]] = {
    "flight_disruption": [
        (r"\b(delay|delayed|cancel|cancell?ed|cancellation)\b", 3),
        (r"\b(missed|miss) (my |our |the )?(connection|flight)\b", 3),
        (r"\b(rebook|re-?book|rebooking|stranded|stuck)\b", 2),
        (r"\b(diverted|tarmac|weather delay)\b", 2),
    ],
    "baggage": [
        (r"\b(bag|bags|baggage|luggage|suitcase)\b", 3),
        (r"\b(lost|missing|delayed|damaged) (bag|bags|luggage|suitcase)\b", 3),
        (r"\b(carousel|belt|claim tag|file a claim)\b", 2),
        (r"\bbaggage fee\b", 1),
    ],
    "booking_change": [
        (r"\b(change|modify|rebook|cancel) (my |our )?(flight|booking|reservation|ticket)\b", 3),
        (r"\b(seat|seats|seat selection|upgrade|first class|comfort\+?)\b", 2),
        (r"\b(reschedule|different flight|earlier flight|later flight)\b", 2),
    ],
    "refund_billing": [
        (r"\b(refund|refunded|reimburse|money back)\b", 3),
        (r"\b(charged|charge|double charge|overcharged|billing)\b", 3),
        (r"\b(voucher|travel credit|ecredit|credit)\b", 2),
        (r"\b(dispute|fare difference)\b", 1),
    ],
    "check_in_boarding": [
        (r"\b(check[\s-]?in|checkin)\b", 3),
        (r"\b(boarding pass|mobile boarding|gate|kiosk)\b", 2),
        (r"\b(app|application) (crash|error|won'?t|not working|glitch)\b", 2),
        (r"\b(can'?t check in|unable to check in)\b", 3),
    ],
    "loyalty_program": [
        (r"\b(skymiles|sky miles|medallion|miles|points)\b", 3),
        (r"\b(status|silver|gold|platinum|diamond) (medallion|member)?\b", 1),
        (r"\b(award (flight|travel|ticket)|redeem)\b", 2),
        (r"\b(miles (didn'?t|not) post|missing miles)\b", 3),
    ],
    "complaint_feedback": [
        (r"\b(worst|terrible|horrible|awful|disgusting|unacceptable|ridiculous)\b", 2),
        (r"\b(rude|disrespectful|unprofessional) (staff|agent|crew|gate agent)\b", 3),
        (r"\b(never (fly|flying)|done with|fed up|complaint)\b", 2),
    ],
    "praise": [
        (r"\b(thank you|thanks|thx|ty|appreciate|kudos|shout ?out)\b", 3),
        (r"\b(amazing|awesome|great (flight|crew|service)|best airline|love (delta|flying delta))\b", 2),
        (r"\b(went above and beyond|excellent|wonderful)\b", 2),
    ],
    "general_info": [
        (r"\b(policy|allowed|allowance|how (do|much|many)|can i (bring|carry))\b", 2),
        (r"\b(pet|dog|cat|emotional support|service animal)\b", 2),
        (r"\b(wifi|wi-fi|internet|power outlet)\b", 2),
        (r"\b(carry[\s-]?on|checked bag size|liquids|tsa)\b", 2),
    ],
}

_COMPILED = {
    intent: [(re.compile(p), w) for p, w in pats] for intent, pats in RULES.items()
}


class RuleClassifier:
    name = "rules"

    def __init__(self):
        self.majority = None
        self.labels: List[str] = list(RULES.keys())

    def fit(self, X: List[str], y: List[str]) -> "RuleClassifier":
        self.majority = Counter(y).most_common(1)[0][0]
        self.labels = sorted(set(y) | set(RULES.keys()))
        return self

    def _score(self, text: str) -> Dict[str, float]:
        t = (text or "").lower()
        scores = {lab: 0.0 for lab in self.labels}
        for intent, pats in _COMPILED.items():
            for rx, w in pats:
                if rx.search(t):
                    scores[intent] += w
        return scores

    def strength(self, text: str) -> float:
        """Total matched keyword weight — how strongly rules 'fired'. 0 = no match."""
        return sum(self._score(text).values())

    def predict(self, text: str) -> Prediction:
        scores = self._score(text)
        total = sum(scores.values())
        if total <= 0:
            proba = {lab: (1.0 if lab == self.majority else 0.0) for lab in self.labels}
            return Prediction(label=self.majority, proba=proba)
        proba = {lab: s / total for lab, s in scores.items()}
        label = max(proba, key=proba.get)
        return Prediction(label=label, proba=proba)
