"""Main intent model: signal-refined rule+Naive-Bayes hybrid.

Pipeline
--------
    rule keyword scores  ─┐
                          ├─▶ blended probability ─▶ signal disambiguation ─▶ label
    Naive Bayes (weak)   ─┘

The blend is the earlier `HybridClassifier` idea (trust rules when a domain keyword
fires, defer to NB when they are silent). On top of it sits a **disambiguation
layer** driven by `signals.py`, which fixes the three systematic error classes that
bag-of-words models cannot fix by reweighting words:

  * complaint tone outranks whatever noun happens to be in the sentence,
  * generic policy questions are separated from actions on the customer's own trip,
  * contentless / off-topic messages are not confidently routed anywhere.

Every adjustment is an explicit, ordered rule with a recorded justification, so a
reviewer can audit *why* a prediction changed. `predict_verbose` returns those
justifications.

The rule ordering below was developed against the **dev split only**
(`eval/splits.py`); the test split was untouched during tuning.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..signals import Signals, analyze
from .base import Prediction
from .nb import NaiveBayesClassifier
from .rules import RuleClassifier
from .rules_refined import RULES_V2

# Topic keywords used to route a "how do I ..." question to its topic intent
# instead of the generic information bucket.
_TOPIC_HINTS: List[Tuple[str, tuple]] = [
    ("refund_billing", ("refund", "charge", "charged", "billed", "billing", "ecredit",
                        "travel credit", "voucher", "fare difference", "reimburse")),
    ("loyalty_program", ("miles", "skymiles", "medallion", "mqm", "mqms", "award",
                         "upgrade certificate", "elite", "status")),
    ("check_in_boarding", ("check in", "check-in", "checkin", "boarding pass",
                           "boarding group", "precheck", "known traveler", "kiosk",
                           "standby", "board")),
    ("baggage", ("bag", "bags", "baggage", "luggage", "suitcase", "duffel", "carousel")),
    ("booking_change", ("reservation", "booking", "itinerary", "seat", "seats",
                        "upgrade", "cancel", "change my", "rebook", "ticket",
                        "travel dates", "dates changed", "change fee",
                        "passengers", "voucher")),
    ("flight_disruption", ("delay", "delayed", "cancelled", "canceled", "rebook",
                           "connection", "misconnect", "diverted")),
]


def _topic_from_text(text: str, exclude: tuple = ()) -> Optional[str]:
    """First topic whose keywords appear in the text (priority-ordered)."""
    low = (text or "").lower()
    for intent, kws in _TOPIC_HINTS:
        if intent in exclude:
            continue
        if any(k in low for k in kws):
            return intent
    return None


class RefinedClassifier:
    name = "refined_hybrid"

    def __init__(self, rule_weight: float = 0.7, nb_alpha: float = 0.3,
                 nb_min_df: int = 1, nb_ngram_range=(1, 1)):
        self.rule_weight = rule_weight
        # Corrected keyword table (the frozen baseline table stays in rules.RULES).
        self.rules = RuleClassifier(rules_table=RULES_V2)
        self.nb = NaiveBayesClassifier(alpha=nb_alpha, min_df=nb_min_df,
                                       ngram_range=nb_ngram_range)
        self.labels: List[str] = []

    # ------------------------------------------------------------------- fit
    def fit(self, X: List[str], y: List[str]) -> "RefinedClassifier":
        self.rules.fit(X, y)
        self.nb.fit(X, y)
        self.labels = sorted(set(y) | set(self.nb.classes))
        return self

    # --------------------------------------------------------------- blending
    def _blend(self, text: str) -> Dict[str, float]:
        rp = self.rules.predict(text).proba
        npb = self.nb.predict(text).proba
        fired = self.rules.strength(text) > 0
        a = self.rule_weight if fired else 0.0
        labels = set(rp) | set(npb)
        blended = {l: a * rp.get(l, 0.0) + (1 - a) * npb.get(l, 0.0) for l in labels}
        z = sum(blended.values()) or 1.0
        return {l: v / z for l, v in blended.items()}

    # -------------------------------------------------------------- refinement
    def _refine(self, text: str, proba: Dict[str, float], s: Signals
                ) -> Tuple[str, List[str]]:
        """Ordered disambiguation. Returns (label, list of reasons applied)."""
        label = max(proba, key=proba.get)
        why: List[str] = []

        # (0) Genuinely contentless ("?", emoji-only) or off-topic (giveaways, stock
        #     price). Nothing to route on; park it in the generic bucket. The
        #     escalation policy sends these to a human regardless.
        if s.contentless or s.offtopic:
            if label != "general_info":
                why.append("contentless_or_offtopic→general_info")
            return "general_info", why

        # (1) Sarcasm is not praise. But if the customer also states a concrete
        #     operational problem ("another 'on-time' departure that's 2 hours
        #     late"), the operational intent is the more actionable label.
        if s.sarcasm:
            concrete = _topic_from_text(text, exclude=("general_info",))
            if concrete is None:
                why.append("sarcasm→complaint_feedback")
                return "complaint_feedback", why
            if concrete != label:
                why.append(f"sarcasm→{concrete}")
            return concrete, why

        # (2) An explicit request to move money is the head of the ask, so it wins.
        #     Money merely *mentioned* alongside a booking action does not — "change
        #     my return but the fare difference errors out" is a booking change.
        if s.refund_request and label != "refund_billing":
            asking_fee_policy = s.policy_question and not s.account_scope
            if not asking_fee_policy:
                why.append("refund_request→refund_billing")
                return "refund_billing", why
        if (label == "refund_billing" and not s.refund_request
                and s.money_mention and s.booking_action):
            topic = _topic_from_text(text, exclude=("general_info", "refund_billing"))
            if topic:
                why.append(f"money_mention+booking_action→{topic}")
                return topic, why

        # (3) Complaint stance outranks the salient noun ("gate", "bag", "seat").
        #     Suppressed when the customer is actually asking us to service their
        #     own booking — then the transactional intent is more useful.
        if s.complaint_evidence >= 1:
            transactional = s.account_scope and label in (
                "baggage", "booking_change", "check_in_boarding",
                "loyalty_program", "refund_billing",
            )
            if not transactional:
                if label != "complaint_feedback":
                    why.append("complaint_stance→complaint_feedback")
                return "complaint_feedback", why

        # (4) Unambiguous positive feedback with nothing to action is praise, even
        #     if it mentions a topic noun ("best boarding process of any airline").
        #     Skipped when the message narrates a concrete incident, since then the
        #     topic team still needs to see it.
        if s.strong_positive and not s.sarcasm and s.complaint_evidence == 0 \
                and not s.refund_request and not s.question and label != "praise":
            why.append("strong_positive→praise")
            return "praise", why

        # (5) Loyalty vocabulary is highly specific ("elite bag waiver", "upgrade
        #     certificates", "MQMs"). When it is present, it outranks the generic
        #     bag/booking nouns that happen to co-occur.
        if label in ("baggage", "booking_change", "check_in_boarding"):
            rs = self.rules.scores(text)
            if rs.get("loyalty_program", 0.0) >= 3.0:
                why.append("loyalty_marker→loyalty_program")
                return "loyalty_program", why

        # (5b) "How do I ..." is a request for a procedure; route it to the team that
        #      owns that procedure, using keyword strength to pick the topic.
        if s.howto:
            rs_h = self.rules.scores(text)
            best = max(rs_h, key=rs_h.get) if rs_h else None
            if best and rs_h.get(best, 0.0) > 0 and best != label:
                why.append(f"howto_topic→{best}")
                return best, why

        # (6) Questions route by *topic* when one is identifiable, falling back to
        #     general_info otherwise. Airline support treats "how much for a second
        #     checked bag" as baggage, not generic info — the topic decides who
        #     handles it. We compare keyword strength so travel-general questions
        #     ("is there a lounge in Terminal 4", "bassinet seat") stay general_info.
        rs = self.rules.scores(text)
        gi = rs.get("general_info", 0.0)
        if s.question and label == "general_info":
            topic = _topic_from_text(text, exclude=("general_info",))
            if topic and rs.get(topic, 0.0) > gi:
                why.append(f"question_topic→{topic}")
                return topic, why

        # (7) A generic policy question that landed on an action intent but never
        #     references the customer's own trip is information — provided the
        #     general-info vocabulary is at least as strong as the topic's.
        if s.policy_question and not s.account_scope and label != "general_info":
            if gi >= rs.get(label, 0.0):
                why.append("policy_question→general_info")
                return "general_info", why

        # (7) An account-scoped message sitting in general_info is an action.
        if s.account_scope and not s.policy_question and label == "general_info":
            topic = _topic_from_text(text, exclude=("general_info",))
            if topic:
                why.append(f"account_scope→{topic}")
                return topic, why

        return label, why

    # ---------------------------------------------------------------- predict
    def predict(self, text: str) -> Prediction:
        proba = self._blend(text)
        s = analyze(text)
        label, _ = self._refine(text, proba, s)
        # Keep the blended distribution but make the chosen label the mode, so
        # downstream confidence/margin logic stays coherent.
        proba = self._reproject(proba, label)
        return Prediction(label=label, proba=proba)

    def predict_verbose(self, text: str) -> Tuple[Prediction, List[str]]:
        proba = self._blend(text)
        s = analyze(text)
        label, why = self._refine(text, proba, s)
        return Prediction(label=label, proba=self._reproject(proba, label)), why

    @staticmethod
    def _reproject(proba: Dict[str, float], label: str) -> Dict[str, float]:
        """Ensure `label` holds the highest mass without discarding the shape.

        When a disambiguation rule overrides the argmax we transfer enough mass to
        the chosen label to make it the mode. This keeps `confidence`/`margin`
        meaningful for the abstain gate instead of reporting the old winner's score.
        """
        if not proba:
            return {label: 1.0}
        top = max(proba.values())
        cur = proba.get(label, 0.0)
        if cur >= top:
            return proba
        out = dict(proba)
        # Give the overridden label the leading share, scaled so it stays a
        # probability distribution.
        out[label] = top + (top - cur) * 0.5 + 1e-6
        z = sum(out.values()) or 1.0
        return {k: v / z for k, v in out.items()}

    def explain(self, text: str, top_n: int = 6):
        return self.nb.explain(text, top_n)
