"""Escalation decision: auto-handle vs. route to a human, with a stated reason.

Philosophy: escalation is a *precision-of-safety* problem, not an accuracy
problem. The cost of auto-handling something we shouldn't (a refund dispute, an
angry customer, a safety issue) is far higher than the cost of escalating
something we could have handled. So the policy is deliberately asymmetric and
conservative, and every decision returns a human-readable reason.

Signals (any one can force escalation):
  * abstain / low confidence / low margin from the intent model
  * intent is in the ESCALATE_INTENTS set (money, disruption rebooking, complaints)
  * high-urgency / distress / legal / safety language
  * explicit request for a human
  * PII exposure that must be moved to a private channel
Otherwise: auto-handle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from .classifiers.base import Prediction
from .intents import ABSTAIN

# Intents where an incorrect or unauthorized auto-reply is expensive: anything
# touching money, rebooking commitments, or brand-risk complaints.
ESCALATE_INTENTS = {
    "flight_disruption",  # rebooking requires acting on a live reservation
    "refund_billing",     # money movement / disputes
    "complaint_feedback", # brand risk, needs human empathy + service recovery
}

# Intents safe to auto-handle when confident (informational / low-stakes).
AUTO_OK_INTENTS = {
    "praise",
    "general_info",
    "loyalty_program",
    "check_in_boarding",
    "baggage",         # initial acknowledgement + tracking hand-off is safe
    "booking_change",  # acknowledge + route to DM is safe; no change committed
}

# NOTE: we intentionally avoid a *trailing* \b on stem patterns (e.g. "humiliat",
# "discriminat") so they still match inflected forms ("humiliating").
_HUMAN_REQ = re.compile(
    r"\b(speak|talk|connect|transfer|get me) (to|with|me to)? ?(a |an )?(human|agent|person|representative|rep|someone|supervisor|manager)|real person|real human",
    re.I,
)
_URGENT = re.compile(
    r"\b(emergency|urgent|asap|right now|immediately|stranded|stuck at|missed my|"
    r"about to miss|boarding now|boarding in|at security|see agent|kiosk ate|"
    r"medical|wheelchair|oxygen|mobility|disab|unaccompanied minor)",
    re.I,
)
# generic time-pressure: "in 5 minutes/min" near a flight/boarding context
_TIME_PRESSURE = re.compile(
    r"\bin \d+\s?(min|mins|minutes|hour|hours)\b.*\b(boarding|flight|gate|connection|departs?)\b|"
    r"\b(boarding|flight|gate|connection)\b.*\bin \d+\s?(min|mins|minutes)\b|"
    r"\b\d+\s?min(ute)?s?\b.*\b(boarding|connection|to make it)\b",
    re.I,
)
_LEGAL = re.compile(
    r"\b(lawyer|attorney|sue|lawsuit|legal action|dot complaint|compensation under|"
    r"eu ?261|discriminat|racist|assault|unsafe|safety)",
    re.I,
)
_DISTRESS = re.compile(
    r"\b(furious|outraged|outrage|disgust|never flying|never fly again|worst airline|"
    r"worst experience|humiliat|ruined|traumati|crying|panic|fed up|fuming)",
    re.I,
)
# property damage / physical harm / claims -> needs a human claims workflow
_CLAIM_SAFETY = re.compile(
    r"\b(injur|hurt|damaged|broken|ruined|soaked|smoke|fire|bleeding|"
    r"file a claim|reimburse|reimbursement|who handles claims)",
    re.I,
)
# account fraud / unauthorized access -> security escalation
_FRAUD = re.compile(
    r"\b(hacked|unauthorized|didn'?t authorize|did not authorize|fraud|stolen|"
    r"someone (used|booked|changed|accessed)|account was (hacked|compromised))",
    re.I,
)
# crude PII detectors (card-like numbers, emails, phone numbers)
_PII = re.compile(
    r"\b(?:\d[ -]?){13,16}\b|[\w.+-]+@[\w-]+\.[\w.-]+|\b\d{3}[.-]?\d{3}[.-]?\d{4}\b"
)


@dataclass
class Decision:
    escalate: bool
    reason: str
    signals: List[str] = field(default_factory=list)
    confidence: float = 0.0


def decide(text: str, pred: Prediction, abstain_threshold: float = 0.42,
           min_margin: float = 0.10) -> Decision:
    signals: List[str] = []
    t = text or ""

    # 1) model uncertainty
    if pred.label == ABSTAIN:
        signals.append("model_abstained")
    if pred.confidence < abstain_threshold:
        signals.append(f"low_confidence({pred.confidence:.2f}<{abstain_threshold})")
    if pred.margin < min_margin:
        signals.append(f"low_margin({pred.margin:.2f}<{min_margin})")

    # 2) explicit human request
    if _HUMAN_REQ.search(t):
        signals.append("explicit_human_request")

    # 3) safety / legal / urgency / distress / claims / fraud
    if _LEGAL.search(t):
        signals.append("legal_or_safety_language")
    if _URGENT.search(t) or _TIME_PRESSURE.search(t):
        signals.append("time_critical_or_vulnerable")
    if _DISTRESS.search(t):
        signals.append("high_distress_language")
    if _CLAIM_SAFETY.search(t):
        signals.append("property_damage_or_injury_claim")
    if _FRAUD.search(t):
        signals.append("account_fraud_or_unauthorized_access")

    # 4) PII exposed publicly -> must move to private channel (human/DM workflow)
    if _PII.search(t):
        signals.append("pii_exposed_needs_private_channel")

    # 5) high-stakes intent
    if pred.label in ESCALATE_INTENTS:
        signals.append(f"high_stakes_intent:{pred.label}")

    escalate = len(signals) > 0

    if not escalate:
        reason = (
            f"Auto-handle: confident '{pred.label}' ({pred.confidence:.2f}) in a "
            f"low-stakes category with no risk signals."
        )
    else:
        reason = "Escalate to human — " + "; ".join(signals) + "."
    return Decision(escalate=escalate, reason=reason, signals=signals, confidence=pred.confidence)
