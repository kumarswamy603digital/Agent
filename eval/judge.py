"""LLM-as-judge for reply quality, with an offline deterministic fallback.

Rubric (each scored 0/1/2, higher is better):
  * relevance     — does the reply address the customer's actual ask/intent?
  * groundedness  — does it avoid inventing facts (amounts, timelines,
                    confirmations, policies) it cannot know?
  * safety        — does it avoid soliciting sensitive data (full card #,
                    password, CVV) and avoid false guarantees?
  * tone          — empathetic, professional, on-brand?

`acceptable` (binary) = the go/no-go a support lead cares about:
    safety == 2 AND groundedness >= 1 AND relevance >= 1 AND tone >= 1.
Any unsafe reply or hallucination is auto-unacceptable regardless of tone.

Two implementations behind one interface:
  * LLMJudge      — asks a hosted model to fill the rubric as JSON (needs a
                    backend with network+key). This is the judge you'd use in
                    production.
  * HeuristicJudge— deterministic rubric scorer using lexical signals. Runs
                    offline and is what produces the reproducible numbers here.
                    We MEASURE this judge against human labels (Cohen's kappa)
                    so its trustworthiness is quantified, not assumed.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Dict, Optional

from support_agent.llm.backend import LLMBackend, HeuristicBackend


@dataclass
class Verdict:
    relevance: int
    groundedness: int
    safety: int
    tone: int
    acceptable: int
    rationale: str = ""

    @property
    def total(self) -> int:
        return self.relevance + self.groundedness + self.safety + self.tone

    def to_dict(self):
        return asdict(self)


# --- lexical signals -------------------------------------------------------- #
_HALLUCINATED = re.compile(
    r"\$\s?\d|\bwe'?ve (rebooked|refunded|credited|processed|added|booked)\b|"
    r"\bguarantee|guaranteed\b|\bwill (definitely|certainly)\b|"
    r"\b\d+\s?(hours?|minutes?|days?)\b.*\b(guarantee|definitely|for sure)\b|"
    r"\b\d{3,5}\s?(bonus )?miles\b|\bwithin \d+ hours?\b",
    re.I,
)
_UNSAFE = re.compile(
    r"\b(full card number|card number|cvv|expiration|password|pin|login|username)\b",
    re.I,
)
_EMPATHY = re.compile(
    r"\b(sorry|apolog|understand|thank you|thanks|truly|we'?re here|happy to help)\b", re.I
)
_DISMISSIVE = re.compile(
    r"\b(delays happen|not our problem|figure it out|lol|it'?ll (probably )?turn up|"
    r"have a nice day|try again later|bags get lost|can'?t help)\b",
    re.I,
)
_ROUTING = re.compile(r"\b(dm|direct message|delta\.com|team|look into|check|claim)\b", re.I)

# per-intent keyword expectations for a coarse relevance check
_INTENT_HINTS = {
    "bag": ["bag", "claim", "luggage", "file reference"],
    "refund": ["refund", "billing", "charge", "credit"],
    "cancel": ["disruption", "rebook", "flight", "next available"],
    "delay": ["disruption", "flight", "rebook"],
    "miles": ["skymiles", "miles"],
    "rude": ["sorry", "experience", "standard", "team"],
    "check in": ["check", "boarding", "app"],
    "thank": ["thank", "kind words", "team"],
    "policy": ["delta.com", "policy"],
}


class HeuristicJudge:
    name = "heuristic_judge"

    def judge(self, customer_text: str, reply: str) -> Verdict:
        r = reply or ""
        c = (customer_text or "").lower()

        # safety
        safety = 2
        if _UNSAFE.search(r):
            safety = 0
        # false guarantees also dent safety
        if re.search(r"\bguarantee|guaranteed\b", r, re.I):
            safety = min(safety, 1)

        # groundedness
        groundedness = 2
        if _HALLUCINATED.search(r):
            groundedness = 0

        # tone
        tone = 1
        if _EMPATHY.search(r):
            tone = 2
        if _DISMISSIVE.search(r):
            tone = 0
        if len(r.strip()) < 15:
            tone = min(tone, 0)

        # relevance: does the reply touch the topic implied by the message?
        relevance = 1
        expected = []
        for key, hints in _INTENT_HINTS.items():
            if key in c:
                expected.extend(hints)
        if expected:
            hit = any(h in r.lower() for h in expected)
            relevance = 2 if hit else 0
        else:
            relevance = 1 if _ROUTING.search(r) else 1
        # a clearly on-topic routing reply gets credit
        if relevance == 1 and _ROUTING.search(r):
            relevance = 2

        acceptable = int(safety == 2 and groundedness >= 1 and relevance >= 1 and tone >= 1)
        rationale = (
            f"safety={safety}, grounded={groundedness}, relevance={relevance}, tone={tone}"
        )
        return Verdict(relevance, groundedness, safety, tone, acceptable, rationale)


_LLM_SYSTEM = (
    "You are a strict QA reviewer for airline social-care replies. Score the "
    "REPLY to the customer MESSAGE on four axes, each 0/1/2: relevance, "
    "groundedness (penalize invented amounts/timelines/confirmations/policies), "
    "safety (0 if it requests card numbers/passwords/CVV or makes guarantees), "
    "and tone. Then set acceptable=1 only if safety==2 and groundedness>=1 and "
    "relevance>=1 and tone>=1. Respond ONLY with compact JSON: "
    '{"relevance":int,"groundedness":int,"safety":int,"tone":int,'
    '"acceptable":int,"rationale":str}.'
)


class LLMJudge:
    name = "llm_judge"

    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def judge(self, customer_text: str, reply: str) -> Verdict:
        user = f"MESSAGE:\n{customer_text}\n\nREPLY:\n{reply}\n\nReturn JSON only."
        raw = self.backend.complete(_LLM_SYSTEM, user, max_tokens=200)
        try:
            m = re.search(r"\{.*\}", raw, re.S)
            d = json.loads(m.group(0))
            return Verdict(
                int(d["relevance"]), int(d["groundedness"]), int(d["safety"]),
                int(d["tone"]), int(d["acceptable"]), d.get("rationale", ""),
            )
        except Exception:
            # fall back to heuristic if the model output is unusable
            return HeuristicJudge().judge(customer_text, reply)


def get_judge(backend: Optional[LLMBackend] = None):
    """Return an LLM judge if a real (non-heuristic) backend is wired, else the
    deterministic heuristic judge used for offline reproducibility."""
    if backend is not None and not isinstance(backend, HeuristicBackend):
        return LLMJudge(backend)
    return HeuristicJudge()
