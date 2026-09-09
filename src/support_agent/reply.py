"""Grounded reply drafting.

Strategy: retrieval-augmented, brand-voiced, and *conservative*. A support
reply that promises the wrong thing is worse than one that safely routes the
customer. So the drafter:

  1. Retrieves the brand's real historical resolutions for similar asks.
  2. Builds an intent-specific scaffold in Delta's public voice (empathy ->
     concrete next step -> hand-off channel).
  3. Grounds the "next step" in what the retrieved resolutions actually did
     (e.g. "send us your confirmation # via DM").
  4. Passes the assembled draft through the LLM backend. With the default
     backend this is a no-op smoothing pass (deterministic); with a hosted
     backend it rewrites the scaffold into fluent prose while staying grounded.

Every draft carries the evidence (retrieved exemplars) so a reviewer can audit
what it was grounded in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

from .llm.backend import LLMBackend, get_backend
from .retriever import Exemplar, ResolutionRetriever

# Intent-specific scaffolds: empathy/opening + {evidence action} + hand-off channel.
# The fixed text carries NO action promise so the grounded {evidence} clause (a
# "we'll .../we can ..." promise pulled from real history) is not duplicated.
SCAFFOLDS = {
    "flight_disruption": (
        "So sorry for the disruption to your travel plans — that's stressful. "
        "{evidence} Please DM us your confirmation number so we can help."
    ),
    "baggage": (
        "We're sorry to hear about your bag. {evidence} Please DM us your file "
        "reference or confirmation number to get started."
    ),
    "booking_change": (
        "Happy to help with your reservation. {evidence} Please DM us your "
        "confirmation number and the change you'd like."
    ),
    "refund_billing": (
        "We understand — billing issues are stressful. {evidence} Please DM us "
        "your confirmation number and the charge details."
    ),
    "check_in_boarding": (
        "Sorry you're hitting a snag checking in. {evidence} If it keeps "
        "happening, DM us your confirmation number."
    ),
    "loyalty_program": (
        "Thanks for being a SkyMiles member. {evidence} Please DM us your "
        "SkyMiles number and the flight details."
    ),
    "complaint_feedback": (
        "We're truly sorry about your experience — this isn't the standard we "
        "aim for. {evidence} Please DM us the details so the right team can follow up."
    ),
    "praise": (
        "Thank you so much for the kind words — we'll be sure to pass this along "
        "to the team! ✈️ Safe travels."
    ),
    "general_info": (
        "Great question! {evidence} You can find the full policy details on "
        "delta.com, and we're here if anything's unclear."
    ),
    "uncertain": (
        "Thanks for reaching out. So we can point you to the right place, could "
        "you share a little more detail? A team member can also help — please DM us."
    ),
}

# Short grounded "action promise" phrases used when retrieval is weak/absent.
FALLBACK_EVIDENCE = {
    "flight_disruption": "We can look at the next available flights and rebooking options.",
    "baggage": "We can open or check a delayed-baggage report for you.",
    "booking_change": "We can review change or cancellation options on your fare.",
    "refund_billing": "We can check the status of your refund or travel credit.",
    "check_in_boarding": "Try checking in again on the Fly Delta app first.",
    "loyalty_program": "We can look into miles that haven't posted yet.",
    "complaint_feedback": "",
    "praise": "",
    "general_info": "",
    "uncertain": "",
}

SYSTEM_PROMPT = (
    "You are a Delta Air Lines social-care agent. Rewrite the DRAFT into a warm, "
    "concise public tweet reply (<=280 chars). Stay strictly grounded in the "
    "DRAFT and EVIDENCE; do not invent policies, compensation, timelines, or "
    "confirmation details. Never ask for full card numbers or passwords. Keep "
    "Delta's calm, empathetic voice."
)


@dataclass
class DraftedReply:
    text: str
    intent: str
    evidence: List[Exemplar] = field(default_factory=list)
    grounded: bool = False


def _summarize_resolution(res: str) -> str:
    """Extract the grounded *action promise* ('we'll .../we can ...') from a
    historical resolution, stripping DM/confirmation-number boilerplate and the
    agent signature (^AB) so it doesn't duplicate the scaffold's channel ask."""
    res = re.sub(r"http\S+", "", res or "")
    res = re.sub(r"\s*\^[A-Z]{2}\b", "", res).strip()  # drop agent sig
    # Pull the clause that states what Delta will do.
    m = re.search(r"\b(we'?ll|we can|we'?d)\b[^.!?]*[.!?]?", res, re.I)
    if m:
        clause = m.group(0).strip()
        # Trim leading "and " if the match started mid-sentence.
        clause = re.sub(r"^and\s+", "", clause, flags=re.I).strip()
        # Capitalize first letter, ensure it ends with a period.
        if clause:
            clause = clause[0].upper() + clause[1:]
            if not clause.endswith((".", "!", "?")):
                clause += "."
            if 8 <= len(clause) <= 160:
                return clause
    return ""


class ReplyDrafter:
    def __init__(self, retriever: ResolutionRetriever, backend: LLMBackend = None,
                 retrieval_k: int = 3, min_retrieval_sim: float = 0.08):
        self.retriever = retriever
        self.backend = backend or get_backend()
        self.k = retrieval_k
        self.min_sim = min_retrieval_sim

    def draft(self, text: str, intent: str) -> DraftedReply:
        exemplars = self.retriever.query(
            text, k=self.k, intent=None if intent == "uncertain" else intent,
            min_sim=self.min_sim,
        )
        grounded = False
        evidence_line = FALLBACK_EVIDENCE.get(intent, "")
        if exemplars:
            top = exemplars[0]
            # Only borrow a historical action promise when the exemplar is actually
            # about the same thing. The retriever backs off to unfiltered results
            # when an intent has few matches, which previously let (for example) a
            # SkyMiles resolution supply the next step for a baggage-policy question.
            same_intent = (intent == "uncertain") or (top.intent == intent)
            if same_intent and top.similarity >= self.min_sim:
                summ = _summarize_resolution(top.resolution_text)
                if summ:
                    evidence_line = summ
                    grounded = True

        scaffold = SCAFFOLDS.get(intent, SCAFFOLDS["uncertain"])
        draft = scaffold.format(evidence=evidence_line).replace("  ", " ").strip()

        # Route through backend (default backend: deterministic passthrough of `draft`).
        control = json.dumps({"draft": draft})
        user = (
            f"DRAFT:\n{draft}\n\nEVIDENCE (historical resolutions):\n"
            + "\n".join(f"- {e.resolution_text}" for e in exemplars[:2])
            + f"\n\n<<CONTROL>>{control}<<END>>"
        )
        try:
            out = self.backend.complete(SYSTEM_PROMPT, user, max_tokens=180)
            final = out.strip() or draft
        except Exception:
            final = draft
        # Hard safety clamp on length for a tweet reply.
        if len(final) > 300:
            final = final[:297].rsplit(" ", 1)[0] + "..."
        return DraftedReply(text=final, intent=intent, evidence=exemplars, grounded=grounded)
