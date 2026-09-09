"""Linguistic signal detectors used to disambiguate intents.

Motivation (from error analysis on the dev split)
-------------------------------------------------
A bag-of-words model keys on the most salient *noun* in a tweet, which causes three
systematic error classes:

1. **Complaints get absorbed by nouns.** "your gate agent was openly mocking a
   disabled passenger" contains "gate" → check_in_boarding. But a complaint is
   defined by *evaluative stance toward the service*, not by its nouns. So we detect
   complaint tone explicitly.

2. **Policy questions look like account actions.** "do you allow snowboards as
   checked baggage" (policy → general_info) and "my bag never arrived" (action →
   baggage) share the word "bag". The distinguishing feature is grammatical:
   a *generic question about the airline's rules* vs. a *first-person reference to
   the customer's own trip*. So we detect question framing and account scope.

3. **Off-topic / contentless messages** ("?", "following for the giveaway") have no
   in-vocabulary signal at all and should not be confidently routed anywhere.

These detectors are deliberately transparent regex/lexicon features rather than a
learned model: they are auditable, they need no labelled data, and they encode
domain knowledge a support lead can review and edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# 1. Complaint / evaluative-negative stance                                    #
# --------------------------------------------------------------------------- #
# Judgment words applied to the airline or its people.
_EVALUATIVE_NEG = re.compile(
    r"\b(unacceptable|disgrace|disgust|appall|shameful|shame|awful|terrible|horrible|"
    r"horrendous|worst|ridiculous|absurd|outrageous|outrage|humiliat|disrespect|"
    r"insult|degrading|abysmal|pathetic|atrocious|furious|fuming|livid|"
    r"unprofessional|incompetent|disorganiz|filthy|disgraceful|joke\b|"
    r"nickel and dim|treated like|never (again|flying|fly with)|done with (you|delta)|"
    r"fed up|had enough|last time i fly)",
    re.I,
)
# Staff-conduct descriptions (behaviour of a person, not a system).
_STAFF_CONDUCT = re.compile(
    r"\b(rude|mocking|mocked|snapped at|yelled|shouted|rolled (her|his|their) eyes|"
    r"dismissive|condescending|ignored (me|us)|laughed at|refused to help|"
    r"wouldn'?t help|no one (cared|helped|would help)|nobody (cared|helped|acknowledg)|"
    r"spoke to my|talked to me like|treated (me|us|my))",
    re.I,
)
# Service-failure narrations (something was done TO the customer).
_SERVICE_FAILURE = re.compile(
    r"\b(no (explanation|announcement|updates?|information|food|water|air condition)|"
    r"without (asking|warning|notice|explanation)|left (us|me|an?|my|the) .{0,30}"
    r"(sitting|waiting|stranded|unattended)|unattended|"
    r"no one (explained|told us|announced|would help|will help)|take responsibility|"
    r"nearly missed|no staff|not a single (agent|person))",
    re.I,
)
# Physical harm or a formal grievance — always a complaint, never a product query.
_HARM_OR_GRIEVANCE = re.compile(
    r"\b(injur|was hurt|got hurt|hit me|fell on me|falling bag|"
    r"legal action|lawsuit|sue|attorney|lawyer|"
    r"how (my|our) .{0,30}(was|were) handled|way (my|our|your) .{0,20}(was|were))",
    re.I,
)

# --------------------------------------------------------------------------- #
# 2. Question framing vs. account scope                                        #
# --------------------------------------------------------------------------- #
# Generic questions about the airline's rules/products (→ general_info).
_POLICY_QUESTION = re.compile(
    r"\b(do you (allow|offer|have|fly|serve|provide)|are there|is there|"
    r"what'?s? (the|your) (policy|limit|rule|cutoff|deal|process|youngest|oldest)|"
    r"what is (the|your) (policy|limit|rule|process)|"
    r"how much (is|are|do you charge|does it cost)|"
    r"am i allowed|can i (bring|carry|take|travel|use)\b|"
    r"is .{0,30}(included|considered|counted|available)|"
    r"do .{0,25}(count|apply|get)\b|"
    r"what (terminal|time|age)|how early|do i need|does delta (allow|offer|have))",
    re.I,
)
# Reference to the customer's own booking/account (→ an action intent).
_ACCOUNT_SCOPE = re.compile(
    r"\b(my (reservation|booking|itinerary|ticket|account|confirmation|trip|fare|"
    r"seat assignment|upgrade|status|miles|skymiles|bag|luggage|suitcase|refund|"
    r"credit|ecredit|voucher|flight|connection|pass|card)|"
    r"our (booking|reservation|itinerary|flight)|"
    r"existing reservation|already (made|checked in|booked)|"
    r"conf(irmation)? ?[#:]?\s?[a-z0-9]{5,}|case ?#)",
    re.I,
)
# "how do I ..." — a how-to, which we route by *topic*, not to general_info.
_HOWTO = re.compile(
    r"\b(how do i|how can i|how would i|where do i|what'?s the best way to)\b", re.I
)

# --------------------------------------------------------------------------- #
# 3. Money / transactional asks (outrank complaint tone)                       #
# --------------------------------------------------------------------------- #
# A genuine request to move money — this is the head of the customer's ask.
_REFUND_REQUEST = re.compile(
    r"\b(refund|reimburse|reimbursement|money back|difference back|chargeback|"
    r"double ?charg|overcharg|charged twice|billed twice|needs? reversing|"
    r"want .{0,20}back|compensat)",
    re.I,
)
# Money merely *mentioned* while asking for something else ("change my flight but
# the fare difference errors out"). Must NOT hijack the intent.
_MONEY_MENTION = re.compile(
    r"\b(charge|charged|charges|billed|billing|fare difference|voucher|ecredit|"
    r"e-credit|travel credit|credit|fee|fees|cost|price)",
    re.I,
)
# Verbs that mark an action on an existing/new booking.
_BOOKING_ACTION = re.compile(
    r"\b(change|changed|changing|move|moving|switch|switching|push|cancel|"
    r"rebook|reschedule|split|add|book|booking|select|selection|selected|"
    r"upgrade|travel dates)\b",
    re.I,
)

# --------------------------------------------------------------------------- #
# 4. Off-topic / contentless                                                   #
# --------------------------------------------------------------------------- #
_OFFTOPIC = re.compile(
    r"\b(stock price|giveaway|follow(ing)? (for|back)|pick me|contest|sweepstake|"
    r"official account|is this a bot|are you a bot|promo code|sponsor)",
    re.I,
)
# A message is treated as "contentless" only when it is genuinely too short to
# route (e.g. "?", "🔥🔥🔥"). An earlier version tested membership in a keyword
# list, which wrongly flagged real messages that used vocabulary outside the list
# ("our plane diverted...", "flt 2281 just got scrubbed"). Length is the honest
# signal for "there is nothing here to classify".
_WORDISH = re.compile(r"[a-z']{2,}", re.I)
_MIN_CONTENT_TOKENS = 3

# --------------------------------------------------------------------------- #
# 5. Sarcasm (positive words + negative frame) — protects `praise`             #
# --------------------------------------------------------------------------- #
_POSITIVE = re.compile(
    r"\b(thank|thanks|thx|appreciate|amazing|awesome|excellent|wonderful|incredible|"
    r"fantastic|great|best|kudos|shout ?out|hats off|legend|impressed|smooth|"
    r"went above|so kind|professional|solved)",
    re.I,
)
# Emphatic praise. Deliberately excludes a bare "thanks", which is often just a
# sign-off on an otherwise transactional message ("false alarm, thanks anyway").
_STRONG_POSITIVE = re.compile(
    r"\b(thank you|thank (him|her|them)|amazing|awesome|excellent|wonderful|"
    r"incredible|fantastic|kudos|shout ?out|hats off|legend|impressed|"
    r"went above|went out in|so kind|best (flight|crew|service|airline|boarding)|"
    r"made my|well done)",
    re.I,
)
_SARCASM_FRAME = re.compile(
    r"\b(only took|so much for|thanks for nothing|great(,| ) another|oh (great|wonderful)|"
    r"love (it|that) when|apparently|'\w+'|\"\w+\"|not thrilled|amazing service, only)",
    re.I,
)
_NEGATION_NEAR_POS = re.compile(
    r"\b(not|never|no|zero|didn'?t|couldn'?t|wouldn'?t|failed to)\b[^.!?]{0,40}"
    r"\b(help|solve|resolve|care|acknowledg|work|arrive|show up)",
    re.I,
)


@dataclass
class Signals:
    complaint_tone: bool
    staff_conduct: bool
    service_failure: bool
    harm_or_grievance: bool
    policy_question: bool
    account_scope: bool
    howto: bool
    refund_request: bool
    money_mention: bool
    booking_action: bool
    offtopic: bool
    contentless: bool
    positive: bool
    strong_positive: bool
    sarcasm: bool
    question: bool

    @property
    def complaint_evidence(self) -> int:
        """How many independent complaint-ish signals fired."""
        return sum([self.complaint_tone, self.staff_conduct,
                    self.service_failure, self.harm_or_grievance])

    @property
    def money_ask(self) -> bool:
        """Any money involvement at all (request or mention)."""
        return self.refund_request or self.money_mention

    @property
    def pure_praise(self) -> bool:
        """Positive sentiment with no complaint, no sarcasm, and nothing to action."""
        return (self.positive and not self.sarcasm and self.complaint_evidence == 0
                and not self.refund_request and not self.question)


def analyze(text: str) -> Signals:
    t = text or ""
    # Strip the leading brand mention so "@Delta" doesn't count as content.
    body = re.sub(r"@\w+", " ", t).strip()

    positive = bool(_POSITIVE.search(t))
    complaint_tone = bool(_EVALUATIVE_NEG.search(t))
    staff_conduct = bool(_STAFF_CONDUCT.search(t))
    service_failure = bool(_SERVICE_FAILURE.search(t))
    sarcasm = bool(positive and (_SARCASM_FRAME.search(t) or _NEGATION_NEAR_POS.search(t)))

    return Signals(
        complaint_tone=complaint_tone,
        staff_conduct=staff_conduct,
        service_failure=service_failure,
        harm_or_grievance=bool(_HARM_OR_GRIEVANCE.search(t)),
        policy_question=bool(_POLICY_QUESTION.search(t)),
        account_scope=bool(_ACCOUNT_SCOPE.search(t)),
        howto=bool(_HOWTO.search(t)),
        refund_request=bool(_REFUND_REQUEST.search(t)),
        money_mention=bool(_MONEY_MENTION.search(t)),
        booking_action=bool(_BOOKING_ACTION.search(t)),
        offtopic=bool(_OFFTOPIC.search(t)),
        contentless=(len(_WORDISH.findall(body)) < _MIN_CONTENT_TOKENS),
        positive=positive,
        strong_positive=bool(_STRONG_POSITIVE.search(t)),
        sarcasm=sarcasm,
        question=("?" in t or bool(_POLICY_QUESTION.search(t)) or bool(_HOWTO.search(t))),
    )
