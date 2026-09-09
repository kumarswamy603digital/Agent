"""Corrected keyword table used by the main model.

This is the baseline table (`rules.RULES`) with the specific defects found during
dev-split error analysis fixed. The baseline itself is deliberately left untouched
so the main-model-vs-baseline comparison stays honest.

What changed and why
--------------------
1. **Bare "gate" removed from `check_in_boarding`.** It is the single most
   overloaded token in airline support: "at the gate since 6am" (disruption),
   "no gate agent anywhere" (disruption/complaint), "connecting gate on the other
   side" (disruption), "gate-checked my stroller" (baggage). Check-in is now keyed
   on unambiguous terms (boarding pass, kiosk, check in, boarding group, scanner).
2. **Disruption vocabulary broadened** with the words real travellers use:
   scrubbed, ground stop, pushed back, deplaned, diverted, inbound aircraft,
   crew shortage, stranded, misconnect, oversold/bumped.
3. **`general_info` given its own positive vocabulary** (pets, wifi, lounge,
   terminal, meals, routes, car seat, bassinet, screens) instead of acting purely
   as the fallback bucket — the fallback role was making it absorb everything.
4. **Baggage/booking/loyalty terms extended** (duffel, carousel, gate-checked,
   lap infant, MQM, upgrade certificate, rollover).
"""

from __future__ import annotations

from typing import Dict, List

RULES_V2: Dict[str, List[tuple]] = {
    "flight_disruption": [
        (r"\b(delay|delayed|delays|cancel|cancell?ed|cancellation|cancelling)\b", 3),
        (r"\b(missed|miss|missing) (my |our |the )?(connection|connecting|flight)\b", 3),
        (r"\b(rebook|re-?book|rebooking|reaccommodat|stranded|stuck (at|in|on))\b", 3),
        (r"\b(diverted|divert|tarmac|ground stop|ground hold|scrubbed|deplaned)\b", 3),
        (r"\b(pushed back|keeps? getting pushed|inbound (aircraft|plane)|crew shortage)\b", 3),
        (r"\b(misconnect|oversold|bumped off|involuntarily denied)\b", 3),
        (r"\b(weather delay|mechanical (delay|issue)|still says on time)\b", 2),
    ],
    "baggage": [
        (r"\b(bag|bags|baggage|luggage|suitcase|duffel)\b", 3),
        (r"\b(lost|missing|delayed|damaged|soaked|torn) (bag|bags|luggage|suitcase)\b", 3),
        (r"\b(carousel|belt|claim tag|file a claim|gate[- ]checked|baggage claim)\b", 2),
        (r"\b(baggage (fee|allowance|tracker)|checked bag|carry[- ]?on)\b", 2),
        (r"\b(stroller|golf bag|overweight fee)\b", 2),
    ],
    "booking_change": [
        # Allow words between the verb and its object: "move two passengers off a
        # four-person itinerary", "push my Friday flight to Sunday".
        (r"\b(change|modify|rebook|cancel|move|switch|push|split)\b.{0,34}?\b"
         r"(flight|booking|reservation|ticket|trip|itinerary|dates?|passengers?)\b", 3),
        (r"\b(travel dates|change fee|fare difference|seat selection)\b", 2),
        (r"\b(seat|seats|seat selection|seat assignment|upgrade|first class|comfort\+?)\b", 2),
        (r"\b(reschedule|different flight|earlier flight|later flight|same[- ]day change)\b", 3),
        (r"\b(lap infant|add (my|a) (bag|newborn|daughter|son|child|passenger))\b", 3),
        (r"\b(name (change|spelling)|misspelled|split my|companion certificate)\b", 3),
        (r"\b(voucher (code|won'?t))\b", 2),
    ],
    "refund_billing": [
        (r"\b(refund|refunded|refunds|reimburse|reimbursement|money back)\b", 3),
        (r"\b(charged|charge|double ?charge|overcharg|billed|billing)\b", 3),
        (r"\b(voucher|travel credit|ecredit|e-credit)\b", 3),
        (r"\b(dispute|fare difference|difference back|compensat|chargeback)\b", 2),
    ],
    "check_in_boarding": [
        (r"\b(check[\s-]?in|checkin|checking in)\b", 3),
        (r"\b(boarding pass|mobile boarding|boarding group|paper boarding)\b", 3),
        (r"\b(kiosk|scanner|see agent|counter)\b", 2),
        (r"\b(precheck|pre[- ]check|known traveler|tsa precheck)\b", 3),
        (r"\b(can'?t check in|unable to check in|check[- ]?in unavailable)\b", 3),
        # Standby is a day-of-travel airport action handled by the boarding team.
        (r"\b(standby|stand by list|opt into standby)\b", 3),
    ],
    "loyalty_program": [
        (r"\b(skymiles|sky miles|medallion|miles|mqms?|mqd)\b", 3),
        (r"\b(silver|gold|platinum|diamond) (medallion|member|status)?\b", 2),
        (r"\b(award (flight|travel|ticket)|redeem|upgrade certificates?|rollover)\b", 3),
        (r"\b(miles (didn'?t|not|never) post|missing miles|requalif|elite)\b", 3),
        (r"\b(medallion status|my status|status (renewed|reset)|bag waiver)\b", 3),
    ],
    "complaint_feedback": [
        (r"\b(worst|terrible|horrible|awful|disgusting|unacceptable|ridiculous|"
         r"appalling|shameful|disgrace)\b", 3),
        (r"\b(rude|disrespect|mocking|dismissive|unprofessional|condescending)\b", 3),
        (r"\b(never (fly|flying)|done with|fed up|complaint|file a formal)\b", 2),
    ],
    "praise": [
        (r"\b(thank you|thanks|thx|appreciate|kudos|shout ?out|hats off)\b", 3),
        (r"\b(amazing|awesome|excellent|wonderful|incredible|fantastic|best airline)\b", 2),
        (r"\b(went above and beyond|so kind|legend|impressed|well done)\b", 2),
    ],
    "general_info": [
        (r"\b(pet|dog|cat|emotional support|service animal)\b", 3),
        (r"\b(wifi|wi-fi|internet|power outlet|seatback|screens?|entertainment)\b", 3),
        (r"\b(lounge|sky club|terminal|quiet zone|family section)\b", 3),
        (r"\b(liquids|tsa rules|car seat|bassinet|stroller policy)\b", 2),
        (r"\b(meals?|snacks?|food served|fly direct|nonstop|route)\b", 2),
        (r"\b(unaccompanied minor|doctor'?s note|trimester|pregnan|oxygen|"
         r"wheelchair|mobility)\b", 3),
        (r"\b(what'?s? your policy|what is your policy|policy on)\b", 2),
        # "Can I bring/travel with X" — an allowance question about an item rather
        # than a problem with the customer's own checked baggage.
        (r"\b(can i bring|am i allowed|do you allow)\b", 3),
        (r"\b(guitar|instrument|snowboard|skis?|surfboard|bicycle|golf clubs?|"
         r"sporting equipment)\b", 3),
    ],
}
