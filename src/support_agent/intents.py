"""Intent taxonomy for the Delta (airline) support agent.

Design notes
------------
The taxonomy was derived by reading a sample of inbound customer tweets to
@Delta and clustering the recurring "asks". We deliberately keep it *small*
(9 classes) because:

* A support agent's value is routing + resolving the common cases well, not
  producing a fine-grained ontology. Fine buckets fragment the training signal
  and confuse both the model and the humans who audit it.
* Escalation policy is defined per-intent, so every intent must map cleanly to
  an action. Buckets we couldn't act on differently were merged.

What we chose NOT to model: language/sentiment as separate intents, product
lines (Delta Vacations, Amex card) as separate intents, and channel routing
(DM vs public) as an intent. Those are handled as *features*, not classes.
"""

from __future__ import annotations

# Canonical intent labels. Order is stable and used for confusion matrices.
INTENTS = [
    "flight_disruption",   # delay, cancellation, missed connection, rebooking
    "baggage",             # lost / delayed / damaged bag, baggage fees & tracking
    "booking_change",      # change/cancel reservation, seat selection, upgrade requests
    "refund_billing",      # refund status, double charge, vouchers/credits, fare disputes
    "check_in_boarding",   # check-in errors, boarding pass, app/kiosk problems
    "loyalty_program",     # SkyMiles / Medallion status, miles posting, points
    "complaint_feedback",  # negative experience / staff complaint with no concrete ask
    "praise",              # thanks / positive sentiment
    "general_info",        # policy / how-to questions (bag allowance, pets, wifi, pets)
]

# A human-readable one-liner per intent (used in reports and reply templates).
INTENT_DESCRIPTIONS = {
    "flight_disruption": "Delay, cancellation, missed connection, or rebooking help.",
    "baggage": "Lost, delayed, or damaged baggage; baggage fees and tracking.",
    "booking_change": "Change or cancel a reservation, pick seats, request upgrades.",
    "refund_billing": "Refund status, double charges, travel credits, fare disputes.",
    "check_in_boarding": "Check-in errors, boarding pass, app/kiosk problems.",
    "loyalty_program": "SkyMiles / Medallion status, miles posting, award travel.",
    "complaint_feedback": "Negative experience or staff complaint with no concrete ask.",
    "praise": "Positive feedback or thanks.",
    "general_info": "Policy or how-to questions (bag allowance, pets, wifi, pets).",
}

# Fallback label the classifier emits when confidence is below the abstain
# threshold. It is NOT part of INTENTS (never a gold label) — it is an
# operational signal that always forces a human review.
ABSTAIN = "uncertain"

assert set(INTENT_DESCRIPTIONS) == set(INTENTS), "description/label mismatch"
