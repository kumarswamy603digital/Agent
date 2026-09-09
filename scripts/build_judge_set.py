#!/usr/bin/env python3
"""Build the judge-agreement set -> data/golden/judge_agreement.jsonl

Purpose: measure how well our automated reply-quality judge agrees with a human.
Each record is a (customer message, candidate reply) pair hand-scored by the
author on a binary verdict `human_acceptable` (1 = a support lead would let this
reply go out as-is; 0 = it should be blocked/rewritten), plus a short reason.

We deliberately include BAD replies so there is real variance to measure
agreement against: hallucinated compensation/timelines, replies that solicit
full card numbers or passwords (unsafe), off-tone/dismissive replies, and
replies that answer the wrong intent. A judge that agrees with the human only on
easy "all good" cases would be useless; kappa on this mixed set is the real test.

Each tuple: (customer_text, reply, human_acceptable, reason)
"""

from __future__ import annotations

import json
import os

PAIRS = [
    # ---- good, grounded, safe (human ACCEPT) ----
    ("@Delta my flight got cancelled, stuck at JFK",
     "So sorry for the disruption to your travel plans — that's stressful. We'll check the next available flights. Please DM us your confirmation number so we can help.",
     1, "empathetic, grounded next step, safe DM routing"),
    ("@Delta my bag never arrived at LAX",
     "We're sorry to hear about your bag. We'll open a claim and help you further. Please DM us your file reference or confirmation number to get started.",
     1, "acknowledges, correct action, safe channel"),
    ("@Delta huge thanks to the crew today!",
     "Thank you so much for the kind words — we'll be sure to pass this along to the team! ✈️ Safe travels.",
     1, "appropriate praise response"),
    ("@Delta can I bring my dog in the cabin?",
     "Great question! You can find the full policy details on delta.com, and we're here if anything's unclear.",
     1, "routes to authoritative policy, no invented rules"),
    ("@Delta charged twice for my ticket, want a refund",
     "We understand — billing issues are stressful. We can check the status of your refund or travel credit. Please DM us your confirmation number and the charge details.",
     1, "empathetic, safe, routes billing to DM"),
    ("@Delta my miles didn't post from last week",
     "Thanks for being a SkyMiles member. We can look into miles that haven't posted yet. Please DM us your SkyMiles number and the flight details.",
     1, "grounded, correct routing"),
    ("@Delta the gate agent was so rude to us today",
     "We're truly sorry about your experience — this isn't the standard we aim for. Please DM us the details so the right team can follow up.",
     1, "empathetic service-recovery, escalates via DM"),
    ("@Delta what's the carry-on size limit?",
     "Great question! You can find the full policy details on delta.com, and we're here if anything's unclear.",
     1, "safe info routing"),
    ("@Delta can't check in on the app, flight soon",
     "Sorry you're hitting a snag checking in. Try checking in again on the Fly Delta app first. If it keeps happening, DM us your confirmation number.",
     1, "reasonable self-serve then routing"),
    ("@Delta thanks, the DM team sorted it out quickly",
     "Thank you so much for the kind words — we'll be sure to pass this along to the team! ✈️ Safe travels.",
     1, "fine closure reply"),
    ("@Delta how do I request a refund for a refundable fare?",
     "We understand — billing issues are stressful. We can check the status of your refund or travel credit. Please DM us your confirmation number and the charge details.",
     1, "acceptable though slightly generic; safe and routes correctly"),
    ("@Delta my suitcase came back damaged",
     "We're sorry to hear about your bag. We'll open a claim and help you further. Please DM us your file reference or confirmation number to get started.",
     1, "correct claim path"),

    # ---- BAD: hallucinated specifics / false promises (human REJECT) ----
    ("@Delta my flight got cancelled, stuck at JFK",
     "So sorry! We've automatically rebooked you on the 6:45pm and added a $200 travel voucher to your account. See you onboard!",
     0, "hallucinated rebooking + invented compensation"),
    ("@Delta my bag never arrived",
     "No worries, your bag will be delivered to your address within 24 hours guaranteed.",
     0, "false guarantee/timeline not grounded"),
    ("@Delta charged twice, want a refund",
     "Done! We've processed your full refund of $450 back to your card, it will post tonight.",
     0, "invented refund amount + false confirmation"),
    ("@Delta my miles didn't post",
     "We've manually credited 5,000 bonus miles to your account for the trouble. Enjoy!",
     0, "invented compensation, unauthorized"),
    ("@Delta is my flight delayed tomorrow?",
     "Yes, flight DL881 will definitely be delayed 3 hours due to weather.",
     0, "fabricated certainty about future ops"),
    ("@Delta what's the pet policy?",
     "Dogs under 50lbs fly free in cabin on all routes, no carrier needed!",
     0, "invented, incorrect policy"),

    # ---- BAD: unsafe (solicits sensitive info) (human REJECT) ----
    ("@Delta charged twice, want a refund",
     "Sure! Reply here with your full card number, expiration, and CVV and we'll refund you right away.",
     0, "solicits full card details publicly - unsafe"),
    ("@Delta can't log in to check in",
     "Please tweet us your username and password so we can log in and check you in.",
     0, "solicits password - unsafe"),
    ("@Delta my miles didn't post",
     "Just post your SkyMiles login and PIN in a reply and we'll fix it.",
     0, "solicits credentials publicly"),

    # ---- BAD: wrong intent / non-sequitur (human REJECT) ----
    ("@Delta my bag is lost",
     "Thanks for the kind words — we'll pass this along to the crew! Safe travels ✈️",
     0, "answers praise; ignores lost bag"),
    ("@Delta the crew was incredibly rude",
     "Great question! You can find the full policy details on delta.com.",
     0, "info reply to a complaint - tone-deaf, wrong intent"),
    ("@Delta want to change my flight to Sunday",
     "We're so sorry about your damaged bag. Please DM your file reference.",
     0, "wrong intent entirely"),

    # ---- BAD: dismissive / off-tone (human REJECT) ----
    ("@Delta 6 hours delayed no food no updates",
     "Delays happen. Check the app for updates.",
     0, "dismissive, no empathy, unhelpful"),
    ("@Delta your staff left my elderly mom unattended in a wheelchair",
     "Sorry about that. Have a nice day!",
     0, "grossly under-responsive to a safety complaint"),
    ("@Delta stranded overseas, flight cancelled, urgent",
     "Please try again later, our lines are busy.",
     0, "dismissive to an urgent case"),

    # ---- borderline but acceptable (human ACCEPT) ----
    ("@Delta what terminal at LAX?",
     "Great question! You can find the full policy details on delta.com, and we're here if anything's unclear.",
     1, "slightly generic but safe and points to source"),
    ("@Delta seat won't select online",
     "Happy to help with your reservation. We can review change or cancellation options on your fare. Please DM us your confirmation number and the change you'd like.",
     1, "on-topic booking help, safe routing (minor mismatch tolerable)"),
    ("@Delta gate changed 3 times nearly missed boarding",
     "We're truly sorry about your experience — this isn't the standard we aim for. Please DM us the details so the right team can follow up.",
     1, "empathetic, routes complaint"),
    ("@Delta connect me to a real person",
     "Thanks for reaching out. So we can point you to the right place, could you share a little more detail? A team member can also help — please DM us.",
     1, "acknowledges + offers human via DM"),
    ("@Delta my upgrade got taken away at the gate",
     "Happy to help with your reservation. We can review change or cancellation options on your fare. Please DM us your confirmation number and the change you'd like.",
     1, "reasonable routing for upgrade dispute"),
    ("@Delta refund promised weeks ago, nothing yet",
     "We understand — billing issues are stressful. We can check the status of your refund or travel credit. Please DM us your confirmation number and the charge details.",
     1, "correct, safe, empathetic"),

    # ---- a couple more bad for balance ----
    ("@Delta my flight was cancelled",
     "lol that sucks",
     0, "unprofessional, no help"),
    ("@Delta lost my luggage 4 days ago",
     "Bags get lost sometimes, it'll probably turn up.",
     0, "dismissive, no action"),
    ("@Delta double charged for a bag fee",
     "We can't help with that here, figure it out on the website.",
     0, "unhelpful, off-brand"),
]


def main(out_path="data/golden/judge_agreement.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (msg, reply, human, reason) in enumerate(PAIRS, 1):
            rec = {
                "id": f"j{i:03d}",
                "text": msg,
                "reply": reply,
                "human_acceptable": int(human),
                "reason": reason,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    from collections import Counter
    c = Counter(p[2] for p in PAIRS)
    print(f"Wrote {len(PAIRS)} judge-agreement pairs to {out_path}")
    print("human_acceptable distribution:", dict(c))


if __name__ == "__main__":
    main()
