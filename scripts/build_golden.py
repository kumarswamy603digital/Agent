#!/usr/bin/env python3
"""Build the hand-labelled golden evaluation set -> data/golden/golden_eval.jsonl

HOW THIS SET WAS SAMPLED & LABELLED (see data/golden/README.md for the full note)
---------------------------------------------------------------------------------
* Each example below was hand-authored and hand-labelled by the project author,
  deliberately written to be *distinct from the training templates* (different
  wording, real-world messiness) so it is a fair out-of-distribution test rather
  than a memorization check.
* We stratified by intent to guarantee coverage of every class, then added a
  block of hard cases: multi-intent, ambiguous, sarcastic, PII-bearing,
  explicit human requests, off-topic, and empty/low-signal messages.
* `intent` = the single best intent (the customer's primary ask).
* `escalate` = independent human judgment of whether a HUMAN should handle this,
  using this policy: escalate when the message needs account-specific action,
  involves money, is a serious complaint, contains legal/safety/urgency/distress,
  explicitly asks for a human, exposes PII, or is genuinely ambiguous. Otherwise
  a confident templated auto-reply is acceptable (auto-handle).
* For ambiguous/off-topic items with no clean intent, we still record our best
  intent guess but the escalate label is what the harness weighs most.

Each tuple: (text, intent, escalate, note)
"""

from __future__ import annotations

import json
import os

EXAMPLES = [
    # ------------------------------------------------------------------ flight_disruption (needs action -> escalate)
    ("@Delta sitting on a cancelled flight to Denver with a 3yo and no rebooking in sight, please help", "flight_disruption", True, "stranded family, needs rebooking"),
    ("hey @delta flt 2281 just got scrubbed, what are my options to get to boston tonight", "flight_disruption", True, "rebooking action"),
    ("@Delta 4 hour delay and now I'll miss my cruise. someone needs to fix this", "flight_disruption", True, "downstream connection impact"),
    ("@Delta why does my flight keep getting pushed back, been at the gate since 6am", "flight_disruption", True, "delay, likely needs rebooking"),
    ("@Delta our plane diverted to Richmond and we're just sitting here, zero info", "flight_disruption", True, "diversion, no info"),
    ("@Delta cancelled my red-eye AGAIN. third time this week. get me on another airline", "flight_disruption", True, "repeat cancellation, reaccommodation"),
    ("@Delta the app says my connection is 20 min after landing and we're late, will I make it??", "flight_disruption", True, "tight connection risk"),
    ("@Delta stranded in Salt Lake overnight because of your cancellation, do I get a hotel?", "flight_disruption", True, "reaccommodation + hotel"),
    ("@Delta is flight DL881 to Paris expected to be delayed tomorrow with the storm?", "flight_disruption", False, "pure status/forecast question, bot can answer"),
    ("@Delta how do I know if my flight is delayed before I leave for the airport?", "flight_disruption", False, "generic how-to, not account action"),
    ("@Delta my whole itinerary fell apart after the first leg cancelled, i need a human NOW", "flight_disruption", True, "explicit human + disruption"),
    ("@Delta 2 hr ground stop at ATL and no gate agent anywhere, missing my connection", "flight_disruption", True, "urgent connection"),

    # ------------------------------------------------------------------ baggage
    ("@Delta landed in Rome but my bag is still in Atlanta apparently. what now", "baggage", True, "delayed bag, cross-border, needs report"),
    ("@Delta my checked bag arrived soaked and everything inside is ruined", "baggage", True, "damaged contents, likely claim"),
    ("@Delta been 4 days and still no bag, the file ref number does nothing", "baggage", True, "long delay, escalate"),
    ("@Delta where do I find my bag if it didn't come out at carousel 5?", "baggage", False, "simple locate, templated ack ok"),
    ("@Delta how much do you charge for a second checked bag to Cancun?", "baggage", False, "fee info"),
    ("@Delta is a golf bag considered oversized?", "baggage", False, "policy/info"),
    ("@Delta someone grabbed a bag that looks like mine off the belt, mine's gone", "baggage", True, "possible mishandling, escalate"),
    ("@Delta my stroller was gate-checked and came back with a broken wheel", "baggage", True, "damage claim"),
    ("@delta can i track my delayed luggage online somewhere", "baggage", False, "self-service info"),
    ("@Delta bag delayed on DL55, when will it be delivered to my hotel?", "baggage", True, "delivery status, account lookup"),
    ("@Delta what's the weight limit before overweight fees kick in", "baggage", False, "policy info"),

    # ------------------------------------------------------------------ booking_change
    ("@Delta need to push my Friday flight to Sunday, what's the change fee situation", "booking_change", False, "fee question, templated route ok"),
    ("@Delta trying to select seats but the site throws an error every time", "booking_change", False, "self-service tech hiccup"),
    ("@Delta how do I add a checked bag to my existing reservation online?", "booking_change", False, "how-to"),
    ("@Delta want to cancel my trip and get an ecredit, conf JQ8812", "booking_change", True, "cancel-for-credit touches money/PII conf"),
    ("@Delta can you change the misspelled last name on my ticket before Thursday?", "booking_change", True, "name change needs agent"),
    ("@Delta is it possible to upgrade to first with miles on my LAX flight?", "booking_change", False, "info/how-to"),
    ("@Delta I need to add my newborn as a lap infant to my booking", "booking_change", True, "reservation edit, agent needed"),
    ("@Delta how late can I change my flight before departure?", "booking_change", False, "policy info"),
    ("@Delta want to switch to an earlier flight today if there's space, standby?", "booking_change", True, "same-day change action"),
    ("@Delta my companion certificate won't apply at checkout, help", "booking_change", True, "account/benefit issue"),

    # ------------------------------------------------------------------ refund_billing (money -> mostly escalate)
    ("@Delta you charged my card twice for the same ticket, I want one reversed", "refund_billing", True, "double charge"),
    ("@Delta still waiting on a refund from a flight you cancelled 3 weeks ago", "refund_billing", True, "refund status/dispute"),
    ("@Delta I was billed $200 for seats I never selected", "refund_billing", True, "billing dispute"),
    ("@Delta how do I request a refund for a refundable fare?", "refund_billing", False, "pure how-to"),
    ("@Delta my travel credit vanished, I had over $400 in there", "refund_billing", True, "missing credit"),
    ("@Delta what's your refund policy for basic economy?", "refund_billing", False, "policy info"),
    ("@Delta charged a bag fee twice at the kiosk, need $60 back", "refund_billing", True, "duplicate fee"),
    ("@Delta refund promised 'in 7-10 days' — it's been 40. Escalating to my bank", "refund_billing", True, "chargeback threat, dispute"),
    ("@Delta do refunds go back to the original card or as a voucher?", "refund_billing", False, "info"),

    # ------------------------------------------------------------------ check_in_boarding
    ("@Delta the app won't generate my boarding pass and I'm at security", "check_in_boarding", True, "time-critical at airport"),
    ("@Delta can't check in online, it says 'see agent', flight's in 90 min", "check_in_boarding", True, "blocked + time pressure"),
    ("@Delta why does check-in open only 24h before? trying to plan", "check_in_boarding", False, "policy info"),
    ("@Delta the fly delta app keeps crashing when I tap check in", "check_in_boarding", False, "app bug, self-service retry"),
    ("@Delta kiosk ate my card and won't print a pass, line's not moving", "check_in_boarding", True, "airport blocker"),
    ("@Delta how do I add my known traveler number so precheck shows up?", "check_in_boarding", False, "how-to"),
    ("@Delta boarding in 5 minutes and my mobile pass just disappeared!!!", "check_in_boarding", True, "urgent"),
    ("@Delta do I need to check in again for my connecting flight?", "check_in_boarding", False, "info"),

    # ------------------------------------------------------------------ loyalty_program
    ("@Delta flew 3 segments last week and none of the miles posted", "loyalty_program", True, "missing miles, account lookup"),
    ("@Delta how many miles for a one-way award to Hawaii in economy?", "loyalty_program", False, "info"),
    ("@Delta I requalified for Platinum but my account still shows Gold", "loyalty_program", True, "status correction"),
    ("@Delta do Diamond members get free upgrades on award tickets?", "loyalty_program", False, "policy info"),
    ("@Delta my miles were deducted but the award booking never confirmed", "loyalty_program", True, "miles + booking mismatch"),
    ("@Delta when do MQMs reset each year?", "loyalty_program", False, "info"),
    ("@Delta can I transfer miles to my spouse's account?", "loyalty_program", False, "how-to"),
    ("@Delta someone used my SkyMiles to book a flight I didn't authorize", "loyalty_program", True, "account fraud -> escalate"),

    # ------------------------------------------------------------------ complaint_feedback (brand risk -> escalate)
    ("@Delta your gate agent in Charlotte was openly mocking a disabled passenger. disgraceful", "complaint_feedback", True, "serious complaint + vulnerable"),
    ("@Delta worst airline experience of my life, rude crew, filthy plane, never again", "complaint_feedback", True, "complaint"),
    ("@Delta your staff left an elderly woman in a wheelchair unattended for an hour", "complaint_feedback", True, "safety/duty of care"),
    ("@Delta the flight attendant snapped at my kid for no reason today", "complaint_feedback", True, "staff complaint"),
    ("@Delta completely unacceptable how we were treated at boarding. want to file a formal complaint", "complaint_feedback", True, "formal complaint"),
    ("@Delta I've never been so disrespected by an airline. utterly humiliating", "complaint_feedback", True, "distress complaint"),

    # ------------------------------------------------------------------ praise (auto-handle)
    ("@Delta shoutout to Captain Reyes and crew on DL1420, smoothest flight ever!", "praise", False, "praise"),
    ("@Delta your agent Maria in MSP saved my trip today, absolute legend 🙌", "praise", False, "praise"),
    ("just flew @Delta after years away and honestly? really impressed", "praise", False, "praise"),
    ("@Delta thank you for getting us home safe through that storm ❤️", "praise", False, "praise"),
    ("@Delta best boarding process of any US airline, well done", "praise", False, "praise"),
    ("@Delta the wifi actually worked the whole flight for once, thanks!", "praise", False, "praise (mild feature mention but positive)"),

    # ------------------------------------------------------------------ general_info (auto-handle unless safety)
    ("@Delta can my emotional support animal fly in cabin now or did rules change?", "general_info", False, "policy info"),
    ("@Delta whats the deal with carry on liquids for international flights", "general_info", False, "policy"),
    ("@Delta do you have power outlets on your 737s?", "general_info", False, "info"),
    ("@Delta how early should I get to LAX for an international departure?", "general_info", False, "info"),
    ("@Delta is there a lounge in Terminal 4 at JFK I can use with my ticket?", "general_info", False, "info"),
    ("@Delta do you allow snowboards as checked baggage and any fees?", "general_info", False, "info"),
    ("@Delta my mother uses oxygen and needs to fly next month, what's the medical process?", "general_info", True, "medical accommodation -> escalate"),
    ("@Delta can an unaccompanied minor connect through Atlanta alone?", "general_info", True, "UM safety policy -> escalate"),

    # ------------------------------------------------------------------ HARD: multi-intent
    ("@Delta my flight got cancelled AND you lost my bag. i want a refund for both", "refund_billing", True, "multi-intent, money dominates"),
    ("@Delta delayed flight made me miss my connection, now I need rebooking and a hotel voucher", "flight_disruption", True, "disruption + money"),
    ("@Delta thanks for the quick rebooking earlier but my miles still haven't posted", "loyalty_program", True, "praise + miles issue; miles is the ask"),
    ("@Delta love flying with you but this double charge on my card is not it", "refund_billing", True, "praise + billing dispute"),

    # ------------------------------------------------------------------ HARD: explicit human request
    ("@Delta can I just talk to a real person please, the bot isn't helping", "flight_disruption", True, "explicit human, ambiguous topic"),
    ("@Delta connect me to a supervisor immediately", "complaint_feedback", True, "explicit human/supervisor"),
    ("@Delta I need to speak with an agent about my reservation", "booking_change", True, "explicit human"),

    # ------------------------------------------------------------------ HARD: PII exposed
    ("@Delta here's my confirmation ABC123 and card ending 4485219930021234, please refund", "refund_billing", True, "PII (card) + refund"),
    ("@Delta my number is 404-555-0172 call me about my cancelled flight", "flight_disruption", True, "PII phone + disruption"),
    ("@Delta email me at traveler.jane@example.com about my delayed bag claim", "baggage", True, "PII email + claim"),

    # ------------------------------------------------------------------ HARD: sarcasm / negation
    ("@Delta oh GREAT, another 'on-time' departure that's already 2 hours late. love it", "flight_disruption", True, "sarcasm masking delay"),
    ("@Delta wow amazing service, only took 5 agents and 3 hours to not solve anything", "complaint_feedback", True, "sarcastic complaint"),
    ("@Delta not thrilled that my 'free' upgrade came with a $99 charge", "refund_billing", True, "sarcasm + billing"),

    # ------------------------------------------------------------------ HARD: legal / safety
    ("@Delta I was injured by a falling bag from the overhead bin, who handles claims", "complaint_feedback", True, "injury/safety + claim"),
    ("@Delta considering legal action over how my disability was handled on DL55", "complaint_feedback", True, "legal + safety"),
    ("@Delta there was smoke in the cabin and no one explained anything, terrifying", "complaint_feedback", True, "safety incident"),

    # ------------------------------------------------------------------ HARD: off-topic / spam / empty
    ("@Delta what's your stock price doing today lol", "general_info", True, "off-topic, low confidence -> route"),
    ("@Delta 🔥🔥🔥", "praise", True, "no signal, ambiguous -> route"),
    ("@Delta ?", "general_info", True, "empty/low-signal -> route"),
    ("@Delta following for the giveaway!! pick me pick me", "general_info", True, "spam/off-topic -> route"),
    ("@Delta is this the official account or a bot", "general_info", False, "meta question, low stakes"),

    # ------------------------------------------------------------------ extra coverage to reach ~190, varied phrasing
    ("@Delta plane's been boarding for 40 min then they said mechanical, are we cancelled?", "flight_disruption", True, "mechanical delay"),
    ("@Delta got bumped off an oversold flight against my will, what compensation applies", "flight_disruption", True, "IDB compensation"),
    ("@Delta my bag's tag says JFK but I flew to SFO, classic", "baggage", True, "misrouted bag"),
    ("@Delta do I get miles for a flight booked with a companion pass?", "loyalty_program", False, "info"),
    ("@Delta seat map won't load for DL212, want a window before they're gone", "booking_change", False, "self-service seats"),
    ("@Delta I paid for comfort+ but got moved to a middle seat in economy", "booking_change", True, "downgrade, refund-adjacent"),
    ("@Delta refund status? case #DL-99182 opened last month, radio silence", "refund_billing", True, "refund status w/ case id"),
    ("@Delta why is checking in for an international flight so different from domestic", "check_in_boarding", False, "info"),
    ("@Delta lost my Medallion status after one bad year, any way to appeal?", "loyalty_program", True, "status appeal, account"),
    ("@Delta the crew on DL9 were so warm and professional, thank you all", "praise", False, "praise"),
    ("@Delta can I bring a full-size guitar as carry on?", "general_info", False, "policy"),
    ("@Delta my elderly dad needs wheelchair assistance connecting in ATL tomorrow", "general_info", True, "accessibility arrangement"),
    ("@Delta app logged me out and won't accept my password to check in", "check_in_boarding", False, "self-service auth"),
    ("@Delta charged for wifi that never connected the entire flight, want it refunded", "refund_billing", True, "billing dispute"),
    ("@Delta what's the pet in cabin fee and carrier size limit?", "general_info", False, "policy"),
    ("@Delta stuck in a customs line and about to miss my Delta connection, can you hold it", "flight_disruption", True, "urgent connection"),
    ("@Delta my flight's fine, just wondering if meals are served on the ATL-LHR route", "general_info", False, "info, not disruption despite 'flight'"),
    ("@Delta absolutely fuming, 6 hours delayed no food no updates no staff", "complaint_feedback", True, "distress complaint about delay"),
    ("@Delta how do I use a travel voucher when booking online?", "refund_billing", False, "how-to, no money movement"),
    ("@Delta bag came out on the belt totally fine, false alarm, thanks anyway", "baggage", False, "resolved/positive, low stakes"),
    ("@Delta will my elite bag waiver still apply if I'm on a partner airline segment?", "loyalty_program", False, "benefit info"),
    ("@Delta seat 14C is broken and won't recline, whole flight like this", "complaint_feedback", True, "onboard product complaint"),
    ("@Delta trying to change my return flight but keep getting fare difference errors online", "booking_change", True, "change + fare/money"),
    ("@Delta is TSA precheck available for my flight out of SLC tomorrow morning?", "general_info", False, "info"),
    ("@Delta miles expired even though I flew this year, that can't be right", "loyalty_program", True, "miles dispute"),
    ("@Delta phenomenal crew today made a rough travel day so much better 💙", "praise", False, "praise"),
    ("@Delta gate changed 3 times with no announcement, nearly missed boarding", "complaint_feedback", True, "operational complaint"),
    ("@Delta can I check in with just my passport if I don't have the confirmation email?", "check_in_boarding", False, "how-to"),
    ("@Delta requesting reimbursement for the hotel I paid during your cancellation", "refund_billing", True, "reimbursement claim"),
    ("@Delta do infants need their own boarding pass?", "check_in_boarding", False, "info"),
    ("@Delta why is your phone line a 3 hour wait, i just need to rebook one flight", "flight_disruption", True, "rebooking + frustration"),
    ("@Delta what terminal is Delta at LAX these days?", "general_info", False, "info"),
    ("@Delta I think my account was hacked, someone changed my email and booked flights", "loyalty_program", True, "account security -> escalate"),
    ("@Delta the overhead bins were full and you gate-checked my bag without asking", "complaint_feedback", True, "complaint"),
    ("@Delta how do I add TSA precheck to a reservation I already made?", "check_in_boarding", False, "how-to"),
    ("@Delta cancelled flight, rebooked me 2 days later with no hotel. this is a joke", "flight_disruption", True, "reaccommodation dispute"),
    ("@Delta do you fly direct from Boston to Amsterdam in winter?", "general_info", False, "route info"),
    ("@Delta my upgrade cleared then got taken away at the gate, what happened", "booking_change", True, "upgrade dispute"),
    ("@Delta thanks, the DM team sorted my refund quickly 👍", "praise", False, "praise/closure"),
    ("@Delta is there a bassinet seat option on long haul?", "general_info", False, "info"),
    ("@Delta stranded overseas, flight cancelled, embassy asking for airline docs, urgent", "flight_disruption", True, "urgent international disruption"),
    ("@Delta why do I have to pay to pick a seat now, ridiculous nickel and diming", "complaint_feedback", True, "policy complaint/venting"),
    ("@Delta can I volunteer to be bumped for compensation on an oversold flight?", "general_info", False, "info/how-to"),
    ("@Delta my connecting gate is on the other side of ATL and I have 25 min, help", "flight_disruption", True, "tight connection, urgent"),
    ("@Delta do lap infants get a baggage allowance?", "general_info", False, "policy"),
    ("@Delta seat selection charged me but the seats aren't on my boarding pass", "booking_change", True, "seat/payment mismatch"),
    ("@Delta appreciate the proactive rebooking text before I even noticed the delay!", "praise", False, "praise"),
    ("@Delta what happens to my bag if I misconnect, does it follow automatically?", "baggage", False, "info"),
    ("@Delta i demand a full refund and compensation for this disaster of a trip", "refund_billing", True, "refund + compensation demand"),
    ("@Delta how do I opt into standby for an earlier flight at the airport?", "check_in_boarding", False, "how-to"),
    ("@Delta your app says gate B12 but the screens say B26, which is right??", "flight_disruption", False, "gate confusion, info-ish, no account action"),
]


def main(out_path="data/golden/golden_eval.jsonl"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen = set()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (text, intent, escalate, note) in enumerate(EXAMPLES, 1):
            key = text.strip().lower()
            assert key not in seen, f"duplicate golden text: {text!r}"
            seen.add(key)
            rec = {
                "id": f"g{i:03d}",
                "text": text,
                "intent": intent,
                "escalate": bool(escalate),
                "note": note,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    # quick label distribution report
    from collections import Counter
    ci = Counter(e[1] for e in EXAMPLES)
    ce = Counter(e[2] for e in EXAMPLES)
    print(f"Wrote {len(EXAMPLES)} golden examples to {out_path}")
    print("intent distribution:", dict(ci))
    print("escalate distribution:", dict(ce))


if __name__ == "__main__":
    main()
