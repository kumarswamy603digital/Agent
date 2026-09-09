#!/usr/bin/env python3
"""Generate a synthetic sample corpus in the exact Kaggle `twcs.csv` schema.

Purpose
-------
The repository ships a small, self-contained Delta corpus so the pipeline runs
end-to-end and the headline numbers are reproducible without a multi-gigabyte
Kaggle download. The sample mirrors the real schema and much of its texture:
noisy casing, emojis, typos, @mentions, multi-turn threads, and messy/ambiguous
asks.

Using a synthetic sample instead of the full dataset is a known limitation of the
headline metrics (see REPORT.md -> "What is misleading about my headline
number?"). The *code path* is identical to production:
`data_loader.build_threads` reads this file exactly as it reads the real
`twcs.csv`. To run on the real data:  `export TWCS_PATH=/path/to/twcs.csv`.

Output columns match Kaggle exactly:
  tweet_id, author_id, inbound, created_at, text, response_tweet_id,
  in_response_to_tweet_id
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from datetime import datetime, timedelta

BRAND = "Delta"

# ----------------------------------------------------------------------------- #
# Templates per intent. {slot} filled from banks below. Multiple phrasings and
# noise variants keep the vocabulary realistic rather than trivially separable.
# ----------------------------------------------------------------------------- #
CUSTOMER_TEMPLATES = {
    "flight_disruption": [
        "@Delta my flight {flight} to {city} got cancelled and no one told me. what now??",
        "@Delta stuck at {city} bc {flight} is delayed {mins} mins, gonna miss my connection 😡",
        "why is @Delta flight {flight} delayed AGAIN. i've been waiting {hours} hours",
        "@Delta just missed my connecting flight in {city} because your inbound was late. need to rebook",
        "flight {flight} cancelled @Delta and the app won't let me rebook. help pls",
        "@Delta weather cancelled {flight}, been on hold 2 hrs. can someone rebook me to {city}?",
        "@Delta weve been sitting on the tarmac at {city} for {hours} hours whats going on",
        "@Delta my {city} flight is showing delayed but the board says on time?? which is it",
    ],
    "baggage": [
        "@Delta my bag didn't show up at {city} baggage claim. where is it??",
        "@Delta you lost my luggage on flight {flight}. this is unacceptable",
        "hey @Delta my suitcase came off the belt in {city} completely damaged 😞",
        "@Delta still no sign of my checked bag, it's been {days} days. file ref {ref}",
        "@Delta how much is a checked bag on a domestic flight to {city}?",
        "@Delta delayed baggage again, bag never made my connection in {city}",
        "@Delta the wheel on my suitcase was ripped off by baggage handling. who do i talk to",
    ],
    "booking_change": [
        "@Delta I need to change my flight {flight} to an earlier one to {city}, how?",
        "@Delta can I cancel my reservation and get a credit? confirmation {ref}",
        "@Delta trying to pick seats for {flight} but the app keeps erroring out",
        "@Delta how do I upgrade to comfort+ on my {city} flight?",
        "@Delta I want to move my {city} trip to next week, is there a fee?",
        "@Delta need to add my daughter to reservation {ref}, can you help",
        "@Delta how do i change the name spelling on my ticket to {city}",
    ],
    "refund_billing": [
        "@Delta I was charged twice for flight {flight}. want a refund ASAP",
        "@Delta where is my refund?? cancelled {days} days ago, still nothing. conf {ref}",
        "@Delta you charged me ${amt} for a bag i never checked. refund please",
        "@Delta my travel credit disappeared from my account, i had ${amt}",
        "@Delta requesting a refund for the cancelled {city} flight, this is my 3rd time asking",
        "@Delta got double billed ${amt} on my card for seat selection, fix this",
    ],
    "check_in_boarding": [
        "@Delta I can't check in online for flight {flight}, keeps saying error",
        "@Delta the app won't load my boarding pass for {city}, boarding in {mins} min!!",
        "@Delta kiosk at {city} isn't printing my boarding pass, line is huge",
        "@Delta why won't your app let me check in, it just spins forever",
        "@Delta mobile boarding pass gone from wallet, flight {flight} in an hour",
        "@Delta check in says see agent but there's no agent at the {city} counter",
    ],
    "loyalty_program": [
        "@Delta my miles from flight {flight} last week never posted to my skymiles",
        "@Delta how many miles do i need for an award flight to {city}?",
        "@Delta I hit {miles} MQMs, when does my medallion status upgrade?",
        "@Delta my skymiles number {ref} shows the wrong status, i'm platinum",
        "@Delta booked an award ticket to {city} but no miles were deducted?",
        "@Delta do medallion members get free bags on flights to {city}?",
    ],
    "complaint_feedback": [
        "@Delta absolutely the worst experience today. gate agent in {city} was so rude",
        "@Delta your crew on {flight} were dismissive and unprofessional. never flying again",
        "@Delta ruined my {city} trip, disorganized and rude staff everywhere",
        "@Delta i'm disgusted by how i was treated at the {city} gate today",
        "@Delta terrible service on flight {flight}, no one cared that we were delayed",
    ],
    "praise": [
        "@Delta huge thanks to the crew on {flight} to {city}, they were amazing! ✈️",
        "@Delta shoutout to the gate agent in {city} who helped rebook me, went above and beyond 🙏",
        "@Delta best flight experience in years today, thank you!",
        "@Delta your flight attendant on the {city} route was so kind to my kids ❤️",
        "just want to say @Delta made my {city} trip smooth and easy. appreciate you",
    ],
    "general_info": [
        "@Delta can I bring my dog in the cabin on a flight to {city}?",
        "@Delta what's the carry-on size limit for international?",
        "@Delta is there wifi on the {city} route? need to work",
        "@Delta how early should i arrive for a domestic flight to {city}?",
        "@Delta whats your policy on bringing a car seat for my toddler",
        "@Delta are there power outlets on flight {flight}?",
    ],
}

AGENT_TEMPLATES = {
    "flight_disruption": [
        "We're so sorry for the disruption. Please DM us your confirmation number and we'll look at rebooking options right away. ^AB",
        "Apologies for the delay to your travel plans. Send us a DM with your confirmation # and we'll check the next available flights. ^JK",
        "We understand missing a connection is stressful. DM us your confirmation number and we'll get you rebooked. ^RM",
    ],
    "baggage": [
        "So sorry to hear your bag didn't arrive. Please DM your file reference or confirmation number and we'll track it down. ^TL",
        "We're sorry about your damaged bag. Please DM us your confirmation number and we'll open a claim and help you further. ^DS",
        "Delayed bags are frustrating. DM us your file reference and we'll check its status right away. ^AB",
    ],
    "booking_change": [
        "Happy to help with your reservation. DM us your confirmation number and the change you'd like and we'll take a look. ^JK",
        "We can review your options. Please DM your confirmation number and we'll check change/cancel fees for your fare. ^RM",
    ],
    "refund_billing": [
        "We understand billing issues are stressful. Please DM us your confirmation number and the charge details so we can review it. ^TL",
        "Sorry for the trouble with your refund. DM us your confirmation number and we'll check its status for you. ^DS",
    ],
    "check_in_boarding": [
        "Sorry you're having trouble checking in. Try the Fly Delta app first; if it persists DM us your confirmation number and we'll get you a boarding pass. ^AB",
        "Let's get you boarded. Please DM your confirmation number and we'll pull up your boarding pass. ^JK",
    ],
    "loyalty_program": [
        "Thanks for being a SkyMiles member. Please DM us your SkyMiles number and the flight details and we'll check on those miles. ^RM",
        "We can look into that. DM us your SkyMiles number and we'll review your account and status. ^TL",
    ],
    "complaint_feedback": [
        "We're truly sorry about your experience — this isn't the standard we aim for. Please DM us the details so the right team can follow up. ^DS",
        "This is not the experience we want you to have. Please DM us what happened and your flight details so we can make it right. ^AB",
    ],
    "praise": [
        "Thank you so much for the kind words! We'll be sure to pass this along to the team. Safe travels! ^JK",
        "This made our day — thank you! We'll share your shoutout with the crew. ✈️ ^RM",
    ],
    "general_info": [
        "Great question! You can find the full policy details on delta.com, and we're happy to help if anything's unclear. ^TL",
        "Happy to point you in the right direction — the details are on delta.com. Let us know if you have questions! ^DS",
    ],
}

# ----------------------------------------------------------------------------- #
# HARD / AMBIGUOUS templates.
#
# The straightforward templates above are separable by a single keyword, which means
# a model trained only on them never learns to weigh *stance* or *framing*. Real
# support queues are full of cases where the salient noun points the wrong way:
# complaints that mention a bag, policy questions that mention a bag, sarcasm that
# looks positive. Without these, distilling the rule chain into a learned model
# transfers nothing useful for exactly the cases that matter.
#
# These are written as generic language patterns (not copies of any evaluation
# example) so the learned model sees each ambiguity type many times with varied
# wording.
# ----------------------------------------------------------------------------- #
HARD_TEMPLATES = {
    # complaints whose salient noun belongs to another intent
    "complaint_feedback": [
        "@Delta the gate staff at {city} were unbelievably rude to us this morning",
        "@Delta your agent at the {city} counter was dismissive and unhelpful",
        "@Delta seat on {flight} was filthy and the tray table was broken, unacceptable",
        "@Delta crew on {flight} ignored us the entire flight, appalling service",
        "@Delta gate-checked our bag without asking and nobody explained why",
        "@Delta boarding at {city} was chaos, no announcements, nobody in charge",
        "@Delta left us on the jetbridge for 40 minutes with no explanation at all",
        "@Delta the check-in agent in {city} spoke to my wife appallingly",
        "@Delta baggage staff at {city} laughed when I said my case was damaged",
        "@Delta no one on your crew acknowledged we'd been waiting since {hours}am",
        "@Delta absolutely disgraceful how the {city} team handled our family today",
        "@Delta your staff left an elderly passenger unattended in a wheelchair",
        "@Delta being treated like freight on your {city} route lately, genuinely awful",
        "@Delta third time this month the crew has been dismissive. done with you",
        "@Delta was hurt when a case fell from the overhead bin, who deals with this",
        "@Delta considering legal action over the way my disability was handled",
    ],
    # policy questions that mention a topic noun but are NOT account actions
    "general_info": [
        "@Delta do you allow surfboards as checked baggage and what are the fees?",
        "@Delta can I bring a cello on board or does it need its own seat?",
        "@Delta are there power outlets at every seat on the {city} route?",
        "@Delta is a garment bag counted as a personal item or carry-on?",
        "@Delta do you offer meals on flights to {city} or should I bring food?",
        "@Delta is lounge access included with a Comfort+ fare?",
        "@Delta what's the youngest age you allow a child to fly alone?",
        "@Delta do you have bassinet positions on your long haul aircraft?",
        "@Delta are quiet zones a thing on your widebody flights?",
        "@Delta can I travel late in pregnancy and do you require a note?",
        "@Delta what terminal do you operate from at {city} now?",
        "@Delta do you fly nonstop to {city} during the winter season?",
        "@Delta how early do you recommend arriving for international departures?",
        "@Delta am I allowed to bring a folding bicycle as checked baggage?",
    ],
    # sarcasm / understatement wrapping a concrete operational problem
    "flight_disruption": [
        "@Delta oh wonderful, another 'on time' departure that left {mins} minutes late",
        "@Delta love it when the board says boarding and the crew hasn't arrived",
        "@Delta fantastic, {hours} hours on the tarmac at {city} and still no update",
        "@Delta apparently 'brief delay' means {hours} hours in {city} these days",
        "@Delta great, our {city} flight cancelled again, third time this month",
    ],
    # money MENTIONED while asking for a booking change (not a refund request)
    "booking_change": [
        "@Delta trying to move my {city} flight to Sunday, what's the change fee?",
        "@Delta want to switch to an earlier flight, do I pay a fare difference?",
        "@Delta trying to book with a voucher and the code won't apply at checkout",
        "@Delta need to move two passengers off a four person itinerary, possible?",
        "@Delta can I split my round trip so only the return date changes?",
        "@Delta paid for seat selection but the seats aren't on my boarding pass",
        "@Delta my travel dates changed, is there a fee to move the booking?",
        "@Delta need to add a passenger to reservation {ref}, what does that cost?",
        "@Delta how do I add a checked bag to my existing reservation online?",
        "@Delta want to change the name spelling on my ticket before departure",
    ],
    # praise that mentions a topic noun
    "praise": [
        "@Delta the ground crew at {city} tracked down my gate-checked bag, incredible",
        "@Delta best boarding process of any airline I've flown, well done",
        "@Delta the wifi worked the entire {city} flight for once, thank you",
        "@Delta your DM team sorted my refund in ten minutes, thank you so much",
        "@Delta smooth landing in rough wind at {city}, hats off to the flight deck",
        "@Delta gate agent in {city} rebooked us in minutes, absolute legend",
    ],
    # explicit refund requests (so refund_request stays high precision)
    "refund_billing": [
        "@Delta charged twice for the same {city} ticket, one needs reversing",
        "@Delta want the fare difference back after you swapped the aircraft",
        "@Delta cancelled inside 24 hours and still no refund has posted",
        "@Delta paid ${amt} for wifi that never connected, want that refunded",
        "@Delta my ecredit expired while flights were suspended, can it be reinstated",
    ],
    # account-scoped loyalty exceptions
    "loyalty_program": [
        "@Delta credited miles for {flight} but not the segment MQMs",
        "@Delta my upgrade certificates vanished after my status renewed",
        "@Delta rollover MQMs from last year never appeared in my account",
        "@Delta will my elite bag waiver apply on a partner airline segment?",
        "@Delta do miles earned with a partner count toward Medallion status?",
    ],
    # low-signal / off-topic
    "general_info_offtopic": [
        "@Delta ?",
        "@Delta 🔥🔥🔥",
        "@Delta following for the giveaway, pick me!",
        "@Delta is this the official account or a bot",
        "@Delta what's your share price doing today",
    ],
}

CITIES = ["ATL", "JFK", "LAX", "SEA", "BOS", "DTW", "MSP", "SLC", "LGA", "ORD", "Atlanta", "New York", "Detroit"]
REFS = ["ABC123", "XZ9921", "H7K2L0", "QQ1029", "DL55210", "GKR777"]


def _noise(s: str, rng: random.Random) -> str:
    """Occasionally inject a realistic typo / casing change."""
    if rng.random() < 0.18:
        s = s.replace("you", "u", 1) if "you" in s else s
    if rng.random() < 0.12:
        s = s.replace("please", "pls", 1)
    if rng.random() < 0.10 and len(s) > 20:
        i = rng.randint(5, len(s) - 5)
        s = s[:i] + s[i + 1] + s[i] + s[i + 2:]  # swap two chars
    return s


def _fill(t: str, rng: random.Random) -> str:
    return t.format(
        flight=f"DL{rng.randint(100, 4999)}",
        city=rng.choice(CITIES),
        mins=rng.choice([20, 30, 45, 60, 90, 120]),
        hours=rng.choice([1, 2, 3, 4]),
        days=rng.choice([2, 3, 5, 7, 10]),
        amt=rng.choice([30, 60, 75, 120, 200, 350]),
        miles=rng.choice([25000, 40000, 60000, 75000]),
        ref=rng.choice(REFS),
    )


def generate(n_threads: int, seed: int, out_path: str):
    rng = random.Random(seed)
    intents = list(CUSTOMER_TEMPLATES.keys())
    # Realistic class imbalance (disruption/baggage dominate airline social care).
    weights = {
        "flight_disruption": 0.24, "baggage": 0.16, "booking_change": 0.12,
        "refund_billing": 0.10, "check_in_boarding": 0.10, "loyalty_program": 0.08,
        "complaint_feedback": 0.08, "praise": 0.06, "general_info": 0.06,
    }
    pop = intents
    wts = [weights[i] for i in pop]

    # Fraction of threads drawn from the hard/ambiguous template bank. Without
    # these the corpus is separable by single keywords and a learned model never
    # sees the cases where stance and framing matter.
    hard_fraction = 0.32

    rows = []
    tid = 1000
    base_time = datetime(2017, 10, 1, 8, 0, 0)
    for k in range(n_threads):
        intent = rng.choices(pop, weights=wts, k=1)[0]
        use_hard = rng.random() < hard_fraction
        if use_hard:
            hard_key = rng.choice(list(HARD_TEMPLATES.keys()))
            # "general_info_offtopic" is a template bucket, not a label
            intent = "general_info" if hard_key == "general_info_offtopic" else hard_key
            cust_t = rng.choice(HARD_TEMPLATES[hard_key])
        else:
            cust_t = rng.choice(CUSTOMER_TEMPLATES[intent])
        agent_t = rng.choice(AGENT_TEMPLATES[intent])
        cust_text = _noise(_fill(cust_t, rng), rng)
        agent_text = _fill(agent_t, rng)

        cust_id = tid; tid += 1
        agent_id = tid; tid += 1
        ts = (base_time + timedelta(minutes=7 * k)).strftime("%a %b %d %H:%M:%S +0000 %Y")
        ts2 = (base_time + timedelta(minutes=7 * k + 3)).strftime("%a %b %d %H:%M:%S +0000 %Y")

        # customer inbound tweet -> responded to by agent
        rows.append({
            "tweet_id": cust_id,
            "author_id": f"user_{rng.randint(10000, 99999)}",
            "inbound": "True",
            "created_at": ts,
            "text": cust_text,
            "response_tweet_id": str(agent_id),
            "in_response_to_tweet_id": "",
        })
        # brand agent reply
        rows.append({
            "tweet_id": agent_id,
            "author_id": BRAND,
            "inbound": "False",
            "created_at": ts2,
            "text": agent_text,
            "response_tweet_id": "",
            "in_response_to_tweet_id": str(cust_id),
        })

    # A little cross-brand / off-topic noise that does NOT involve Delta agents,
    # to prove the thread reconstruction correctly ignores it.
    for _ in range(max(10, n_threads // 20)):
        nid = tid; tid += 1
        rows.append({
            "tweet_id": nid,
            "author_id": f"user_{rng.randint(10000, 99999)}",
            "inbound": "True",
            "created_at": base_time.strftime("%a %b %d %H:%M:%S +0000 %Y"),
            "text": rng.choice([
                "@AmericanAir my flight is delayed too, so annoying",
                "does anyone know a good pizza place near ATL",
                "@united lost my bag as well ugh",
            ]),
            "response_tweet_id": "",
            "in_response_to_tweet_id": "",
        })

    rng.shuffle(rows)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fields = ["tweet_id", "author_id", "inbound", "created_at", "text",
              "response_tweet_id", "in_response_to_tweet_id"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} tweets ({n_threads} threads) to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2600, help="number of threads")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="data/raw/twcs_sample_delta.csv")
    args = ap.parse_args()
    generate(args.n, args.seed, args.out)
