# Golden evaluation sets — how they were sampled and labelled

There are two hand-built sets in this folder. Both were authored and labelled by the
project author. The generator scripts (`scripts/build_golden.py`,
`scripts/build_judge_set.py`) keep every example and its label **inline and
reviewable** — the scripts are just serializers, not label generators.

---

## 1. `golden_eval.jsonl` — intent + escalation (n = 150)

Each record:
```json
{"id":"g001","text":"...","intent":"flight_disruption","escalate":true,"note":"..."}
```

### Sampling
- **Stratified by intent** so all 9 classes are covered (no class < 11 examples).
  Final distribution: flight_disruption 24, general_info 22, refund_billing 18,
  booking_change 16, complaint_feedback 16, baggage 15, check_in_boarding 14,
  loyalty_program 14, praise 11.
- On top of the stratified core we deliberately added a **hard-case block**:
  multi-intent messages, sarcasm/negation, explicit "get me a human", messages
  containing PII (card/phone/email), legal/safety language, and off-topic/empty
  messages ("🔥🔥🔥", "?", giveaway spam).
- Escalation label balance: **88 escalate / 62 auto-handle**.

### Why the examples are freshly written (not sampled from the training file)
The bundled training corpus is a synthetic sample (see the project README). If the
golden examples were drawn from that same corpus, intent scores would measure
*memorization*, not generalization.
So every golden message is **hand-written with different vocabulary** from the
training templates on purpose — e.g. the golden set uses "scrubbed", "bumped /
oversold", "misconnect", "reaccommodation", "IDB", "ecredit", "MQMs", "standby"
that never appear in training. This makes the golden set a genuine
out-of-distribution test. (On the real dataset you would instead sample real
tweets and hold them out; the harness supports that unchanged.)

### Labelling rules
- **`intent`** = the single best intent = the customer's *primary* ask. For
  multi-intent messages we label the higher-stakes ask (e.g. "cancelled AND lost
  my bag, want a refund" → `refund_billing`).
- **`escalate`** = an *independent human judgment* of whether a **human** should
  own the case, using this policy (documented so the harness isn't circular):
  escalate when the message
  (a) needs account-specific action (rebooking, name change, missing miles),
  (b) involves money (refunds, disputed charges, reimbursement),
  (c) is a serious complaint / brand-risk,
  (d) contains legal, safety, distress, or time-critical language,
  (e) exposes PII that must move to a private channel,
  (f) explicitly asks for a human, or
  (g) is genuinely ambiguous / off-topic (route to a person).
  Otherwise a confident, templated auto-reply is acceptable → auto-handle.
- A few deliberately debatable calls are included (e.g. a pure flight-*status*
  question labelled auto-handle even though the intent is `flight_disruption`) so
  the evaluation exposes the precision/recall tension rather than hiding it.

---

## 2. `judge_agreement.jsonl` — reply-quality verdicts (n = 36)

Each record:
```json
{"id":"j001","text":"<customer msg>","reply":"<candidate reply>",
 "human_acceptable":1,"reason":"..."}
```

### Purpose
To **measure whether our automated reply-quality judge can be trusted**. We report
judge↔human Cohen's κ on this set, rather than assuming the judge is correct.

### Sampling / construction
- Balanced **18 acceptable / 18 unacceptable** by design.
- The unacceptable half was written to cover the reply failure modes we actually
  care about, so the judge has to earn its agreement:
  - **Hallucination / false promises** ("we've refunded $450", "delivered within
    24 hours guaranteed"),
  - **Unsafe** replies that solicit card numbers / passwords / CVV in public,
  - **Wrong-intent** replies (praise response to a lost bag),
  - **Dismissive / off-tone** replies ("delays happen", "lol that sucks").
- `human_acceptable` = 1 means a support lead would let the reply go out as-is;
  0 means it should be blocked or rewritten.

### Known limitation surfaced by this set
The default rubric judge agrees with the human on **35/36** cases (κ = 0.94).
Its one miss is a *relevance* failure that was otherwise safe and grounded — i.e.
the rubric judge slightly under-detects "answered the wrong thing" when the
reply is polite and hallucination-free. This is documented in the report as a
reason to prefer an LLM judge (with this set as its regression test) in production.
