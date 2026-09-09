# Golden evaluation sets — how they were sampled and labelled

There are two hand-built sets in this folder. Both were authored and labelled by the
project author. The generator scripts (`scripts/build_golden.py`,
`scripts/build_judge_set.py`) keep every example and its label **inline and
reviewable** — the scripts are just serializers, not label generators.

---

## 1. `golden_eval.jsonl` — intent + escalation (n = 200)

Each record:
```json
{"id":"g001","text":"...","intent":"flight_disruption","escalate":true,"note":"..."}
```

### Sampling
- **Stratified by intent** so all 9 classes are covered (no class < 15 examples).
  Final distribution: flight_disruption 31, general_info 27, refund_billing 24,
  booking_change 22, complaint_feedback 22, baggage 21, check_in_boarding 19,
  loyalty_program 19, praise 15.
- On top of the stratified core we deliberately added a **hard-case block**:
  multi-intent messages, sarcasm/negation, explicit "get me a human", messages
  containing PII (card/phone/email), legal/safety language, and off-topic/empty
  messages ("🔥🔥🔥", "?", giveaway spam).
- Escalation label balance: **117 escalate / 83 auto-handle**.
- The set was built in two rounds: examples **g001–g150** first, then **g151–g200**
  added to broaden coverage so the set could support a dev/test split. The second
  batch was written *before* any model tuning and was not aimed at known failures.

### How the set is used (dev / test discipline)
`eval/splits.py` performs a single deterministic, intent-stratified split into a
**dev half (119)** and a **test half (81)**. All model iteration used dev only; test
was scored once at the end. The harness additionally reports accuracy on
`test ∩ g151–g200` — items never inspected during error analysis — as the most
conservative generalization estimate. Reporting whole-set accuracy alone would be
optimistically biased because the rule ordering is hand-tuned.

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

### Known label-consistency limitation
The boundary between `general_info` and a topic intent is **not perfectly consistent**
in this set. For example *"how much do you charge for a second checked bag?"* is
labelled `baggage`, while *"do you allow snowboards as checked baggage and any fees?"*
is labelled `general_info` — both are bag-fee policy questions. The rule I applied
(topic-specific questions go to the topic; travel-general questions go to
`general_info`) was not applied uniformly across every item. This means some residual
model error is irreducible label noise, and it is why the report calls for a second
annotator to measure the human ceiling before trusting small accuracy differences.

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
