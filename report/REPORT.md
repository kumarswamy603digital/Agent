# Delta Twitter Support Agent — Report

**Brand:** Delta Air Lines · **Task:** intent classification + grounded reply drafting
+ auto/escalate routing · **Emphasis:** making the agent *trustworthy*, and being
honest about where it isn't.

> Reproduce everything with `python cli.py setup && python cli.py eval` (≈2s, no deps,
> no network). All numbers below come straight from `results/results.json`.

---

## 0. TL;DR

| Axis | Trivial baseline | Simple baseline | **This agent** |
|---|---|---|---|
| Intent accuracy | 16.0% | 55.3% (rules) | **66.0%** |
| Intent macro-F1 | 0.03 | 0.55 | **0.65** |
| Escalation F1 | 0.74* | 0.48 | **0.75** |
| Missed-escalation rate | 0%* | 38.7% | **18.7%** |
| Auto-handle rate | 0%* | 66% | **51%** |

\* The trivial escalation baseline escalates *everything*: perfect recall and zero
missed escalations, but it automates nothing, so it is not a usable product. The
agent's contribution is getting a comparable F1 **while safely auto-handling half
the volume.**

Reply quality: **96.7% acceptable / 83% grounded**. Judge trustworthiness:
**κ = 0.94** vs. human on 36 verdicts.

---

## 1. Problem framing — what "good" means for Delta, and what I chose *not* to build

Delta's public social care is a **brand-risk and triage** channel, not a resolution
engine. Almost every real resolution ("here's your rebooking / refund / bag status")
requires acting on a specific reservation, which cannot and must not happen in a
public tweet. That reframes the objective:

> **Good = route correctly and safely, acknowledge in Delta's voice, and never make a
> commitment the brand can't keep — while auto-handling the genuinely low-stakes
> volume so humans can focus on the rest.**

Concretely, "good" for this agent is:
1. **Escalation safety first.** The expensive error is auto-handling something that
   needed a human (a refund dispute, an angry customer, a stranded family, a safety
   issue). So the routing objective is deliberately **asymmetric**: minimise
   *missed escalations* even at the cost of escalating some cases we could have
   handled. Recall on the escalate class matters more than raw accuracy.
2. **Grounded, non-committal replies.** A reply that invents a $200 voucher or a
   "delivered in 24h" promise is worse than a safe "DM us your confirmation number".
   Replies are grounded in Delta's *actual historical resolutions* and pass a safety
   clamp (no invented amounts/timelines, no soliciting card numbers/passwords).
3. **Auditability.** Every decision emits a reason and the evidence it was grounded
   in, so a human reviewer can trust or overrule it.

### What I explicitly chose **not** to build (and why)
- **No account actions.** No rebooking, refunding, or reservation edits. Those are
  exactly the operations that need auth + a human; the agent routes them.
- **No fine-grained ontology.** 9 action-oriented intents, not 40 topical ones.
  Extra buckets fragment training signal and don't change the routing action.
- **No sentiment/language as separate intents.** They're *features* feeding the
  escalation policy, not classes.
- **No fine-tuned model / embeddings in the default path.** The environment is
  offline and dependency-free; a from-scratch hybrid + TF-IDF retrieval is enough to
  make the trust argument, and the LLM path is wired in for when it's available.
- **No multi-turn dialogue management.** We answer the *first* inbound ask. Threading
  is reconstructed for grounding, but we don't try to hold a conversation.

---

## 2. System overview

```
inbound tweet
   │
   ├─▶ HybridClassifier ──▶ intent + calibrated probability
   │      (rule-prior blended with weakly-supervised Naive Bayes)
   │            │
   │            ├─ if top-prob < 0.42 or margin < 0.10 ▶ intent = "uncertain"
   │
   ├─▶ ResolutionRetriever ──▶ top-k similar *resolved* Delta threads (TF-IDF cosine)
   │
   ├─▶ ReplyDrafter ──▶ empathy + grounded action-promise (from history) + DM hand-off
   │      (routed through the LLM backend; offline = deterministic scaffold)
   │
   └─▶ EscalationPolicy ──▶ auto-handle | escalate  + stated reason + signals
```

**Training uses weak supervision.** The real `twcs.csv` has no intent labels, so we
label the historical threads with the high-precision keyword rules and train Naive
Bayes on those weak labels. The hybrid then blends rules (trusted when a domain
keyword fires) with NB (which generalises to co-occurring vocabulary and always
makes a real guess instead of a majority fallback). This mirrors how you'd bootstrap
labels on the real dataset.

**Escalation is a union of risk signals** (any one triggers escalation): model
abstain / low confidence / low margin; high-stakes intent (disruption, refund,
complaint); explicit human request; legal/safety, distress, time-pressure, or
vulnerability language; property-damage/injury claims; account fraud; and exposed
PII. Everything else with a confident low-stakes intent is auto-handled.

---

## 3. Evaluation methodology

- **Golden set (n=150)**, hand-labelled, stratified by intent + a hard-case block,
  written with *different vocabulary* from the training templates so it tests
  generalisation, not memorization. See `data/golden/README.md`.
- **Three systems per task**: trivial (majority / escalate-all), simple (rules /
  intent-only rule), and the full agent — so improvements are attributable.
- **Reply quality** is scored by a 4-axis rubric judge (relevance, groundedness,
  safety, tone; `acceptable` requires safety=2 and the rest ≥1).
- **The judge is itself validated**: on a separate 36-example set of hand-labelled
  reply verdicts we report judge↔human accuracy and **Cohen's κ**. We don't assume
  the judge is right; we measure it. κ = 0.94 ("almost perfect").
- Everything is **deterministic offline** so numbers are stable run-to-run.

---

## 4. Results vs. baselines

### 4.1 Intent classification (golden n=150)

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| trivial — majority class | 16.0% | 0.031 | 0.044 |
| simple — keyword rules | 55.3% | 0.554 | 0.537 |
| **main — rule+NB hybrid** | **66.0%** | **0.653** | **0.644** |

The hybrid beats the rules by **+10.7 points** of accuracy. The gain comes almost
entirely from cases where **no rule keyword fires** — there the rules fall back to a
blind majority guess while NB still contributes learned vocabulary.

### 4.2 Escalation decision (positive class = escalate)

| policy | precision | recall | F1 | accuracy | auto-handle | missed-escalation |
|---|---|---|---|---|---|---|
| trivial — escalate everything | 0.587 | 1.000 | 0.739 | 58.7% | 0% | 0% |
| simple — intent-only rule | 0.833 | 0.341 | 0.484 | 57.3% | 66% | 38.7% |
| **full agent** | **0.822** | **0.682** | **0.745** | **72.7%** | **51%** | **18.7%** |

Reading this table is the crux of the trust argument. The trivial policy has perfect
recall but is useless (it automates nothing). The simple intent-rule policy automates
a lot but **misses 38.7% of cases that needed a human** — unacceptable. The full
agent lands the best F1 *and* the best accuracy while **auto-handling ~half the
volume and cutting missed escalations to 18.7%.** Confusion counts: TP=60, FP=13,
FN=28, TN=49.

### 4.3 Reply quality & judge trust

- Rubric averages (0–2): relevance **1.93**, groundedness **2.00**, safety **2.00**,
  tone **2.00**. **96.7% acceptable**, **83% grounded** in retrieved history.
- Judge↔human: **97.2% agreement, Cohen's κ = 0.944**, judge P/R/F1 = 0.95/1.00/0.97.

---

## 5. Failure analysis — top 5 modes (with real examples)

**1. `complaint_feedback` is the weakest intent (recall 3/16).** It gets absorbed
into whatever concrete noun appears.
- *"@Delta seat 14C is broken and won't recline, whole flight like this"* → predicted
  `complaint`? No → predicted around booking/general; *"the flight attendant snapped
  at my kid"* → drifts to `check_in_boarding`/`baggage`.
- **Hypothesis:** complaints are defined by *tone*, not vocabulary, and share nouns
  with every other intent ("seat", "bag", "gate"). A bag-of-words model keys on the
  noun. **Mitigation that already helps:** the escalation policy has an
  intent-independent distress/complaint signal, so many misclassified complaints are
  *still escalated* (defense in depth) even when the intent label is wrong.

**2. Intent error → off-topic reply, but routing saved by a risk signal.**
- *"@Delta the gate agent was openly rude to a disabled passenger. disgraceful"* →
  misclassified `check_in_boarding`, so the drafted reply is about check-in ("Sorry
  you're hitting a snag checking in…"). **But** the `disab`-vulnerability signal fired
  and it escalated. **Hypothesis:** the reply drafter is intent-conditioned, so an
  intent error produces a tone-deaf draft. In production the escalate path suppresses
  the auto-reply, so the customer never sees the bad draft — but this is the scariest
  latent failure and argues for an intent-independent complaint detector feeding the
  drafter, not just the router.

**3. Missed escalations on "quiet" account actions (18.7% overall).**
- *"@Delta flew 3 segments last week and none of the miles posted"*, *"I requalified
  for Platinum but my account still shows Gold"* → auto-handled. A human judged these
  should be escalated (they need an account lookup). **Hypothesis:** `loyalty_program`
  and `baggage`/`booking_change` are in the auto-OK set because *most* of their volume
  is safe how-to/acknowledgement, but a minority ("didn't post", "wrong status") are
  real account exceptions with no distinctive risk keyword. This is a
  precision/recall dial, not a bug.

**4. `general_info` ⇄ `flight_disruption` / `booking_change` confusion.**
- *"@Delta what's your stock price"* and *"@Delta following for the giveaway"* map to
  `general_info` (correct-ish) but shouldn't be auto-answered; conversely
  *"how late can I change my flight?"* (info) can look like `booking_change`.
  **Hypothesis:** off-topic/meta messages have no in-vocabulary signal; the abstain
  gate catches the emptiest ones ("🔥🔥🔥" → `uncertain` → escalate) but not the ones
  that contain airline words in an off-topic way.

**5. The judge under-detects relevance failures.** Its single disagreement with the
human: *"@Delta want to change my flight to Sunday"* answered with a *damaged-bag*
reply — human = unacceptable (wrong intent), heuristic judge = acceptable (it was
polite, safe, hallucination-free). **Hypothesis:** the offline heuristic judge scores
relevance from coarse keyword overlap and over-credits safe, on-brand phrasing. An
LLM judge (with this exact case as a regression test) closes this gap.

---

## 6. What is misleading about my headline number?

This section is mandatory and I take it seriously. The headline "**66% intent
accuracy / 0.75 escalation F1 / 96.7% acceptable replies**" is misleading in several
directions — some optimistic, some pessimistic:

**Optimistic (the numbers are probably too good):**
1. **The data is synthetic.** The offline environment can't fetch the 3M-row Kaggle
   file, so training and (schema-faithful) evaluation both run on a generated Delta
   corpus. It has noise, typos, and emojis, but it is **far cleaner and less diverse
   than real Twitter** (no code-switching, no image-only complaints, no adversarial
   sarcasm at scale, limited slang). **Expect intent accuracy and reply groundedness
   to drop materially on real `twcs.csv`.** This is the single biggest caveat.
2. **"96.7% acceptable" mostly measures a template, not a language model.** The offline
   replies are safe *by construction* (fixed scaffolds, hard clamps), so groundedness
   and safety are near-ceiling almost tautologically. It says the system won't say
   something dangerous; it says little about fluency, specificity, or whether a
   customer feels *helped*. A real generative backend would raise fluency but
   introduce hallucination risk the template doesn't have.
3. **The judge is graded on a set I also wrote.** κ = 0.94 is against my own 36
   verdicts. A second independent annotator would lower apparent agreement.
4. **I tuned the hybrid weight and escalation signals against this same golden set.**
   With only 150 examples and no held-out test split, there is mild optimistic bias
   from fitting to the eval set. The honest number would come from a fresh, larger,
   independently-labelled test set.

**Pessimistic (the numbers understate the design):**
5. **Raw intent accuracy hides the safety net.** 66% sounds mediocre, but many intent
   errors are on complaints that are *still correctly escalated*, so the
   customer-facing outcome is better than the intent number implies. The escalation
   F1 is the metric that actually governs trust.
6. **Weak-supervision cap.** NB is trained on rule labels, so it can't exceed its
   teacher by much — the ceiling is an artifact of not having gold training labels,
   not of the model. With even a few thousand hand-labelled real tweets (or an LLM
   labeler), this rises substantially.

**Bottom line:** trust the *shape* of the results (main ≫ baselines; escalation
recall is the lever; the judge is validated, not assumed) more than the absolute
values, and treat every number as an upper bound until it's re-measured on real,
independently-labelled data.

---

## 7. What I'd do next with one more week

1. **Get real data + a real held-out test set.** Reconstruct real Delta threads from
   `twcs.csv`, hand-label ~500 tweets, and split train/dev/**test** so no tuning
   touches test. Re-report every number. (Highest priority — it recalibrates
   everything in §6.)
2. **Replace weak supervision with an LLM few-shot labeler** for the training set,
   then distil into the cheap hybrid for serving. Breaks the weak-supervision cap.
3. **Turn on the LLM backend for replies + judging**, keep the template as a safety
   fallback, and A/B the two on the judge (with a *second annotator* for κ).
4. **Fix the complaint failure mode** with an intent-independent sentiment/complaint
   detector that feeds *both* the router and the drafter (so an intent miss can't
   produce a tone-deaf draft).
5. **Calibrate the abstain/escalation thresholds on a cost curve** (assign real
   dollar costs to missed-escalation vs. needless-escalation and pick the operating
   point), and add per-intent thresholds instead of one global one.
6. **Add a groundedness/hallucination check** on generated replies (entailment
   against retrieved evidence) before anything auto-sends.

See [`DECISIONS.md`](../DECISIONS.md) for the decision log.
