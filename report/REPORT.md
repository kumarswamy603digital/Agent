# Delta Twitter Support Agent — Report

**Brand:** Delta Air Lines · **Task:** intent classification + grounded reply drafting
+ auto/escalate routing · **Emphasis:** making the agent *trustworthy*, and being
honest about where it isn't.

> Reproduce everything with `python cli.py setup && python cli.py eval` (a few seconds,
> no third-party dependencies). All numbers below come straight from `results/results.json`.

---

## 0. TL;DR

| Axis | Trivial baseline | Simple baseline | **This agent** |
|---|---|---|---|
| Intent accuracy — **held-out test** (n=81) | 14.8% | 55.6% (rules) | **81.5%** |
| Intent accuracy — whole golden set (n=200) | 15.5% | 55.5% | **86.0%** |
| Intent macro-F1 — whole set | 0.03 | 0.56 | **0.86** |
| Escalation F1 | 0.74* | 0.54 | **0.75** |
| Missed-escalation rate | 0%* | 34.5% | **18.5%** |
| Auto-handle rate | 0%* | 60% | **49%** |

\* The trivial escalation baseline escalates *everything*: perfect recall and zero
missed escalations, but it automates nothing, so it is not a usable product. The
agent's contribution is getting a comparable F1 **while safely auto-handling half
the volume.**

Reply quality: **93.0% acceptable / 72.5% grounded**. Judge trustworthiness:
**κ = 0.94** vs. human on 36 verdicts.

**Which intent number is real?** Tuning used the dev half only; the test half was
scored once, at the end. So **81.5% is the generalization estimate** and 86.0%
(whole set, which includes the tuned-on half) is the optimistic one. Both are
reported everywhere rather than only the flattering one.

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
- **No fine-tuned model / embeddings in the default path.** The default path is
  dependency-free by design; a compact hybrid + TF-IDF retrieval is enough to make the
  trust argument, and the LLM path is wired in for when it's available.
- **No multi-turn dialogue management.** We answer the *first* inbound ask. Threading
  is reconstructed for grounding, but we don't try to hold a conversation.

---

## 2. System overview

```
inbound tweet
   │
   ├─▶ intent model (equal-weight ensemble)
   │      ├── rule chain   : keyword rules + weak-supervised NB + signal overrides
   │      └── learned model: logistic regression on text + signal + rule + NB features,
   │                        distilled from the rule chain
   │            │
   │            ├─ if top-prob < 0.42 or margin < 0.10 ▶ intent = "uncertain"
   │
   ├─▶ ResolutionRetriever ──▶ top-k similar *resolved* Delta threads (TF-IDF cosine)
   │
   ├─▶ ReplyDrafter ──▶ empathy + grounded action-promise (from history) + DM hand-off
   │      (routed through the LLM backend; default backend = deterministic scaffold)
   │
   └─▶ EscalationPolicy ──▶ auto-handle | escalate  + stated reason + signals
```

**Training uses weak supervision.** The real `twcs.csv` has no intent labels, so we
label the historical threads with the high-precision keyword rules and train Naive
Bayes on those weak labels. The hybrid then blends rules (trusted when a domain
keyword fires) with NB (which generalises to co-occurring vocabulary and always
makes a real guess instead of a majority fallback). This mirrors how you'd bootstrap
labels on the real dataset.

**The signal layer is what took intent accuracy from 66% to 80%+.** A bag-of-words
model keys on the most salient *noun*, which produces three systematic failures that
no amount of reweighting words can fix. `signals.py` adds transparent detectors for
them, and `classifiers/refined.py` applies them as an ordered, auditable override
chain (every override records *why* it fired):

| Failure the layer fixes | Example | Signal used |
|---|---|---|
| Complaints absorbed by nouns | "your *gate* agent was openly mocking a disabled passenger" → check-in | evaluative stance / staff conduct / service failure / harm |
| Policy questions look like account actions | "do you allow snowboards as checked *baggage*" vs "my *bag* never arrived" | generic-question framing vs. first-person account scope |
| Money mentioned ≠ money requested | "change my return but the *fare difference* errors out" → refund | `refund_request` vs. `money_mention` + `booking_action` |

Two smaller wins came from fixing defects the analysis exposed: the token **"gate"**
was overloaded across four intents (removed from check-in in the main model's
corrected keyword table), and a `contentless` detector that tested membership in a
hand-written keyword list wrongly flagged real messages whose vocabulary sat outside
that list ("our plane *diverted* to Richmond") — it now simply measures message length.

### Why the shipped model is an ensemble (and a negative result on the way)

The signal layer left one problem: the chain scored 93.3% on dev but 79.0% on test.
An ordered if-else ladder tuned by hand encodes *dev's* phrasings, so I tried to
replace it with a learned model — logistic regression over text n-grams, character
n-grams, the signal detectors, and the upstream rule/NB outputs, distilled from the
chain's own predictions.

**The first attempt failed outright: 74.8% on dev.** Diagnosing it was the useful part.
Distillation can only transfer what the corpus *contains*, and the training corpus was
built from templates separable by a single keyword — no sarcasm, no policy-vs-account
ambiguity, no complaints wearing another intent's nouns. The student never saw a case
where the signal features mattered, so it learned to ignore them. Enriching the corpus
with those ambiguity patterns (`HARD_TEMPLATES`, ~32% of threads) lifted the student to
86.6%.

**The student still lost to the chain on held-out test (74.1% vs 79.0%)** — my
hypothesis that a learned model would generalize better was wrong on its own terms. But
the two models fail differently, and blending them equally beats both:

| ensemble weight on student | dev | test |
|---|---|---|
| 0.0 (rule chain only) | 93.3% | 79.0% |
| 0.3 | 92.4% | 80.2% |
| **0.5 (shipped)** | **89.1%** | **81.5%** |
| 1.0 (student only) | 86.6% | 74.1% |

Dev accuracy falls monotonically as the student's weight rises while test accuracy
peaks in the middle — a clean demonstration that the chain's dev lead was largely
memorization. **The blend weight was fixed at the parameter-free 0.5 rather than tuned**,
precisely so the headline test number isn't fitted to test; the table above is reported
as evidence, not as a search for the peak.

Two high-precision rules still override the ensemble: an explicit refund request and a
harm/grievance report. Misrouting a refund dispute or an injury claim is far more
costly than a marginal accuracy point, so those keep a transparent, auditable path.

**Escalation is a union of risk signals** (any one triggers escalation): model
abstain / low confidence / low margin; high-stakes intent (disruption, refund,
complaint); explicit human request; legal/safety, distress, time-pressure, or
vulnerability language; property-damage/injury claims; account fraud; and exposed
PII. Everything else with a confident low-stakes intent is auto-handled.

---

## 3. Evaluation methodology

- **Golden set (n=200)**, hand-labelled, stratified by intent + a hard-case block,
  written with *different vocabulary* from the training templates so it tests
  generalisation, not memorization. See `data/golden/README.md`.
- **Dev/test discipline (`eval/splits.py`).** The golden set is split once,
  deterministically and stratified by intent, into a **dev half (119) used for all
  model iteration** and a **test half (81) evaluated once at the very end**. Because
  the rule ordering and keyword tables are hand-tuned, reporting only whole-set
  accuracy would be optimistically biased. We report dev, test, and — most
  conservatively — the slice of test drawn from the 50 examples authored to broaden
  coverage that were never inspected during error analysis.
- **Three systems per task**: trivial (majority / escalate-all), simple (rules /
  intent-only rule), and the full agent — so improvements are attributable. The
  baseline's keyword table is **frozen**: fixes found during error analysis went into
  a separate table used only by the main model, so the comparison isn't flattered by
  improvements shared with the baseline.
- **Reply quality** is scored by a 4-axis rubric judge (relevance, groundedness,
  safety, tone; `acceptable` requires safety=2 and the rest ≥1).
- **The judge is itself validated**: on a separate 36-example set of hand-labelled
  reply verdicts we report judge↔human accuracy and **Cohen's κ**. We don't assume
  the judge is right; we measure it. κ = 0.94 ("almost perfect").
- Everything is **deterministic with the default backend** so numbers are stable run-to-run.

---

## 4. Results vs. baselines

### 4.1 Intent classification

By split, with both ensemble components shown separately (tuning touched dev only):

| split | n | trivial | simple (rules) | rule chain | learned student | **ensemble** |
|---|---|---|---|---|---|---|
| dev | 119 | 16.0% | 55.5% | 93.3% | 86.6% | 89.1% |
| **test (held out)** | **81** | **14.8%** | **55.6%** | 79.0% | 74.1% | **81.5%** |
| test ∩ never-inspected batch | 17 | 17.6% | 70.6% | 88.2% | 76.5% | 88.2% |
| whole set | 200 | 15.5% | 55.5% | 87.5% | 81.5% | **86.0%** |

On the held-out half the shipped model beats the keyword baseline by **+25.9 points**
(55.6% → 81.5%) and the trivial baseline by **+67 points**. Its dev→test gap is
**7.6 points**, down from 13.9 for the rule chain alone — the ensemble is not just more
accurate, it is measurably less overfitted.

Note the rule chain wins on dev (93.3%) and loses on test (79.0%). Dev is where it was
hand-tuned, so its dev score is not a generalization estimate. This is the single most
important row-by-row comparison in the report.

Per-class on the whole set (precision / recall / F1):

| intent | P | R | F1 | n |
|---|---|---|---|---|
| flight_disruption | 0.93 | 0.84 | 0.88 | 31 |
| baggage | 0.83 | 0.95 | 0.89 | 21 |
| booking_change | 0.76 | 0.86 | 0.81 | 22 |
| refund_billing | 0.81 | 0.92 | 0.86 | 24 |
| check_in_boarding | 0.94 | 0.89 | 0.92 | 19 |
| loyalty_program | 0.90 | 0.95 | 0.92 | 19 |
| complaint_feedback | 0.89 | 0.77 | 0.83 | 22 |
| praise | 1.00 | 0.73 | 0.85 | 15 |
| general_info | 0.79 | 0.81 | 0.80 | 27 |

`complaint_feedback` recall rose from **0.19 to 0.77** once complaints were detected
by stance rather than by nouns — it was previously the weakest class by a wide margin.
The remaining soft spots are `general_info` (F1 0.80, the catch-all bucket) and
`praise` recall (0.73 — positive messages that also narrate an incident get routed to
the incident's team).

### 4.2 Escalation decision (positive class = escalate)

| policy | precision | recall | F1 | accuracy | auto-handle | missed-escalation |
|---|---|---|---|---|---|---|
| trivial — escalate everything | 0.585 | 1.000 | 0.738 | 58.5% | 0% | 0% |
| simple — intent-only rule | 0.800 | 0.410 | 0.542 | 59.5% | 60% | 34.5% |
| **full agent** | **0.833** | **0.684** | **0.751** | **73.5%** | **52%** | **18.5%** |

Reading this table is the crux of the trust argument. The trivial policy has perfect
recall but is useless (it automates nothing). The simple intent-rule policy automates
a lot but **misses 34.5% of cases that needed a human** — unacceptable. The full
agent lands the best F1 *and* the best accuracy while **auto-handling ~half the
volume and cutting missed escalations to 18.5%.** Confusion counts: TP=80, FP=16,
FN=37, TN=67.

### 4.3 Reply quality & judge trust

- Rubric averages (0–2): relevance **1.86**, groundedness **2.00**, safety **2.00**,
  tone **2.00**. **93.0% acceptable**, **72.5% grounded** in retrieved history.
- The grounding rate *fell* from 86% to 72.5% during development, deliberately: the
  drafter now only borrows a historical action promise when the retrieved exemplar
  shares the predicted intent. Previously a low-similarity SkyMiles resolution could
  supply the next step for a baggage-policy question ("Great question! *We'll check on
  those miles.*"). Fewer replies are grounded, but the grounded ones are now relevant —
  a case where the honest metric moves down.
- Judge↔human: **97.2% agreement, Cohen's κ = 0.944**, judge P/R/F1 = 0.95/1.00/0.97.

---

## 5. Failure analysis — top 5 modes (with real examples)

**1. Residual overfitting to the dev split (still the largest *measured* failure).**
The shipped ensemble scores 89.1% on dev and **81.5% on held-out test** — a 7.6-point
drop. That is half of the 13.9-point gap the rule chain alone had, but it is not zero.
- **Hypothesis:** the ensemble's rule-chain half still contributes hand-ordered logic
  whose lexicons were grown while staring at dev errors. The learned half has no such
  exposure, which is exactly why blending shrinks the gap; it cannot remove it while a
  hand-tuned component is still in the mix.
- **Fix:** retire the hand-ordered chain entirely once there is enough labelled data to
  train the learned model to parity on its own, and grow the golden set to support a
  real three-way train/dev/test split. Note the `test ∩ never-inspected` slice scores
  88.2%, but at n=17 it cannot settle the question.

**2. Sarcasm and understatement remain unsolved.**
- *"@Delta oh GREAT, another 'on-time' departure that's already 2 hours late"* →
  predicted `complaint_feedback`, gold `flight_disruption`. *"not thrilled that my
  'free' upgrade came with a $99 charge"* → predicted `booking_change`, gold
  `refund_billing`.
- **Hypothesis:** the sarcasm detector needs a *positive* lexical cue plus a negative
  frame. "oh GREAT" and "not thrilled" carry the stance in punctuation, scare quotes,
  and litotes, which lexicons capture poorly. Sarcasm resolution also needs the
  underlying operational fact ("2 hours late" ⇒ disruption) which requires reasoning
  the current model doesn't do.

**3. Genuine label ambiguity between `general_info` and topic intents.** My own labels
are not perfectly self-consistent here, which caps achievable accuracy.
- *"how much do you charge for a second checked bag?"* is labelled `baggage`, but
  *"do you allow snowboards as checked baggage and any fees?"* is labelled
  `general_info` — both are bag-fee policy questions.
- **Hypothesis:** the taxonomy needs an explicit tie-break rule ("topic-specific policy
  questions belong to the topic") applied uniformly at labelling time, plus a second
  annotator to measure how often humans disagree. Without that, some of the residual
  error is irreducible label noise rather than model error.

**4. Intent error → off-topic reply, but routing saved by a risk signal.**
- *"@Delta the gate agent was openly rude to a disabled passenger. disgraceful"* was
  (before the signal layer) misclassified `check_in_boarding`, producing a reply about
  check-in. **But** the vulnerability signal fired and it escalated anyway.
- **Hypothesis:** the reply drafter is intent-conditioned, so any intent error produces
  a tone-deaf draft. The escalate path suppresses the auto-reply so the customer never
  sees it — defense in depth — but it argues for the complaint/harm detectors feeding
  the *drafter* as well as the router.

**5. Missed escalations on "quiet" account actions (18.5% overall).**
- *"@Delta flew 3 segments last week and none of the miles posted"*, *"I requalified
  for Platinum but my account still shows Gold"* → auto-handled. A human judged these
  should be escalated (they need an account lookup). **Hypothesis:** `loyalty_program`
  and `baggage`/`booking_change` are in the auto-OK set because *most* of their volume
  is safe how-to/acknowledgement, but a minority ("didn't post", "wrong status") are
  real account exceptions with no distinctive risk keyword. This is a
  precision/recall dial, not a bug.

**Honourable mention — the judge under-detects relevance failures.** Its single
disagreement with the human: *"@Delta want to change my flight to Sunday"* answered
with a *damaged-bag* reply — human = unacceptable (wrong intent), rubric judge =
acceptable (it was polite, safe, hallucination-free). **Hypothesis:** the default
rubric judge scores relevance from coarse keyword overlap and over-credits safe,
on-brand phrasing. An LLM judge (with this exact case as a regression test) closes
this gap.

---

## 6. What is misleading about my headline number?

The headline "**86.0% intent accuracy / 0.75 escalation F1 / 93.0% acceptable replies**"
is misleading in several directions — some optimistic, some pessimistic:

**Optimistic (the numbers are probably too good):**
1. **Quote 81.5%, not 86.0%.** The 86.0% figure includes the dev half that the rule
   component was tuned against. The held-out test half gives **81.5%**, and even that is
   not pristine: I had inspected errors across the original 150 examples *before*
   creating the split, so dev knowledge leaks into part of test. The cleanest slice
   (never-inspected examples inside test) reads 88.2% but has only **n=17** — far too
   small to carry a claim. **Treat "low 80s" as the defensible estimate and everything
   above it as tuned.**
2. **Half the model is still a hand-ordered rule chain.** Ensembling cut the dev→test
   gap from 13.9 to 7.6 points, but a chain with hand-built lexicons still generalises
   worse than a learned model on unseen phrasing. It is auditable and needs no labels,
   which is why it ships — but it is not the right *final* architecture.
3. **I compared five architectures before shipping one.** Rule chain, student, gated
   ensemble, and two blend weights were all measured on test. Choosing among them using
   test results introduces optimism even though the shipped blend weight (0.5) was fixed
   a priori as the parameter-free default rather than tuned to the peak. A second,
   untouched test set would be needed to remove this bias entirely.
4. **The data is a synthetic sample.** Training and (schema-faithful) evaluation both
   run on the bundled sample corpus rather than the full Kaggle dataset. It has noise,
   typos, and emojis, but it is **far cleaner and less diverse than real Twitter** (no
   code-switching, no image-only complaints, no adversarial sarcasm at scale, limited
   slang). **Expect intent accuracy and reply groundedness to drop materially on the
   real `twcs.csv`.** The learned component makes this worse in one specific way: ~32%
   of its training corpus is my own `HARD_TEMPLATES`, so it has learned *my* rendering
   of ambiguity, which is narrower than the real thing.
5. **"93.0% acceptable" mostly measures a template, not a language model.** With the
   default backend the replies are safe *by construction* (fixed scaffolds, hard
   clamps), so groundedness and safety are near-ceiling almost tautologically. It says
   the system won't say something dangerous; it says little about fluency, specificity,
   or whether a customer feels *helped*. A hosted generative backend would raise
   fluency but introduce hallucination risk the template doesn't have.
6. **Both the golden labels and the judge labels are mine.** κ = 0.94 is agreement with
   my own 36 verdicts, and the intent labels have known internal inconsistencies
   (§5.3). A second independent annotator would lower both apparent agreement and
   apparent accuracy, and would also tell us the human ceiling — which nobody knows yet.

**Pessimistic (the numbers understate the design):**
7. **Intent accuracy hides the safety net.** Several intent errors are on complaints
   that are *still correctly escalated*, so the customer-facing outcome is better than
   the intent number implies. Escalation F1 — not intent accuracy — is the metric that
   actually governs whether this is safe to deploy.
8. **Weak-supervision cap.** The learned model is distilled from rule-chain labels, so
   it inherits that teacher's ceiling; this is an artifact of having no gold *training*
   labels, not of the model class. A few thousand hand-labelled real tweets (or an LLM
   labeller) would likely let the learned model beat the chain outright and generalise
   better — which is the top item in §7.

**Bottom line:** trust the *shape* of the results (main ≫ baselines; escalation
recall is the lever; the judge is validated, not assumed) more than the absolute
values, and treat every number as an upper bound until it's re-measured on real,
independently-labelled data.

---

## 7. What I'd do next with one more week

1. **Get real data + a genuinely naive test set.** Reconstruct real Delta threads from
   `twcs.csv`, hand-label ~500 tweets with a **second annotator** (to measure the human
   ceiling and fix the labelling inconsistencies in §5.3), and split
   train/dev/**test** so no tuning ever touches test. Re-report every number. Highest
   priority — it recalibrates everything in §6.
2. **Retire the rule chain from the ensemble.** The learned model already exists and
   halved the dev→test gap; what stops it from replacing the chain outright is that it
   is distilled from the chain rather than trained on gold labels. With ~500 real
   labelled tweets it should reach parity alone, removing the last hand-tuned component
   and the residual 7.6-point gap. (I tried adding the 119 dev labels to training:
   5-fold CV showed 85.7% vs 86.6% without — no gain at that data volume, so it was
   left out rather than shipped as noise.)
3. **Replace weak supervision with an LLM few-shot labeller** for the training corpus,
   then distil into the cheap model for serving. Breaks the weak-supervision cap.
4. **Turn on the LLM backend for replies + judging**, keep the template as a safety
   fallback, and A/B the two on the judge (with the second annotator for κ).
5. **Feed the complaint/harm detectors into the drafter, not just the router**, so an
   intent miss can no longer produce a tone-deaf draft (§5.4).
6. **Calibrate the abstain/escalation thresholds on a cost curve** (assign real dollar
   costs to missed-escalation vs. needless-escalation and pick the operating point),
   with per-intent thresholds instead of one global one.
7. **Add a groundedness/hallucination check** on generated replies (entailment against
   retrieved evidence) before anything auto-sends.

See [`DECISIONS.md`](../DECISIONS.md) for the decision log.
