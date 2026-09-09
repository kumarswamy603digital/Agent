# Decision log

The 10–15 non-obvious decisions behind this agent, and why.

1. **Picked Delta (airline) as the brand.** Airline social care has naturally
   separable, action-oriented intents *and* an unusually clean auto-vs-escalate story
   (rebooking/refunds/complaints → human; policy/how-to/praise → auto). That makes it
   the best vehicle for the project's real focus: the *trust* argument.

2. **Framed the objective as safe triage, not resolution.** Public tweets can't
   perform account actions, so "good" = correct routing + non-committal grounded
   acknowledgement, not "resolve the ticket". This decision shapes the whole design
   (esp. that escalation recall > raw accuracy).

3. **Kept the taxonomy small (9 intents).** Every intent maps to a distinct routing
   action; buckets we couldn't act on differently were merged. Sentiment/language are
   features, not classes.

4. **Made the escalation policy deliberately asymmetric and rule-driven, not
   learned.** The cost of a missed escalation (angry customer, refund dispute, safety
   issue auto-answered) dwarfs the cost of a needless one. A transparent union-of-
   signals policy is auditable and lets us reason about *why* each case escalated —
   more trustworthy than a black-box classifier for a safety decision.

5. **Weak supervision for training labels.** The real `twcs.csv` has no intent labels,
   so we label historical threads with the keyword rules and train NB on those. It's
   honest about the real-data situation and reproducible. Its ceiling (NB ≤ its
   teacher) is documented rather than hidden.

6. **Main model = rule+NB *hybrid*, not pure NB.** Pure NB (weakly supervised) only
   *matched* the rules on the golden set. Blending — trust rules when a domain keyword
   fires, defer to NB when they're silent instead of guessing the majority — lifted
   accuracy 55%→66% and yields one coherent probability for the abstain/escalation
   gate. Chose a blend weight of 0.7 by a small sweep on the dev split.

6b. **Added a signal-disambiguation layer on top of the blend (66% → 80%+).** Error
   analysis showed three failures a bag-of-words model *cannot* fix by reweighting
   words: complaints being absorbed by whatever noun appears, policy questions looking
   like account actions, and money being *mentioned* rather than *requested*. These are
   about stance and grammatical framing, so they get explicit detectors
   (`signals.py`) applied as an ordered override chain that records why each override
   fired. Chose transparent regex/lexicon detectors over a learned model because they
   need no labelled data and a support lead can audit and edit them — while
   documenting that a learned version should replace them (see the report's next-steps).

6c. **Split the golden set into dev/test *before* tuning, and froze the baseline's
   keyword table.** Hand-tuned rules can trivially memorise an evaluation set, so all
   iteration used the dev half and the test half was scored once at the end
   (`eval/splits.py`). Fixes discovered during error analysis went into a *separate*
   corrected table used only by the main model, so the main-vs-baseline comparison
   isn't flattered by shared improvements. The resulting 13.9-point dev→test gap is
   reported as a headline caveat rather than buried.

7. **Naive Bayes over logistic regression** for the ML component: one-pass, no SGD
   seeding/convergence variance (so results are bit-stable), and its class-conditional
   log-probs are directly inspectable (`explain()`), which matters for a system we're
   asking a team to trust.

8. **Unigrams for the classifier, uni+bigrams for retrieval.** Bigrams *hurt* the
   classifier on the small corpus (sparse, training bigrams don't match golden
   phrasings) but *help* retrieval (phrase matches improve grounding). Different jobs,
   different features.

9. **Grounded replies extract the historical *action promise* ("we'll open a claim"),
   not the whole past reply.** Early drafts duplicated the DM/confirmation-number ask.
   The drafter now pulls the "we'll…/we can…" clause from real resolutions and strips
   agent signatures, so the reply is grounded without being repetitive or committing
   to specifics.

10. **Reply drafting is template-scaffolded even when an LLM is available.** The LLM
    (if wired) *rewrites* a safe scaffold and is instructed to stay grounded; it never
    free-generates from scratch. This bounds hallucination and keeps a deterministic
    default fallback. Hard clamps forbid soliciting card numbers/passwords and cap
    length.

11. **One `LLMBackend` interface with a deterministic default.** All LLM use
    (replies *and* judging) goes through `complete(system, user)`. The default backend
    is a deterministic rubric/heuristic that needs no API key; with a key it routes to
    OpenAI/Anthropic via stdlib `urllib`. Stable reproducibility by default, real
    generative capability when available, and zero call-site changes to switch.

12. **Everything is pure standard library.** No numpy/pandas/sklearn dependency —
    TF-IDF, NB, all metrics, and Cohen's κ are implemented from scratch. This keeps the
    project reproducible on any stock Python install; `python cli.py eval` runs in a
    couple of seconds.

13. **Validated the judge instead of trusting it.** Built a separate, balanced,
    hand-labelled reply-verdict set and report Cohen's κ (0.94). Also documented the
    judge's one failure (under-detecting wrong-intent replies) rather than hiding it.

14. **Golden examples are hand-written with vocabulary that's absent from training.**
    Prevents the intent metric from measuring memorization; forces an out-of-
    distribution test. Included deliberately debatable escalation calls so the eval
    surfaces the precision/recall tension.

15. **Shipped a sample corpus in the exact Kaggle schema and documented it clearly.**
    This makes the pipeline runnable end-to-end without a multi-gigabyte download;
    `data_loader` reads the real file unchanged via `TWCS_PATH`. The synthetic-data
    caveat is the #1 item in the report's "what's misleading" section.


16. **A reply only counts as "grounded" when the retrieved exemplar shares the
    predicted intent.** The retriever backs off to unfiltered results for sparse
    intents, which let a low-similarity SkyMiles resolution supply the next step for a
    baggage-policy question ("Great question! *We'll check on those miles.*"). Adding
    the intent-match requirement dropped the reported grounding rate from 86% to 78.5%
    — a metric moving *down* because it became honest. Relevance matters more than a
    flattering grounding percentage.
