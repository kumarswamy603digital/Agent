# Delta Twitter Support Agent

An AI customer-support agent for a single brand (**Delta Air Lines**), built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Given an inbound customer tweet, the agent:

1. **Classifies intent** into a small, action-oriented taxonomy (9 intents).
2. **Drafts a reply** grounded in how Delta has *historically* resolved similar issues.
3. **Decides auto-handle vs. escalate** to a human — with a stated, auditable reason.

The emphasis of this project is trust: showing, with evidence, that the agent is
reliable enough to act on. That focus drives the design — a rigorous **evaluation
harness**, a hand-labelled **golden set**, an LLM-as-judge whose agreement with humans
is **measured** rather than assumed, honest **baselines**, and a candid failure
analysis.

> See [`report/REPORT.md`](report/REPORT.md) for the full write-up (problem framing,
> results, failure modes, the "what's misleading about my headline number" section,
> and next steps) and [`DECISIONS.md`](DECISIONS.md) for the decision log.

---

## Quick start — reproduce the headline results (well under 15 minutes)

**Requirements:** Python 3.10+ only. No third-party dependencies.

```bash
# from the repo root (the 'Agent' directory)
python cli.py setup      # generate the bundled sample + golden sets (deterministic)
python cli.py eval       # run the full evaluation harness
```

`python cli.py eval` prints the headline tables and writes:

- `results/results.json` — all metrics, machine-readable
- `results/summary.md` — the same, human-readable
- `results/confusion_main.txt` — the intent confusion matrix

### Headline numbers

Computed on the bundled sample corpus (see [Data](#data) below) with the default
zero-dependency backend and judge.

**Intent classification** (golden set, n=150):

| system | accuracy | macro-F1 |
|---|---|---|
| trivial (majority class) | 16.0% | 0.03 |
| simple (keyword rules) | 55.3% | 0.55 |
| **main (rule + Naive Bayes hybrid)** | **66.0%** | **0.65** |

**Escalation decision** (positive class = *escalate*):

| policy | precision | recall | F1 | auto-handle rate | missed-escalation rate |
|---|---|---|---|---|---|
| trivial (escalate everything) | 0.59 | 1.00 | 0.74 | 0% | 0% |
| simple (intent-only rule) | 0.83 | 0.34 | 0.48 | 66% | 38.7% |
| **full agent** | **0.82** | **0.68** | **0.75** | **51%** | **18.7%** |

**Reply quality** (scored by the judge): 96.7% acceptable, 83% grounded in retrieved history.

**Judge trustworthiness**: on 36 hand-labelled reply verdicts, judge↔human agreement is
**97.2%**, **Cohen's κ = 0.94**.

---

## Design notes

Two deliberate engineering choices keep the project portable and easy to reproduce,
and both are worth stating up front because they affect how the numbers should be read.

**1. Zero third-party dependencies.** The entire pipeline — TF-IDF, Naive Bayes,
retrieval, all evaluation metrics, and Cohen's κ — is implemented against the Python
standard library. This makes the headline results reproducible on any machine with a
stock Python install, with no environment setup and no version drift. The trade-off is
that the models are intentionally simple; heavier components (transformer embeddings,
a fine-tuned classifier) are natural upgrades and are discussed in the report.

**2. A bundled sample corpus.** Rather than requiring a multi-gigabyte Kaggle download
to run anything, the repo ships a self-contained sample dataset in the *exact* Kaggle
`twcs.csv` schema (`data/raw/twcs_sample_delta.csv`). `data_loader.py` reads the real
`twcs.csv` unchanged — pointing at the full dataset is a one-line environment change
(see [below](#running-on-the-real-kaggle-data)).

> **Reading the numbers honestly:** the headline metrics above are computed on the
> bundled sample, which is cleaner and less diverse than the full Twitter corpus.
> They should be treated as an upper bound until re-measured on the real data. This is
> the first item in the report's
> ["What is misleading about my headline number?"](report/REPORT.md#6-what-is-misleading-about-my-headline-number)
> section.

All calls to a language model go through a single interface (`llm/backend.py`). The
default backend is deterministic and requires no API key, which is what makes the
evaluation stable run-to-run; `openai` and `anthropic` adapters are implemented and
activate when an API key is present.

---

## Try it on individual messages

```bash
python cli.py demo                       # a curated set of examples
python cli.py handle "@Delta my flight got cancelled, stuck at JFK, need to rebook"
```

Example output:

```
IN : @Delta my flight got cancelled, stuck at JFK, need to rebook
INTENT: flight_disruption  (conf=1.00)   ROUTE: ESCALATE
WHY : Escalate to human — time_critical_or_vulnerable; high_stakes_intent:flight_disruption.
REPLY: So sorry for the disruption to your travel plans — that's stressful. We'll check
       the next available flights. Please DM us your confirmation number so we can help.
GROUNDED: True | top evidence (sim=0.30): "Apologies for the delay to your travel plans..."
```

---

## Running on the real Kaggle data

```bash
# download twcs.csv from Kaggle, then:
export TWCS_PATH=/path/to/twcs.csv
python cli.py eval           # thread reconstruction + training now use real Delta threads
```

To generate replies and judge them with a hosted model instead of the default backend:

```bash
export SUPPORT_AGENT_LLM=openai            # or anthropic
export OPENAI_API_KEY=sk-...
export SUPPORT_AGENT_LLM_MODEL=gpt-4o-mini
python cli.py demo
```

Nothing else changes: the same `build_threads`, classifier, retriever, escalation
policy, and evaluation harness run against real data.

---

## Data

| File | What it is |
|---|---|
| `data/raw/twcs_sample_delta.csv` | Self-contained sample corpus in the Kaggle `twcs.csv` schema, produced by `scripts/generate_sample_data.py`. Used so the pipeline runs end-to-end without the full download. |
| `data/golden/golden_eval.jsonl` | 150 hand-labelled examples (intent + escalation). |
| `data/golden/judge_agreement.jsonl` | 36 hand-labelled reply-quality verdicts used to validate the judge. |
| `data/golden/README.md` | How the golden sets were sampled and labelled. |

---

## Repository layout

```
Agent/
├── cli.py                       # entrypoint: setup | handle | demo | eval
├── src/support_agent/
│   ├── intents.py               # the 9-intent taxonomy + abstain label
│   ├── data_loader.py           # twcs.csv reader + thread reconstruction (schema-faithful)
│   ├── text.py                  # tweet normalization / tokenization
│   ├── vectorizer.py            # TF-IDF + cosine
│   ├── classifiers/
│   │   ├── trivial.py           # baseline 1: majority class
│   │   ├── rules.py             # baseline 2: keyword rules (also the weak-supervision teacher)
│   │   ├── nb.py                # multinomial Naive Bayes
│   │   └── hybrid.py            # main model: rule-prior + Naive Bayes blend
│   ├── retriever.py             # retrieves historical resolved threads for grounding
│   ├── reply.py                 # grounded, brand-voiced reply drafting
│   ├── escalation.py            # auto-handle vs escalate + stated reason
│   ├── llm/backend.py           # pluggable LLM interface: default | openai | anthropic
│   └── agent.py                 # end-to-end orchestration
├── eval/
│   ├── metrics.py               # accuracy, macro-F1, confusion, Cohen's κ
│   ├── judge.py                 # LLM-as-judge rubric + default rubric judge
│   └── run_eval.py              # the harness that produces results/
├── data/                        # sample corpus + golden sets (see table above)
├── scripts/
│   ├── generate_sample_data.py  # builds the sample corpus
│   ├── build_golden.py          # emits the golden set (labels are inline & reviewable)
│   └── build_judge_set.py       # emits the judge-agreement set
├── tests/                       # unit tests for core invariants
├── report/REPORT.md             # the full report
└── DECISIONS.md                 # the decision log
```

## Tests

```bash
python -m unittest discover -s tests -v
```
