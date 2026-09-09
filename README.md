# Delta Twitter Support Agent

An AI customer-support agent for a single brand (**Delta Air Lines**) built on the
[Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. Given an inbound customer tweet, it:

1. **Classifies intent** into a small, action-oriented taxonomy (9 intents).
2. **Drafts a reply** grounded in how Delta has *historically* resolved similar issues.
3. **Decides auto-handle vs. escalate** to a human — with a stated, auditable reason.

The whole point of the project is the third bullet of the brief: *convincing you the
agent is good enough to trust*. So the emphasis is on the **evaluation harness**, the
**golden set**, a **measured** LLM-as-judge, honest **baselines**, and a candid
[failure analysis + "what's misleading about my headline number"](report/REPORT.md).

> **Read [`report/REPORT.md`](report/REPORT.md) for the full write-up** (framing,
> results, failure modes, the mandatory "misleading headline" section, next steps)
> and [`DECISIONS.md`](DECISIONS.md) for the decision log.

---

## ⚠️ Important context: this repo runs fully offline

It was built and validated in a **sandbox with no internet access, no ability to
`pip install`, and no reachable LLM API.** Two consequences, both handled by design:

| Constraint | How the repo handles it |
|---|---|
| Can't download the 3M-row Kaggle file here | Ships a **synthetic sample** (`data/raw/twcs_sample_delta.csv`) in the *exact* Kaggle schema. `data_loader.py` reads the real `twcs.csv` **unchanged** — just set `TWCS_PATH`. |
| Can't call a hosted LLM here | All LLM use goes through one interface (`llm/backend.py`). The default `heuristic` backend is deterministic and offline; `openai`/`anthropic` adapters are real (stdlib `urllib`) and activate with an API key. |
| No numpy/pandas/sklearn available | **Everything is pure Python standard library** — TF-IDF, Naive Bayes, metrics, Cohen's κ are all implemented from scratch. |

This is disclosed prominently because it directly shapes the headline numbers — see
the [**"What is misleading about my headline number?"**](report/REPORT.md#5-what-is-misleading-about-my-headline-number) section, which treats it as the #1 caveat.

---

## Reproduce the headline results (< 15 minutes; actually ~2 seconds)

**Requirements:** Python 3.10+ only. No dependencies. No network.

```bash
# from the repo root (the 'Agent' directory)
python cli.py setup      # (re)generate the synthetic sample + golden sets (deterministic)
python cli.py eval       # run the full evaluation harness
```

`python cli.py eval` prints the headline tables and writes:
- `results/results.json` — all metrics, machine-readable
- `results/summary.md` — the same, human-readable
- `results/confusion_main.txt` — intent confusion matrix

### Headline numbers (synthetic sample, offline `heuristic` backend + judge)

**Intent classification** (golden set, n=150):

| system | accuracy | macro-F1 |
|---|---|---|
| trivial (majority class) | 16.0% | 0.03 |
| simple (keyword rules) | 55.3% | 0.55 |
| **main (rule+NB hybrid)** | **66.0%** | **0.65** |

**Escalation decision** (positive class = *escalate*):

| policy | precision | recall | F1 | auto-handle rate | missed-escalation rate |
|---|---|---|---|---|---|
| trivial (escalate everything) | 0.59 | 1.00 | 0.74 | 0% | 0% |
| simple (intent-only rule) | 0.83 | 0.34 | 0.48 | 66% | 38.7% |
| **full agent** | **0.82** | **0.68** | **0.75** | **51%** | **18.7%** |

**Reply quality** (measured by the judge): 96.7% acceptable, 83% grounded in retrieved history.

**Judge trustworthiness**: on 36 hand-labelled reply verdicts, judge↔human agreement is
**97.2%**, **Cohen's κ = 0.94**.

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

## Run on the **real** Kaggle data / with a **real** LLM

```bash
# 1) real dataset: download twcs.csv from Kaggle, then:
export TWCS_PATH=/path/to/twcs.csv
python cli.py eval           # thread reconstruction + training now use real Delta threads

# 2) real generative replies + LLM judge (needs network + key):
export SUPPORT_AGENT_LLM=openai            # or anthropic
export OPENAI_API_KEY=sk-...
export SUPPORT_AGENT_LLM_MODEL=gpt-4o-mini
python cli.py demo
```

Nothing else changes: the same `build_threads`, classifier, retriever, escalation
policy, and eval harness run against real data.

---

## Repository layout

```
Agent/
├── cli.py                       # entrypoint: setup | handle | demo | eval
├── src/support_agent/
│   ├── intents.py               # the 9-intent taxonomy + abstain label
│   ├── data_loader.py           # twcs.csv reader + thread reconstruction (schema-faithful)
│   ├── text.py                  # tweet normalization / tokenization
│   ├── vectorizer.py            # TF-IDF + cosine (from scratch)
│   ├── classifiers/
│   │   ├── trivial.py           # baseline 1: majority class
│   │   ├── rules.py             # baseline 2: keyword rules (also the weak-supervision teacher)
│   │   ├── nb.py                # multinomial Naive Bayes (from scratch)
│   │   └── hybrid.py            # MAIN model: rule-prior + NB blend
│   ├── retriever.py             # retrieves historical resolved threads for grounding
│   ├── reply.py                 # grounded, brand-voiced reply drafting
│   ├── escalation.py            # auto-handle vs escalate + stated reason
│   ├── llm/backend.py           # pluggable LLM: heuristic (offline) | openai | anthropic
│   └── agent.py                 # end-to-end orchestration
├── eval/
│   ├── metrics.py               # accuracy, macro-F1, confusion, Cohen's κ (from scratch)
│   ├── judge.py                 # LLM-as-judge rubric + offline heuristic judge
│   └── run_eval.py              # the harness that produces results/
├── data/
│   ├── raw/twcs_sample_delta.csv      # synthetic sample in Kaggle schema
│   └── golden/
│       ├── golden_eval.jsonl          # 150 hand-labelled intent+escalation examples
│       ├── judge_agreement.jsonl      # 36 hand-labelled reply-quality verdicts
│       └── README.md                  # how the golden sets were sampled & labelled
├── scripts/
│   ├── generate_sample_data.py  # builds the synthetic corpus (documented provenance)
│   ├── build_golden.py          # emits the golden set (labels are inline & reviewable)
│   └── build_judge_set.py       # emits the judge-agreement set
├── tests/                       # a few stdlib unittests for core invariants
├── report/REPORT.md             # the full report
└── DECISIONS.md                 # the decision log
```

## Tests

```bash
python -m unittest discover -s tests -v
```
