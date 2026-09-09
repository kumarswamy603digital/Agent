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
python cli.py setup      
python cli.py eval       
```

The first `eval` trains the learned component (~75s of gradient descent) and caches its
weights under `.cache/`; later runs take ~2s. The cache key includes the corpus size and
every hyperparameter, so any change retrains automatically.

`python cli.py eval` prints the headline tables and writes:

- `results/results.json` — all metrics, machine-readable
- `results/summary.md` — the same, human-readable
- `results/confusion_main.txt` — the intent confusion matrix

### Headline numbers

Computed on the bundled sample corpus (see [Data](#data) below) with the default
zero-dependency backend and judge.

**Intent classification.** The 200-example golden set is split into a **dev half used
for all tuning** and a **held-out test half scored only once, at the end**. The
held-out number is the one to trust:

| split | n | trivial | keyword rules | rule chain | learned student | **shipped ensemble** |
|---|---|---|---|---|---|---|
| dev | 119 | 16.0% | 55.5% | 93.3% | 86.6% | 89.1% |
| **test (held out)** | **81** | **14.8%** | **55.6%** | 79.0% | 74.1% | **81.5%** |
| test ∩ never-inspected | 17 | 17.6% | 70.6% | 88.2% | 76.5% | 88.2% |
| whole set | 200 | 15.5% | 55.5% | 87.5% | 81.5% | **86.0%** |

The main model is an **equal-weight ensemble** of two components with different
inductive biases: a hand-written rule chain and a learned model distilled from it.
Read the table this way — the rule chain looks best on dev *because it was hand-tuned
there*, while the student never saw dev. On held-out test the ensemble beats both, and
its dev→test gap is **7.6 points versus 13.9** for the rule chain alone. Ensembling
recovered the generalization that hand-tuning had cost.

**Escalation decision** (positive class = *escalate*, whole set):

| policy | precision | recall | F1 | auto-handle rate | missed-escalation rate |
|---|---|---|---|---|---|
| trivial (escalate everything) | 0.59 | 1.00 | 0.74 | 0% | 0% |
| simple (intent-only rule) | 0.80 | 0.41 | 0.54 | 60% | 34.5% |
| **full agent** | **0.83** | **0.68** | **0.75** | **49%** | **18.5%** |

**Reply quality** (scored by the judge): 93.0% acceptable, 72.5% grounded in retrieved
history (a reply counts as grounded only when the retrieved exemplar shares the
predicted intent).

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
python cli.py demo                   
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
python cli.py eval          
```

To generate replies and judge them with a hosted model instead of the default backend:

```bash
export SUPPORT_AGENT_LLM=openai           
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
| `data/golden/golden_eval.jsonl` | 200 hand-labelled examples (intent + escalation), split into dev/test by `eval/splits.py`. |
| `data/golden/judge_agreement.jsonl` | 36 hand-labelled reply-quality verdicts used to validate the judge. |
| `data/golden/README.md` | How the golden sets were sampled and labelled. |

---

## Repository layout

```
Agent/
├── cli.py                       
├── src/support_agent/
│   ├── intents.py               
│   ├── data_loader.py           
│   ├── text.py                  
│   ├── vectorizer.py          
│   ├── signals.py
│   ├── features.py
│   ├── classifiers/
│   │   ├── trivial.py           
│   │   ├── rules.py            
│   │   ├── nb.py              
│   │   ├── hybrid.py          
│   │   ├── rules_refined.py
│   │   ├── refined.py
│   │   ├── logreg.py
│   │   └── stacked.py
│   ├── retriever.py            
│   ├── reply.py                 
│   ├── escalation.py          
│   ├── llm/backend.py         
│   └── agent.py                
├── eval/
│   ├── metrics.py              
│   ├── splits.py
│   ├── judge.py                
│   └── run_eval.py             
├── data/                        
├── scripts/
│   ├── generate_sample_data.py  
│   ├── build_golden.py         
│   └── build_judge_set.py       
├── tests/                       
├── report/REPORT.md             
└── DECISIONS.md                 
```

## Tests

```bash
python -m unittest discover -s tests -v
```
