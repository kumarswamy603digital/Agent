# Evaluation Results — Delta support agent

- Backend: `heuristic` | Judge: `heuristic_judge` | train threads: 2000 | golden: 150

- Total eval time: **0.7s**


## 1. Intent classification

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| trivial_majority | 0.160 | 0.031 | 0.044 |
| simple_rules | 0.553 | 0.554 | 0.537 |
| main_hybrid | 0.660 | 0.653 | 0.644 |

### Per-class (main Naive Bayes)

| intent | precision | recall | f1 | support |
|---|---|---|---|---|
| flight_disruption | 0.63 | 0.79 | 0.70 | 24 |
| baggage | 0.68 | 0.87 | 0.76 | 15 |
| booking_change | 0.54 | 0.44 | 0.48 | 16 |
| refund_billing | 0.82 | 0.78 | 0.80 | 18 |
| check_in_boarding | 0.53 | 0.71 | 0.61 | 14 |
| loyalty_program | 0.87 | 0.93 | 0.90 | 14 |
| complaint_feedback | 1.00 | 0.19 | 0.32 | 16 |
| praise | 0.89 | 0.73 | 0.80 | 11 |
| general_info | 0.48 | 0.55 | 0.51 | 22 |

## 2. Escalation decision (positive class = escalate)

| policy | precision | recall | f1 | accuracy | auto-handle rate | missed-escalation rate |
|---|---|---|---|---|---|---|
| trivial_escalate_all | 0.587 | 1.000 | 0.739 | 0.587 | 0.000 | 0.000 |
| simple_intent_rule | 0.833 | 0.341 | 0.484 | 0.573 | 0.760 | 0.387 |
| full_agent | 0.822 | 0.682 | 0.745 | 0.727 | 0.513 | 0.187 |

## 3. Reply quality (judge)

- avg relevance **1.93/2**, groundedness **2.00/2**, safety **2.00/2**, tone **2.00/2**
- **96.7%** acceptable, **83.3%** grounded in retrieved history


## 4. Judge ↔ human agreement

- n=36, agreement accuracy **97.2%**, **Cohen's κ = 0.944**
- judge precision/recall/F1 vs human = 0.95/1.00/0.97

Disagreements:
- human=0 judge=1 — “@Delta want to change my flight to Sunday…” → “We're so sorry about your damaged bag. Please DM your file r…”
