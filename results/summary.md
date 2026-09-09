# Evaluation Results — Delta support agent

- Backend: `heuristic` | Judge: `heuristic_judge` | train threads: 2600 | golden: 200

- Total eval time: **72.79s**


## 1. Intent classification

| system | accuracy | macro-F1 | weighted-F1 |
|---|---|---|---|
| trivial_majority | 0.155 | 0.030 | 0.042 |
| simple_rules | 0.555 | 0.555 | 0.541 |
| main_ensemble | 0.860 | 0.862 | 0.860 |

### Per-class (main Naive Bayes)

| intent | precision | recall | f1 | support |
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

## 2. Escalation decision (positive class = escalate)

| policy | precision | recall | f1 | accuracy | auto-handle rate | missed-escalation rate |
|---|---|---|---|---|---|---|
| trivial_escalate_all | 0.585 | 1.000 | 0.738 | 0.585 | 0.000 | 0.000 |
| simple_intent_rule | 0.800 | 0.410 | 0.542 | 0.595 | 0.700 | 0.345 |
| full_agent | 0.833 | 0.684 | 0.751 | 0.735 | 0.520 | 0.185 |

## 3. Reply quality (judge)

- avg relevance **1.86/2**, groundedness **2.00/2**, safety **2.00/2**, tone **2.00/2**
- **93.0%** acceptable, **72.5%** grounded in retrieved history


## 4. Judge ↔ human agreement

- n=36, agreement accuracy **97.2%**, **Cohen's κ = 0.944**
- judge precision/recall/F1 vs human = 0.95/1.00/0.97

Disagreements:
- human=0 judge=1 — “@Delta want to change my flight to Sunday…” → “We're so sorry about your damaged bag. Please DM your file r…”
