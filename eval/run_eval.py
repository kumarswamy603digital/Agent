#!/usr/bin/env python3
"""Headline evaluation harness.

Runs three evaluations and writes results/ (JSON + Markdown + confusion matrix):

  1. INTENT CLASSIFICATION on the golden set, for 3 systems:
       - trivial baseline  : majority class
       - simple baseline   : keyword rules
       - main model        : Naive Bayes (weakly supervised by the rules)
     Metrics: accuracy, macro-F1, weighted-F1, per-class P/R/F1, confusion.

  2. ESCALATION DECISION on the golden set, for 3 policies:
       - escalate-everything (trivial, maximally safe, useless throughput)
       - intent-only rule policy (simple)
       - full agent policy (NB confidence + risk signals)
     Metrics: precision/recall/F1 for the ESCALATE class + accuracy + the
     operational quantities (auto-handle rate, missed-escalation rate).

  3. REPLY QUALITY via the judge on the golden set (avg rubric + % acceptable +
     % grounded), PLUS judge<->human agreement (Cohen's kappa) on the
     hand-labelled judge_agreement set — so the judge's trustworthiness is
     measured, not assumed.

Everything is deterministic with the default backend. Run:  python eval/run_eval.py
"""

from __future__ import annotations

import json
import os
import sys
import time

# make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from support_agent.agent import SupportAgent
from support_agent.classifiers import (
    MajorityClassifier, RuleClassifier, RefinedClassifier, RULES_V2,
)
from support_agent.config import Config
from support_agent.data_loader import build_threads
from support_agent.escalation import ESCALATE_INTENTS
from support_agent.intents import INTENTS
from eval import metrics as M
from eval.judge import get_judge
from eval.splits import round2_only, split_golden
from support_agent.llm.backend import get_backend

GOLDEN = "data/golden/golden_eval.jsonl"
JUDGESET = "data/golden/judge_agreement.jsonl"
RESULTS_DIR = "results"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def train_classifiers(cfg):
    threads = build_threads(cfg.data_path, cfg.brand)
    X = [t.customer_text for t in threads]
    # Weak supervision: the corrected keyword table labels the training corpus.
    weak = RuleClassifier(rules_table=RULES_V2).fit(X, ["general_info"] * len(X))
    y_weak = [weak.predict(t).label for t in X]
    maj = MajorityClassifier().fit(X, y_weak)
    # Baseline uses the FROZEN original keyword table (not the corrected one), so
    # the main-model-vs-baseline comparison is not flattered by shared fixes.
    rules = RuleClassifier().fit(X, y_weak)
    main = RefinedClassifier(
        rule_weight=cfg.rule_weight, nb_alpha=cfg.nb_alpha,
        nb_min_df=cfg.nb_min_df, nb_ngram_range=cfg.nb_ngram_range,
    ).fit(X, y_weak)
    return threads, maj, rules, main


def eval_intents(golden, maj, rules, main):
    y_true = [g["intent"] for g in golden]
    texts = [g["text"] for g in golden]
    out = {}
    for name, clf in [("trivial_majority", maj), ("simple_rules", rules),
                      ("main_refined", main)]:
        y_pred = [clf.predict(t).label for t in texts]
        out[name] = {
            "accuracy": M.accuracy(y_true, y_pred),
            "macro_f1": M.macro_f1(y_true, y_pred, INTENTS),
            "weighted_f1": M.weighted_f1(y_true, y_pred, INTENTS),
            "per_class": M.per_class_prf(y_true, y_pred, INTENTS),
            "_y_pred": y_pred,
        }
    out["_y_true"] = y_true
    out["_confusion_main"] = M.confusion_matrix(y_true, out["main_refined"]["_y_pred"], INTENTS)
    return out


def eval_intents_by_split(maj, rules, main):
    """Report intent accuracy on dev, held-out test, and the never-inspected slice.

    Tuning used the dev split only. `test` is the headline generalization number;
    `test_unseen_batch` is the most conservative reading of all (examples authored
    to broaden coverage and never looked at during error analysis).
    """
    dev, test = split_golden()
    unseen = round2_only(test)
    systems = [("trivial_majority", maj), ("simple_rules", rules), ("main_refined", main)]
    out = {}
    for split_name, ds in [("dev", dev), ("test", test), ("test_unseen_batch", unseen)]:
        yt = [g["intent"] for g in ds]
        entry = {"n": len(ds)}
        for name, clf in systems:
            yp = [clf.predict(g["text"]).label for g in ds]
            entry[name] = {
                "accuracy": M.accuracy(yt, yp),
                "macro_f1": M.macro_f1(yt, yp, INTENTS),
            }
        out[split_name] = entry
    return out


def eval_escalation(golden, cfg, agent):
    y_true = [1 if g["escalate"] else 0 for g in golden]
    texts = [g["text"] for g in golden]

    # trivial: escalate everything
    triv = [1] * len(texts)
    # simple: intent-only rule policy (rule classifier intent in ESCALATE_INTENTS)
    rc = agent._weak_labeler
    simple = [1 if rc.predict(t).label in ESCALATE_INTENTS else 0 for t in texts]
    # full agent policy
    full = [1 if agent.handle(t).escalate else 0 for t in texts]

    def pack(pred):
        d = M.binary_prf(y_true, pred, positive=1)
        n = len(pred)
        d["auto_handle_rate"] = sum(1 for p in pred if p == 0) / n
        # missed escalation = gold escalate but predicted auto (the dangerous error)
        d["missed_escalation_rate"] = d["fn"] / n
        return d

    return {
        "trivial_escalate_all": pack(triv),
        "simple_intent_rule": pack(simple),
        "full_agent": pack(full),
        "_y_true": y_true,
        "_full_pred": full,
    }


def eval_reply_quality(golden, agent, judge):
    dims = {"relevance": 0, "groundedness": 0, "safety": 0, "tone": 0}
    acc = 0; grounded = 0; n = 0
    unacceptable = []
    for g in golden:
        resp = agent.handle(g["text"])
        v = judge.judge(g["text"], resp.reply)
        for k in dims:
            dims[k] += getattr(v, k)
        acc += v.acceptable
        grounded += 1 if resp.reply_grounded else 0
        n += 1
        if not v.acceptable:
            unacceptable.append({"text": g["text"], "reply": resp.reply, "rationale": v.rationale})
    return {
        "n": n,
        "avg_relevance": dims["relevance"] / n,
        "avg_groundedness": dims["groundedness"] / n,
        "avg_safety": dims["safety"] / n,
        "avg_tone": dims["tone"] / n,
        "pct_acceptable": acc / n,
        "pct_grounded_in_history": grounded / n,
        "unacceptable_examples": unacceptable[:10],
    }


def eval_judge_agreement(judgeset, judge):
    human = [r["human_acceptable"] for r in judgeset]
    jpred = [judge.judge(r["text"], r["reply"]).acceptable for r in judgeset]
    prf = M.binary_prf(human, jpred, positive=1)
    kappa = M.cohen_kappa(human, jpred)
    disagreements = [
        {"text": r["text"], "reply": r["reply"], "human": r["human_acceptable"],
         "judge": jp, "reason": r["reason"]}
        for r, jp in zip(judgeset, jpred) if r["human_acceptable"] != jp
    ]
    return {"n": len(human), "agreement_accuracy": prf["accuracy"],
            "cohen_kappa": kappa, "judge_precision": prf["precision"],
            "judge_recall": prf["recall"], "judge_f1": prf["f1"],
            "disagreements": disagreements}


def main():
    t0 = time.time()
    cfg = Config()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    golden = load_jsonl(GOLDEN)
    judgeset = load_jsonl(JUDGESET)

    threads, maj, rules, nb = train_classifiers(cfg)
    agent = SupportAgent(cfg).build(threads)
    judge = get_judge(get_backend(cfg.llm_backend, cfg.llm_model))

    intent_res = eval_intents(golden, maj, rules, nb)
    split_res = eval_intents_by_split(maj, rules, nb)
    esc_res = eval_escalation(golden, cfg, agent)
    reply_res = eval_reply_quality(golden, agent, judge)
    judge_res = eval_judge_agreement(judgeset, judge)

    elapsed = time.time() - t0
    results = {
        "config": {"brand": cfg.brand, "data_path": cfg.data_path,
                   "n_train_threads": len(threads), "n_golden": len(golden),
                   "llm_backend": cfg.llm_backend, "judge": judge.name,
                   "abstain_threshold": cfg.abstain_threshold},
        "intent_classification": {k: v for k, v in intent_res.items() if not k.startswith("_")},
        "intent_by_split": split_res,
        "escalation": {k: v for k, v in esc_res.items() if not k.startswith("_")},
        "reply_quality": reply_res,
        "judge_agreement": judge_res,
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # confusion matrix file
    with open(os.path.join(RESULTS_DIR, "confusion_main.txt"), "w") as f:
        f.write("Main model (refined hybrid) confusion matrix (rows=true, cols=pred)\n\n")
        f.write(M.format_confusion(intent_res["_confusion_main"], INTENTS))
        f.write("\n")

    _write_markdown(results, intent_res, esc_res)
    _print_console(results, intent_res, esc_res)
    _print_splits(split_res)
    print(f"\nWrote {RESULTS_DIR}/results.json, summary.md, confusion_main.txt")
    print(f"Total eval time: {elapsed:.1f}s")


def _fmt_pct(x):
    return f"{100*x:5.1f}%"


def _print_splits(split_res):
    print("\n[5] INTENT ACCURACY BY SPLIT  (tuned on dev only; test is held out)")
    print("  %-18s %5s %10s %10s %12s" % ("split", "n", "trivial", "rules", "main"))
    for name in ["dev", "test", "test_unseen_batch"]:
        r = split_res[name]
        print("  %-18s %5d %10s %10s %12s" % (
            name, r["n"],
            _fmt_pct(r["trivial_majority"]["accuracy"]),
            _fmt_pct(r["simple_rules"]["accuracy"]),
            _fmt_pct(r["main_refined"]["accuracy"]),
        ))
    print("  (test_unseen_batch = test items authored to broaden coverage and never")
    print("   inspected during error analysis — the most conservative estimate)")


def _print_console(results, intent_res, esc_res):
    print("\n" + "=" * 70)
    print("HEADLINE RESULTS  (brand=%s, backend=%s, judge=%s)" %
          (results["config"]["brand"], results["config"]["llm_backend"], results["config"]["judge"]))
    print("=" * 70)
    print("\n[1] INTENT CLASSIFICATION (golden n=%d)" % results["config"]["n_golden"])
    print("  %-20s %8s %9s %11s" % ("system", "acc", "macroF1", "weightedF1"))
    for name in ["trivial_majority", "simple_rules", "main_refined"]:
        r = results["intent_classification"][name]
        print("  %-20s %8s %9s %11s" % (name, _fmt_pct(r["accuracy"]),
              f"{r['macro_f1']:.3f}", f"{r['weighted_f1']:.3f}"))
    print("\n[2] ESCALATION DECISION (positive=escalate)")
    print("  %-22s %8s %8s %8s %8s %14s" % ("policy", "prec", "recall", "f1", "acc", "missed_esc"))
    for name in ["trivial_escalate_all", "simple_intent_rule", "full_agent"]:
        r = results["escalation"][name]
        print("  %-22s %8s %8s %8s %8s %14s" % (
            name, f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}",
            _fmt_pct(r["accuracy"]), _fmt_pct(r["missed_escalation_rate"])))
    rq = results["reply_quality"]
    print("\n[3] REPLY QUALITY (judge=%s, golden n=%d)" % (results["config"]["judge"], rq["n"]))
    print("  relevance=%.2f grounded=%.2f safety=%.2f tone=%.2f | acceptable=%s grounded_in_history=%s" % (
        rq["avg_relevance"], rq["avg_groundedness"], rq["avg_safety"], rq["avg_tone"],
        _fmt_pct(rq["pct_acceptable"]), _fmt_pct(rq["pct_grounded_in_history"])))
    ja = results["judge_agreement"]
    print("\n[4] JUDGE <-> HUMAN AGREEMENT (n=%d)" % ja["n"])
    print("  accuracy=%s  Cohen_kappa=%.3f  judge_P/R/F1=%.2f/%.2f/%.2f" % (
        _fmt_pct(ja["agreement_accuracy"]), ja["cohen_kappa"],
        ja["judge_precision"], ja["judge_recall"], ja["judge_f1"]))


def _write_markdown(results, intent_res, esc_res):
    lines = []
    a = lines.append
    cfg = results["config"]
    a(f"# Evaluation Results — {cfg['brand']} support agent\n")
    a(f"- Backend: `{cfg['llm_backend']}` | Judge: `{cfg['judge']}` | "
      f"train threads: {cfg['n_train_threads']} | golden: {cfg['n_golden']}\n")
    a(f"- Total eval time: **{results['elapsed_seconds']}s**\n")

    a("\n## 1. Intent classification\n")
    a("| system | accuracy | macro-F1 | weighted-F1 |")
    a("|---|---|---|---|")
    for name in ["trivial_majority", "simple_rules", "main_refined"]:
        r = results["intent_classification"][name]
        a(f"| {name} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | {r['weighted_f1']:.3f} |")

    a("\n### Per-class (main Naive Bayes)\n")
    a("| intent | precision | recall | f1 | support |")
    a("|---|---|---|---|---|")
    for lab in INTENTS:
        pc = results["intent_classification"]["main_refined"]["per_class"][lab]
        a(f"| {lab} | {pc['precision']:.2f} | {pc['recall']:.2f} | {pc['f1']:.2f} | {pc['support']} |")

    a("\n## 2. Escalation decision (positive class = escalate)\n")
    a("| policy | precision | recall | f1 | accuracy | auto-handle rate | missed-escalation rate |")
    a("|---|---|---|---|---|---|---|")
    for name in ["trivial_escalate_all", "simple_intent_rule", "full_agent"]:
        r = results["escalation"][name]
        a(f"| {name} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | "
          f"{r['accuracy']:.3f} | {r['auto_handle_rate']:.3f} | {r['missed_escalation_rate']:.3f} |")

    rq = results["reply_quality"]
    a("\n## 3. Reply quality (judge)\n")
    a(f"- avg relevance **{rq['avg_relevance']:.2f}/2**, groundedness "
      f"**{rq['avg_groundedness']:.2f}/2**, safety **{rq['avg_safety']:.2f}/2**, "
      f"tone **{rq['avg_tone']:.2f}/2**")
    a(f"- **{rq['pct_acceptable']*100:.1f}%** acceptable, "
      f"**{rq['pct_grounded_in_history']*100:.1f}%** grounded in retrieved history\n")

    ja = results["judge_agreement"]
    a("\n## 4. Judge ↔ human agreement\n")
    a(f"- n={ja['n']}, agreement accuracy **{ja['agreement_accuracy']*100:.1f}%**, "
      f"**Cohen's κ = {ja['cohen_kappa']:.3f}**")
    a(f"- judge precision/recall/F1 vs human = "
      f"{ja['judge_precision']:.2f}/{ja['judge_recall']:.2f}/{ja['judge_f1']:.2f}")
    if ja["disagreements"]:
        a("\nDisagreements:")
        for d in ja["disagreements"]:
            a(f"- human={d['human']} judge={d['judge']} — “{d['text'][:60]}…” → “{d['reply'][:60]}…”")

    with open(os.path.join(RESULTS_DIR, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
