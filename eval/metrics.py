"""Classification / agreement metrics, pure stdlib (no sklearn)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple


def accuracy(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def per_class_prf(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    tp = Counter(); fp = Counter(); fn = Counter()
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    out = {}
    for lab in labels:
        p = tp[lab] / (tp[lab] + fp[lab]) if (tp[lab] + fp[lab]) else 0.0
        r = tp[lab] / (tp[lab] + fn[lab]) if (tp[lab] + fn[lab]) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        out[lab] = {"precision": p, "recall": r, "f1": f, "support": tp[lab] + fn[lab]}
    return out


def macro_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    prf = per_class_prf(y_true, y_pred, labels)
    # macro over labels that actually appear in y_true
    present = [l for l in labels if prf[l]["support"] > 0]
    if not present:
        return 0.0
    return sum(prf[l]["f1"] for l in present) / len(present)


def weighted_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> float:
    prf = per_class_prf(y_true, y_pred, labels)
    total = sum(prf[l]["support"] for l in labels)
    if not total:
        return 0.0
    return sum(prf[l]["f1"] * prf[l]["support"] for l in labels) / total


def confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> List[List[int]]:
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            m[idx[t]][idx[p]] += 1
    return m


def binary_prf(y_true: List[int], y_pred: List[int], positive: int = 1) -> Dict[str, float]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p != positive)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(y_true) if y_true else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def cohen_kappa(y1: List, y2: List) -> float:
    """Cohen's kappa between two raters (labels can be str or int)."""
    n = len(y1)
    if n == 0:
        return 0.0
    po = sum(a == b for a, b in zip(y1, y2)) / n
    labels = set(y1) | set(y2)
    c1 = Counter(y1); c2 = Counter(y2)
    pe = sum((c1[l] / n) * (c2[l] / n) for l in labels)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def format_confusion(m: List[List[int]], labels: List[str]) -> str:
    short = [l[:6] for l in labels]
    header = "true\\pred".ljust(12) + " ".join(s.rjust(6) for s in short)
    lines = [header]
    for i, lab in enumerate(labels):
        row = lab[:12].ljust(12) + " ".join(str(m[i][j]).rjust(6) for j in range(len(labels)))
        lines.append(row)
    return "\n".join(lines)
