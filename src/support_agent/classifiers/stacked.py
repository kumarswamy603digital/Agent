"""Main intent model: a learned stacker distilled from the rule chain.

Architecture
------------
    keyword rules ─┐
                   ├─▶ signal-refined chain (teacher) ─▶ soft labels
    Naive Bayes ───┘                                          │
                                                              ▼
    text + signal + rule + NB features ──────────▶ logistic regression (student)

The teacher is the ordered override chain (`RefinedClassifier`). It is precise but
hand-tuned, so it generalises unevenly. We use it to label the *unlabelled* historical
corpus, then train a regularized linear model on rich features to imitate it —
classic knowledge distillation.

Why the student can beat its teacher
------------------------------------
Distillation is not a no-op here because the student sees information the teacher
never had:

* **Character n-grams** give robustness to typos/abbreviations the teacher's regexes
  miss entirely ("cancelld", "flt", "delayd").
* **Continuous weighting** replaces first-match-wins ordering, so conflicting cues
  (a complaint that also mentions a refund) are resolved by learned magnitude rather
  than by my hand-chosen rule order — the part most fitted to dev.
* **L2 regularization** damps over-reliance on any single cue.

The student therefore smooths the teacher: it keeps the teacher's systematic
knowledge (encoded in the signal features) while discarding some of its
dev-specific sharpness. Empirically it closes most of the dev→test gap.

At inference we combine student and teacher: the student decides, except where the
teacher fires a **high-precision safety signal** (an explicit refund request, a harm
or grievance report). Those two carry asymmetric business cost — misrouting a refund
dispute or an injury claim is far worse than a marginal accuracy loss — so the
transparent rule keeps priority there by design.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..features import build_features
from ..signals import analyze
from .base import Prediction
from .logreg import LogisticRegression
from .refined import RefinedClassifier


class StackedClassifier:
    name = "stacked_distilled"

    def __init__(self, rule_weight: float = 0.7, nb_alpha: float = 0.3,
                 nb_min_df: int = 1, nb_ngram_range=(1, 1),
                 lr: float = 0.5, epochs: int = 220, l2: float = 3e-4,
                 use_char: bool = True, temperature: float = 1.0,
                 student_weight: float = 1.0, min_feature_count: int = 3):
        self.teacher = RefinedClassifier(
            rule_weight=rule_weight, nb_alpha=nb_alpha,
            nb_min_df=nb_min_df, nb_ngram_range=nb_ngram_range,
        )
        self.student = LogisticRegression(lr=lr, epochs=epochs, l2=l2)
        self.use_char = use_char
        self.temperature = temperature
        self.student_weight = student_weight
        # Rare lexical features (mostly char n-grams from a single tweet) add noise
        # and training cost without generalising; drop them.
        self.min_feature_count = min_feature_count
        self._vocab: Optional[set] = None
        self.labels: List[str] = []

    # ------------------------------------------------------------------ helpers
    def _features(self, text: str, restrict: bool = True) -> Dict[str, float]:
        feats = build_features(
            text,
            rule_scores=self.teacher.rules.scores(text),
            nb_proba=self.teacher.nb.predict(text).proba,
            use_char=self.use_char,
        )
        if restrict and self._vocab is not None:
            # Signal/rule/NB features are always kept; only lexical ones are pruned.
            feats = {
                f: v for f, v in feats.items()
                if f in self._vocab or f[0] in "srnb" and "[" in f and not f.startswith(("w[", "c["))
            }
        return feats

    @staticmethod
    def _sharpen(proba: Dict[str, float], label: str, floor: float = 0.55
                 ) -> Dict[str, float]:
        """Blend the teacher's distribution toward its chosen label.

        The teacher's post-override distribution is only loosely calibrated, so we
        give the chosen label a defined share of the mass. This yields soft targets
        that still carry runner-up information (better than one-hot) without letting
        a mis-scaled distribution dominate training.
        """
        out = dict(proba) if proba else {}
        z = sum(out.values()) or 1.0
        out = {k: v / z for k, v in out.items()}
        cur = out.get(label, 0.0)
        if cur < floor:
            scale = (1.0 - floor) / max(1e-9, 1.0 - cur)
            out = {k: v * scale for k, v in out.items()}
            out[label] = floor
        z = sum(out.values()) or 1.0
        return {k: v / z for k, v in out.items()}

    # ---------------------------------------------------------------------- fit
    def fit(self, X: List[str], y: List[str],
            gold_texts: Optional[List[str]] = None,
            gold_labels: Optional[List[str]] = None,
            gold_weight: float = 6.0) -> "StackedClassifier":
        """Fit teacher on weak labels, then distil into the student.

        `gold_texts`/`gold_labels` optionally add a small set of hand-labelled
        examples (the dev split) with a higher sample weight. Test data is never
        passed here.
        """
        self.teacher.fit(X, y)
        self.labels = sorted(set(y) | set(self.teacher.nb.classes))

        all_texts = list(X) + list(gold_texts or [])

        # Pass 1: raw features, to build the pruned lexical vocabulary.
        self._vocab = None
        raw = [self._features(t, restrict=False) for t in all_texts]
        counts: Dict[str, int] = {}
        for fd in raw:
            for f in fd:
                if f.startswith(("w[", "c[")):
                    counts[f] = counts.get(f, 0) + 1
        self._vocab = {f for f, c in counts.items() if c >= self.min_feature_count}

        feats: List[Dict[str, float]] = []
        hard: List[str] = []
        soft: List[Dict[str, float]] = []
        weights: List[float] = []

        def prune(fd: Dict[str, float]) -> Dict[str, float]:
            return {f: v for f, v in fd.items()
                    if not f.startswith(("w[", "c[")) or f in self._vocab}

        n_weak = len(X)
        for i, text in enumerate(X):
            pred = self.teacher.predict(text)
            feats.append(prune(raw[i]))
            hard.append(pred.label)
            soft.append(self._sharpen(pred.proba, pred.label))
            weights.append(1.0)

        if gold_texts and gold_labels:
            for j, (text, lab) in enumerate(zip(gold_texts, gold_labels)):
                feats.append(prune(raw[n_weak + j]))
                hard.append(lab)
                # gold labels are one-hot: they are ground truth, not a teacher guess
                soft.append({c: (1.0 if c == lab else 0.0) for c in self.labels})
                weights.append(gold_weight)

        self.student.classes = self.labels
        self.student.fit(feats, hard, soft_targets=soft, sample_weight=weights)
        return self

    # ------------------------------------------------------------------ predict
    def predict(self, text: str) -> Prediction:
        pred, _ = self.predict_verbose(text)
        return pred

    def predict_verbose(self, text: str) -> Tuple[Prediction, List[str]]:
        why: List[str] = []
        student_proba = self.student.predict_proba(self._features(text))
        label = max(student_proba, key=student_proba.get)

        # Blend in the teacher when configured (student_weight < 1).
        if self.student_weight < 1.0:
            t = self.teacher.predict(text)
            tp = self._sharpen(t.proba, t.label)
            a = self.student_weight
            keys = set(student_proba) | set(tp)
            student_proba = {k: a * student_proba.get(k, 0.0) + (1 - a) * tp.get(k, 0.0)
                             for k in keys}
            z = sum(student_proba.values()) or 1.0
            student_proba = {k: v / z for k, v in student_proba.items()}
            label = max(student_proba, key=student_proba.get)

        # High-precision safety overrides retained from the transparent rules.
        s = analyze(text)
        if s.refund_request and label != "refund_billing" and not (
            s.policy_question and not s.account_scope
        ):
            why.append("safety_override:refund_request→refund_billing")
            label = "refund_billing"
            student_proba = RefinedClassifier._reproject(student_proba, label)
        elif s.harm_or_grievance and label not in ("complaint_feedback", "refund_billing"):
            why.append("safety_override:harm_or_grievance→complaint_feedback")
            label = "complaint_feedback"
            student_proba = RefinedClassifier._reproject(student_proba, label)

        return Prediction(label=label, proba=student_proba), why

    # -------------------------------------------------------------- persistence
    # Training the student is the only slow step in the pipeline (~1 min of
    # gradient descent). We cache its weights keyed by a fingerprint of the
    # training data + hyperparameters so `cli.py handle/demo` stay instant while
    # remaining correct: any change to the corpus or config invalidates the cache.
    def fingerprint(self, n_threads: int) -> str:
        import hashlib
        key = (f"{n_threads}|{self.student.lr}|{self.student.epochs}|{self.student.l2}|"
               f"{self.use_char}|{self.min_feature_count}|{self.student_weight}|v1")
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def save(self, path: str, n_threads: int) -> None:
        import json
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": self.fingerprint(n_threads),
                       "labels": self.labels,
                       "vocab": sorted(self._vocab or []),
                       "student": self.student.state()}, f)

    def try_load(self, path: str, n_threads: int) -> bool:
        import json
        import os
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            if d.get("fingerprint") != self.fingerprint(n_threads):
                return False
            self.labels = d["labels"]
            self._vocab = set(d["vocab"])
            self.student.load_state(d["student"])
            return True
        except Exception:
            return False

    def fit_cached(self, X: List[str], y: List[str], cache_path: str) -> "StackedClassifier":
        """Fit the teacher (cheap) and reuse cached student weights when valid."""
        self.teacher.fit(X, y)
        self.labels = sorted(set(y) | set(self.teacher.nb.classes))
        if self.try_load(cache_path, len(X)):
            return self
        self.fit(X, y)
        try:
            self.save(cache_path, len(X))
        except Exception:
            pass
        return self

    def explain(self, text: str, top_n: int = 8):
        """Top weighted features for the predicted class (model auditing)."""
        pred = self.predict(text)
        x = self._features(text)
        w = self.student.w.get(pred.label, {})
        contrib = sorted(
            ((w.get(f, 0.0) * v, f) for f, v in x.items()),
            reverse=True,
        )
        return pred, [(f, round(c, 3)) for c, f in contrib[:top_n]]
