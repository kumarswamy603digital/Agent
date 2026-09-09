"""Feature extraction for the learned intent model.

Three feature families, deliberately chosen so the model can generalize beyond the
training corpus's vocabulary:

1. **Lexical** — word unigrams/bigrams and character 4-grams. Character n-grams are
   what make this robust to the typos, elongations, and abbreviations in real tweets
   ("cancelld", "flt", "loooong"): an unseen word still shares character substrings
   with words the model has seen.

2. **Signal** — the interpretable stance/framing detectors from `signals.py`
   (complaint tone, staff conduct, policy question, account scope, refund request vs.
   money mention, sarcasm, ...). These are hand-written but *general*: they were
   written to describe language patterns, not to match specific dev examples, which
   is why they transfer to held-out data.

3. **Model-output** — the keyword-rule score per intent and the weakly-supervised
   Naive Bayes posterior per intent. Including these makes the learned model a
   *stacker*: it learns how much to trust each upstream component per intent instead
   of us fixing that trade-off by hand (the old `rule_weight=0.7` constant).

Feature values are kept on a comparable scale (roughly 0–1) so a single learning rate
and L2 strength behave sensibly across families.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

from .signals import analyze
from .text import featurize, normalize

CHAR_NGRAM_N = 4
_NONWORD = re.compile(r"\s+")


def char_ngrams(text: str, n: int = CHAR_NGRAM_N) -> Dict[str, float]:
    """Character n-grams over the normalized string, with word boundaries marked."""
    norm = _NONWORD.sub(" ", normalize(text)).strip()
    if not norm:
        return {}
    padded = f" {norm} "
    out: Dict[str, float] = {}
    for i in range(len(padded) - n + 1):
        g = padded[i : i + n]
        if g.strip():
            out[f"c[{g}]"] = out.get(f"c[{g}]", 0.0) + 1.0
    # sublinear scaling keeps long tweets from dominating
    return {k: 1.0 + (v - 1.0) * 0.25 for k, v in out.items()}


def word_features(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for f in featurize(text, (1, 2)):
        key = f"w[{f}]"
        out[key] = out.get(key, 0.0) + 1.0
    return {k: 1.0 + (v - 1.0) * 0.25 for k, v in out.items()}


def signal_features(text: str) -> Dict[str, float]:
    s = analyze(text)
    feats = {
        "s[complaint_tone]": float(s.complaint_tone),
        "s[staff_conduct]": float(s.staff_conduct),
        "s[service_failure]": float(s.service_failure),
        "s[harm_or_grievance]": float(s.harm_or_grievance),
        "s[complaint_evidence]": min(s.complaint_evidence, 3) / 3.0,
        "s[policy_question]": float(s.policy_question),
        "s[account_scope]": float(s.account_scope),
        "s[howto]": float(s.howto),
        "s[refund_request]": float(s.refund_request),
        "s[money_mention]": float(s.money_mention),
        "s[booking_action]": float(s.booking_action),
        "s[offtopic]": float(s.offtopic),
        "s[contentless]": float(s.contentless),
        "s[positive]": float(s.positive),
        "s[strong_positive]": float(s.strong_positive),
        "s[sarcasm]": float(s.sarcasm),
        "s[question]": float(s.question),
        # useful interactions the linear model cannot form on its own
        "s[policyQ_x_no_account]": float(s.policy_question and not s.account_scope),
        "s[account_x_no_policyQ]": float(s.account_scope and not s.policy_question),
        "s[money_x_booking]": float(s.money_mention and s.booking_action
                                    and not s.refund_request),
        "s[complaint_x_no_account]": float(s.complaint_evidence >= 1
                                           and not s.account_scope),
    }
    return feats


def build_features(text: str, rule_scores: Optional[Dict[str, float]] = None,
                   nb_proba: Optional[Dict[str, float]] = None,
                   use_char: bool = True) -> Dict[str, float]:
    """Assemble the full feature dict for one message."""
    feats: Dict[str, float] = {}
    feats.update(word_features(text))
    if use_char:
        feats.update(char_ngrams(text))
    feats.update(signal_features(text))
    if rule_scores:
        total = sum(rule_scores.values()) or 1.0
        for intent, sc in rule_scores.items():
            if sc:
                feats[f"r[{intent}]"] = sc / total
                feats[f"rabs[{intent}]"] = min(sc, 9.0) / 9.0
    if nb_proba:
        for intent, p in nb_proba.items():
            if p > 1e-4:
                feats[f"nb[{intent}]"] = p
    feats["bias_on"] = 1.0
    return feats
