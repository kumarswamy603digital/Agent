"""Minimal TF-IDF vectorizer + cosine similarity (pure stdlib).

Sparse vectors are plain dicts {feature: weight}. This is intentionally simple
and fast enough for tens of thousands of short tweets.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List

from .text import featurize


class TfidfVectorizer:
    def __init__(self, ngram_range=(1, 2), min_df: int = 2):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()

    def fit(self, docs: Iterable[str]) -> "TfidfVectorizer":
        df: Counter = Counter()
        n = 0
        for d in docs:
            n += 1
            feats = set(featurize(d, self.ngram_range))
            for f in feats:
                df[f] += 1
        self.vocab = {f for f, c in df.items() if c >= self.min_df}
        # smoothed idf
        self.idf = {
            f: math.log((1 + n) / (1 + df[f])) + 1.0 for f in self.vocab
        }
        return self

    def transform(self, doc: str) -> Dict[str, float]:
        counts = Counter(f for f in featurize(doc, self.ngram_range) if f in self.vocab)
        if not counts:
            return {}
        vec = {f: (c) * self.idf[f] for f, c in counts.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {f: w / norm for f, w in vec.items()}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    # iterate over the smaller dict
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b.get(f, 0.0) for f, w in a.items())
