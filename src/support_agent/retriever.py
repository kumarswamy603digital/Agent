"""Retrieve historically-resolved threads similar to an incoming message.

This is the "grounding" store. For each resolved thread we index the customer
ask; at query time we return the most similar asks *and the brand's actual
replies*, optionally filtered to the predicted intent. The reply drafter then
grounds its answer in these real resolutions rather than free-associating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .vectorizer import TfidfVectorizer, cosine


@dataclass
class Exemplar:
    customer_text: str
    resolution_text: str
    intent: str
    similarity: float = 0.0


class ResolutionRetriever:
    def __init__(self, ngram_range=(1, 2), min_df: int = 1):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df)
        self.exemplars: List[Exemplar] = []
        self._vectors: List[Dict[str, float]] = []

    def fit(self, customer_texts: List[str], resolutions: List[str], intents: List[str]) -> "ResolutionRetriever":
        # Only index threads that actually have a brand resolution.
        rows = [
            (c, r, i)
            for c, r, i in zip(customer_texts, resolutions, intents)
            if r and r.strip()
        ]
        self.exemplars = [Exemplar(customer_text=c, resolution_text=r, intent=i) for c, r, i in rows]
        self.vectorizer.fit([e.customer_text for e in self.exemplars])
        self._vectors = [self.vectorizer.transform(e.customer_text) for e in self.exemplars]
        return self

    def query(self, text: str, k: int = 3, intent: Optional[str] = None,
              min_sim: float = 0.0) -> List[Exemplar]:
        qv = self.vectorizer.transform(text)
        scored: List[Exemplar] = []
        for ex, v in zip(self.exemplars, self._vectors):
            if intent is not None and ex.intent != intent:
                continue
            sim = cosine(qv, v)
            if sim < min_sim:
                continue
            scored.append(Exemplar(ex.customer_text, ex.resolution_text, ex.intent, sim))
        scored.sort(key=lambda e: e.similarity, reverse=True)
        # If intent filtering starved results, back off to unfiltered top-k.
        if intent is not None and len(scored) < k:
            extra = self.query(text, k=k, intent=None, min_sim=min_sim)
            seen = {(e.customer_text, e.resolution_text) for e in scored}
            for e in extra:
                if (e.customer_text, e.resolution_text) not in seen:
                    scored.append(e)
                if len(scored) >= k:
                    break
        return scored[:k]
