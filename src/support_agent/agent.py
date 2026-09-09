"""End-to-end support agent: classify -> draft grounded reply -> decide routing.

Training uses *weak supervision*: we do not have gold intent labels on the
historical corpus (the real twcs.csv has none), so we label the training threads
with the high-precision RuleClassifier and train Naive Bayes on those weak
labels. The model then generalizes beyond the seed keywords (it learns
co-occurring vocabulary), which is exactly why NB beats the rules on the
hand-labelled golden set. This mirrors how you'd bootstrap on real data.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .classifiers.base import Prediction
from .classifiers.refined import RefinedClassifier
from .classifiers.rules import RuleClassifier
from .classifiers.rules_refined import RULES_V2
from .config import Config, DEFAULT
from .data_loader import build_threads
from .escalation import Decision, decide
from .intents import ABSTAIN
from .llm.backend import get_backend
from .reply import DraftedReply, ReplyDrafter
from .retriever import ResolutionRetriever


@dataclass
class AgentResponse:
    text_in: str
    intent: str
    confidence: float
    proba: dict
    escalate: bool
    reason: str
    signals: List[str]
    reply: str
    reply_grounded: bool
    evidence: List[dict] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


class SupportAgent:
    def __init__(self, config: Config = DEFAULT):
        self.cfg = config
        self.classifier: Optional[RefinedClassifier] = None
        self.retriever: Optional[ResolutionRetriever] = None
        self.drafter: Optional[ReplyDrafter] = None
        # Weak-supervision teacher uses the corrected keyword table.
        self._weak_labeler = RuleClassifier(rules_table=RULES_V2)

    # ------------------------------------------------------------------ build
    def build(self, threads=None):
        cfg = self.cfg
        if threads is None:
            threads = build_threads(cfg.data_path, cfg.brand)
        if not threads:
            raise RuntimeError(
                f"No threads for brand '{cfg.brand}' in {cfg.data_path}."
            )
        customer_texts = [th.customer_text for th in threads]
        resolutions = [th.resolution_text for th in threads]

        # weak supervision
        self._weak_labeler.fit(customer_texts, ["general_info"] * len(customer_texts))
        weak_labels = [self._weak_labeler.predict(t).label for t in customer_texts]

        self.classifier = RefinedClassifier(
            rule_weight=cfg.rule_weight, nb_alpha=cfg.nb_alpha,
            nb_min_df=cfg.nb_min_df, nb_ngram_range=cfg.nb_ngram_range,
        ).fit(customer_texts, weak_labels)

        self.retriever = ResolutionRetriever(ngram_range=cfg.ngram_range, min_df=1).fit(
            customer_texts, resolutions, weak_labels
        )
        self.drafter = ReplyDrafter(
            self.retriever,
            backend=get_backend(cfg.llm_backend, cfg.llm_model),
            retrieval_k=cfg.retrieval_k,
            min_retrieval_sim=cfg.min_retrieval_sim,
        )
        return self

    # --------------------------------------------------------------- classify
    def classify(self, text: str) -> Prediction:
        pred = self.classifier.predict(text)
        # abstain gate
        if pred.confidence < self.cfg.abstain_threshold or pred.margin < self.cfg.min_margin:
            gated = Prediction(label=ABSTAIN, proba=pred.proba)
            return gated
        return pred

    # -------------------------------------------------------------- full turn
    def handle(self, text: str) -> AgentResponse:
        raw_pred = self.classifier.predict(text)
        gated = self.classify(text)
        # For routing/decision we use the *gated* label (may be 'uncertain');
        # the underlying proba still reflects the model's belief.
        decision: Decision = decide(
            text, gated, self.cfg.abstain_threshold, self.cfg.min_margin
        )
        draft: DraftedReply = self.drafter.draft(text, gated.label)
        return AgentResponse(
            text_in=text,
            intent=gated.label,
            confidence=round(raw_pred.confidence, 4),
            proba={k: round(v, 4) for k, v in sorted(raw_pred.proba.items(), key=lambda kv: -kv[1])},
            escalate=decision.escalate,
            reason=decision.reason,
            signals=decision.signals,
            reply=draft.text,
            reply_grounded=draft.grounded,
            evidence=[
                {
                    "similarity": round(e.similarity, 3),
                    "similar_ask": e.customer_text,
                    "past_resolution": e.resolution_text,
                    "intent": e.intent,
                }
                for e in draft.evidence
            ],
        )
