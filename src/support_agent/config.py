"""Central configuration. Plain dataclass so it works without PyYAML/pydantic."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class Config:
    # --- brand / data ---
    brand: str = "Delta"
    # Path to the tweets CSV. Defaults to the bundled synthetic sample, but point
    # this at the real Kaggle `twcs.csv` and everything else works unchanged.
    data_path: str = os.environ.get(
        "TWCS_PATH", "data/raw/twcs_sample_delta.csv"
    )

    # --- text / retrieval vectorizer ---
    ngram_range: Tuple[int, int] = (1, 2)   # used by TF-IDF retriever
    min_df: int = 2

    # --- rule/NB teacher inside the main model ---
    rule_weight: float = 0.7       # weight on rules when they fire (else defer to NB)
    nb_alpha: float = 0.3          # Laplace smoothing for Naive Bayes
    nb_min_df: int = 1
    nb_ngram_range: Tuple[int, int] = (1, 1)  # unigrams generalize best here

    # --- learned student (distilled logistic regression) ---
    lr_learning_rate: float = 0.6
    lr_epochs: int = 150
    lr_l2: float = 1e-3
    lr_use_char_ngrams: bool = True
    lr_min_feature_count: int = 3
    # Ensemble weight on the STUDENT. 0.5 = equal-weight blend of the hand-written
    # rule chain and the learned model. Deliberately left at the parameter-free
    # default rather than tuned, so the headline test number isn't fitted to test.
    student_weight: float = 0.5

    # Cache for the trained student's weights (keeps the CLI instant).
    model_cache_path: str = ".cache/student_model.json"

    # --- abstain / escalation ---
    # If top intent probability < abstain_threshold, classifier returns ABSTAIN
    # and the message is force-escalated.
    abstain_threshold: float = 0.42
    # Margin between top-1 and top-2 probability below which we also abstain.
    min_margin: float = 0.10

    # --- retrieval / reply ---
    retrieval_k: int = 3           # exemplars pulled for grounding a reply
    min_retrieval_sim: float = 0.08

    # --- llm backend ---
    # "heuristic" (deterministic, no API key, default) | "openai" | "anthropic"
    llm_backend: str = os.environ.get("SUPPORT_AGENT_LLM", "heuristic")
    llm_model: str = os.environ.get("SUPPORT_AGENT_LLM_MODEL", "")

    # --- reproducibility ---
    seed: int = 13

    # intents kept here for convenience
    labels: Tuple[str, ...] = field(default_factory=tuple)


DEFAULT = Config()
