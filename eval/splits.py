"""Deterministic dev / test split of the golden set.

WHY THIS EXISTS
---------------
Model choices (rule patterns, blend weights, thresholds) are tuned by hand. If we
tuned against the whole golden set and then reported accuracy on that same set,
the number would be optimistically biased and not a generalization estimate.

So the golden set is split once, deterministically:

  * **dev**  (60%) — the ONLY split used while iterating on the model.
  * **test** (40%) — held out; evaluated at the end to produce the headline number.

The split is stratified by intent so every class appears in both halves, and it is
seeded, so it is identical for every reviewer on every run.

Additionally we expose the `round2` subset: 50 examples that were authored to grow
the set to 200 *before* any tuning began, to broaden coverage rather than to target
known errors. Accuracy on the `round2 ∩ test` slice is the most conservative
generalization estimate we report, because those items were never inspected during
error analysis.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

GOLDEN_PATH = "data/golden/golden_eval.jsonl"

# Examples g001..g150 were the original set (their errors were inspected during
# development). g151..g200 were added later, before tuning, to broaden coverage.
ROUND1_MAX = 150

DEV_FRACTION = 0.6
SPLIT_SEED = 20240517


def load_golden(path: str = GOLDEN_PATH) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _round_of(rec: dict) -> int:
    """1 for the original examples, 2 for the later coverage-broadening batch."""
    n = int(rec["id"].lstrip("g"))
    return 1 if n <= ROUND1_MAX else 2


def split_golden(records: List[dict] = None,
                 dev_fraction: float = DEV_FRACTION,
                 seed: int = SPLIT_SEED) -> Tuple[List[dict], List[dict]]:
    """Stratified, seeded dev/test split. Returns (dev, test)."""
    if records is None:
        records = load_golden()

    by_intent: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_intent[r["intent"]].append(r)

    rng = random.Random(seed)
    dev: List[dict] = []
    test: List[dict] = []
    for intent in sorted(by_intent):
        group = sorted(by_intent[intent], key=lambda r: r["id"])
        rng.shuffle(group)
        k = max(1, round(len(group) * dev_fraction))
        # guarantee at least one test example per intent
        if k >= len(group):
            k = len(group) - 1
        dev.extend(group[:k])
        test.extend(group[k:])

    dev.sort(key=lambda r: r["id"])
    test.sort(key=lambda r: r["id"])
    return dev, test


def round2_only(records: List[dict]) -> List[dict]:
    """The coverage-broadening batch (never inspected during error analysis)."""
    return [r for r in records if _round_of(r) == 2]


if __name__ == "__main__":
    from collections import Counter

    recs = load_golden()
    dev, test = split_golden(recs)
    print(f"golden total: {len(recs)}  dev: {len(dev)}  test: {len(test)}")
    print(f"round1: {sum(1 for r in recs if _round_of(r)==1)}  "
          f"round2: {sum(1 for r in recs if _round_of(r)==2)}")
    print(f"round2 in test: {len(round2_only(test))}")
    print("\ndev intent counts :", dict(sorted(Counter(r['intent'] for r in dev).items())))
    print("test intent counts:", dict(sorted(Counter(r['intent'] for r in test).items())))
