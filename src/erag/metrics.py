from __future__ import annotations

import math
import random
from typing import Sequence


def hit_at_k(rels: Sequence[bool], k: int) -> float:
    return float(any(rels[:k]))


def mrr(rels: Sequence[bool]) -> float:
    for i, r in enumerate(rels):
        if r:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(rels: Sequence[bool], k: int, n_relevant: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(rels[:k]) if r)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(max(n_relevant, 1), k)))
    return dcg / idcg


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(values)
    if m == 0:
        return (0.0, 0.0)
    means = sorted(sum(values[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def value_in(answer: str, value: str) -> bool:
    norm = lambda s: s.lower().replace(",", "")
    return norm(value) in norm(answer)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"
