"""Nominal agreement formulas retained from the original experiment scripts."""

import math

from collections import Counter

def cohen_kappa(a: list[str], b: list[str]) -> tuple[float, float, int]:
    pairs = [(x, y) for x, y in zip(a, b) if x != "" and y != ""]
    n = len(pairs)
    if n == 0:
        return math.nan, math.nan, 0
    observed = sum(1 for x, y in pairs if x == y) / n
    ca = Counter(x for x, _ in pairs)
    cb = Counter(y for _, y in pairs)
    labels = set(ca) | set(cb)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if expected == 1:
        kappa = 1.0 if observed == 1 else math.nan
    else:
        kappa = (observed - expected) / (1 - expected)
    return kappa, observed, n

def krippendorff_alpha_nominal(units: list[list[str]]) -> tuple[float, int]:
    usable = [[x for x in unit if x != ""] for unit in units]
    usable = [unit for unit in usable if len(unit) >= 2]
    total_ratings = sum(len(unit) for unit in usable)
    if total_ratings <= 1:
        return math.nan, len(usable)

    observed_num = 0.0
    marginals: Counter[str] = Counter()
    for unit in usable:
        n = len(unit)
        counts = Counter(unit)
        marginals.update(unit)
        observed_num += sum(count * (n - count) for count in counts.values()) / (n - 1)

    expected_num = sum(count * (total_ratings - count) for count in marginals.values()) / (
        total_ratings - 1
    )
    if expected_num == 0:
        alpha = 1.0 if observed_num == 0 else math.nan
    else:
        alpha = 1 - observed_num / expected_num
    return alpha, len(usable)
