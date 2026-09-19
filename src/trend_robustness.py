"""Historical tie-corrected Mann–Kendall statistic, retained for reproducibility."""
import math
import numpy as np
from scipy import stats
from collections import Counter

def mann_kendall(y: np.ndarray) -> tuple[float, float, float, float]:
    """Tie-corrected Mann--Kendall test with continuity correction."""
    n = len(y)
    score = 0.0
    for index in range(n - 1):
        score += float(np.sum(np.sign(y[index + 1 :] - y[index])))

    _, counts = np.unique(y, return_counts=True)
    tie_term = float(np.sum(counts * (counts - 1) * (2 * counts + 5)))
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
    if variance <= 0:
        return score, 0.0, 0.0, 1.0

    if score > 0:
        z_value = (score - 1.0) / math.sqrt(variance)
    elif score < 0:
        z_value = (score + 1.0) / math.sqrt(variance)
    else:
        z_value = 0.0
    p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    tau = score / (0.5 * n * (n - 1))
    return score, float(tau), float(z_value), p_value


