"""Likelihood-ratio backtests and a simple traffic-light classification."""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from scipy.special import xlogy


def kupiec_pof(exceptions: np.ndarray, tail_probability: float) -> dict[str, float | int]:
    """Kupiec proportion-of-failures likelihood-ratio test."""
    hit = np.asarray(exceptions, dtype=bool)
    n = int(hit.size)
    x = int(hit.sum())
    if n == 0 or not 0.0 < tail_probability < 1.0:
        raise ValueError("A non-empty exception series and valid tail probability are required")
    observed = x / n
    log_null = float(xlogy(n - x, 1.0 - tail_probability) + xlogy(x, tail_probability))
    log_alt = float(xlogy(n - x, 1.0 - observed) + xlogy(x, observed))
    statistic = max(0.0, -2.0 * (log_null - log_alt))
    return {"observations": n, "exceptions": x, "exception_rate": observed, "lr_stat": statistic, "p_value": float(chi2.sf(statistic, 1))}


def christoffersen_independence(exceptions: np.ndarray) -> dict[str, float | int]:
    """Christoffersen first-order exception-independence likelihood-ratio test."""
    hit = np.asarray(exceptions, dtype=bool)
    if hit.ndim != 1 or hit.size < 2:
        raise ValueError("At least two one-dimensional exception observations are required")
    previous, current = hit[:-1], hit[1:]
    n00 = int((~previous & ~current).sum())
    n01 = int((~previous & current).sum())
    n10 = int((previous & ~current).sum())
    n11 = int((previous & current).sum())
    n0, n1 = n00 + n01, n10 + n11
    total = n0 + n1
    p01 = n01 / n0 if n0 else 0.0
    p11 = n11 / n1 if n1 else 0.0
    pooled = (n01 + n11) / total if total else 0.0
    log_independent = float(xlogy(n01 + n11, pooled) + xlogy(n00 + n10, 1.0 - pooled))
    log_markov = float(xlogy(n01, p01) + xlogy(n00, 1.0 - p01) + xlogy(n11, p11) + xlogy(n10, 1.0 - p11))
    statistic = max(0.0, 2.0 * (log_markov - log_independent))
    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "lr_stat": statistic,
        "p_value": float(chi2.sf(statistic, 1)),
    }


def traffic_light(exceptions: np.ndarray, *, confidence: float) -> str:
    """Classify exactly 250 forecasts at 99% confidence using simple bands."""
    hit = np.asarray(exceptions, dtype=bool)
    if confidence != 0.99 or hit.size != 250:
        return "not_applicable"
    count = int(hit.sum())
    if count <= 4:
        return "green"
    if count <= 9:
        return "yellow"
    return "red"
