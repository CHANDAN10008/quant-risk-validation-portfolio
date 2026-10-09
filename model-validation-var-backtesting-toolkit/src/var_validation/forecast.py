"""Rolling one-step-ahead VaR estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def rolling_var_forecasts(
    returns: pd.Series,
    *,
    lookback: int = 250,
    confidence: float = 0.99,
) -> pd.DataFrame:
    """Return historical and normal VaR forecasts without look-ahead bias."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("returns must be a one-dimensional finite series")
    if lookback < 20 or lookback >= len(values):
        raise ValueError("lookback must be at least 20 and shorter than the series")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1.0")

    dates = returns.index
    rows: list[dict[str, object]] = []
    tail_probability = 1.0 - confidence
    for i in range(lookback, len(values)):
        window = values[i - lookback : i]
        sample_mean = float(np.mean(window))
        sample_std = float(np.std(window, ddof=1))
        hist_var = -float(np.quantile(window, tail_probability, method="linear"))
        normal_var = -(sample_mean + sample_std * float(norm.ppf(tail_probability)))
        realized = float(values[i])
        rows.append(
            {
                "date": dates[i],
                "return": realized,
                "historical_var": hist_var,
                "parametric_var": normal_var,
                "historical_exception": realized < -hist_var,
                "parametric_exception": realized < -normal_var,
            }
        )
    return pd.DataFrame(rows).set_index("date")
