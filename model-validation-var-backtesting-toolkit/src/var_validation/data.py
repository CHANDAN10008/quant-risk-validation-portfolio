"""Input validation and reproducible demonstration data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_returns(
    path: str | Path,
    *,
    date_column: str = "Date",
    value_column: str = "Return",
) -> pd.Series:
    """Load decimal returns or closing prices from CSV.

    If ``value_column`` is ``Close``, simple returns are computed from prices.
    The result is sorted by date and validated for finite observations.
    """
    frame = pd.read_csv(path)
    required = {date_column, value_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    valid = dates.notna() & values.notna()
    frame = pd.DataFrame({"date": dates[valid], "value": values[valid]}).sort_values("date")
    if frame.empty:
        raise ValueError("CSV has no valid date/value rows")
    if frame["date"].duplicated().any():
        raise ValueError("CSV contains duplicate dates")
    if value_column.lower() in {"close", "adj close", "adjusted close"}:
        if (frame["value"] <= 0).any():
            raise ValueError("Closing prices must be positive")
        returns = frame["value"].pct_change().dropna()
    else:
        returns = frame["value"]
    returns.index = frame.loc[returns.index, "date"] if isinstance(returns.index, pd.Index) else returns.index
    returns = pd.Series(returns.to_numpy(dtype=float), index=pd.DatetimeIndex(returns.index), name="return")
    if len(returns) < 2 or not np.isfinite(returns.to_numpy()).all():
        raise ValueError("At least two finite return observations are required")
    return returns


def synthetic_returns(
    *,
    observations: int = 1500,
    seed: int = 2026,
    mean: float = 0.0003,
    volatility: float = 0.012,
) -> pd.Series:
    """Create a labelled synthetic return series for examples only."""
    if observations <= 0 or volatility <= 0:
        raise ValueError("observations and volatility must be positive")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=observations)
    values = rng.normal(mean, volatility, size=observations)
    return pd.Series(values, index=dates, name="return")
