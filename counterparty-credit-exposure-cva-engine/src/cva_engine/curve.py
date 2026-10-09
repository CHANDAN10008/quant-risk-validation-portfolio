"""Discount-curve utilities with log-linear discount-factor interpolation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class DiscountCurve:
    """A continuously compounded zero curve represented by discount factors.

    Discount factors are interpolated linearly in log space. Beyond the last
    knot, the final segment's instantaneous forward rate is held constant.
    """

    times: NDArray[np.float64]
    discount_factors: NDArray[np.float64]

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        dfs = np.asarray(self.discount_factors, dtype=float)
        if times.ndim != 1 or dfs.ndim != 1 or len(times) != len(dfs):
            raise ValueError("times and discount_factors must be equal-length vectors")
        if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(dfs)):
            raise ValueError("the curve needs at least two finite knots")
        if times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
            raise ValueError("curve times must start at zero and increase strictly")
        if not np.isclose(dfs[0], 1.0) or np.any(dfs <= 0.0):
            raise ValueError("discount factors must be positive and start at one")
        times.setflags(write=False)
        dfs.setflags(write=False)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "discount_factors", dfs)

    @classmethod
    def from_zero_rates(
        cls, times: ArrayLike, continuously_compounded_rates: ArrayLike
    ) -> "DiscountCurve":
        """Build from continuously compounded zero rates at curve knot times."""
        t = np.asarray(times, dtype=float)
        r = np.asarray(continuously_compounded_rates, dtype=float)
        if t.ndim != 1 or r.shape != t.shape or len(t) < 2:
            raise ValueError("times and rates must be equal-length vectors")
        dfs = np.exp(-t * r)
        dfs[0] = 1.0
        return cls(t, dfs)

    def discount(self, time: ArrayLike) -> NDArray[np.float64] | float:
        """Return P(0,t), extrapolating with the last segment's forward rate."""
        t = np.asarray(time, dtype=float)
        if np.any(t < 0.0):
            raise ValueError("time cannot be negative")
        log_dfs = np.log(self.discount_factors)
        flat = t.reshape(-1)
        result = np.interp(flat, self.times, log_dfs)
        beyond = flat > self.times[-1]
        if np.any(beyond):
            slope = (log_dfs[-1] - log_dfs[-2]) / (self.times[-1] - self.times[-2])
            result[beyond] = log_dfs[-1] + slope * (flat[beyond] - self.times[-1])
        result = np.exp(result).reshape(t.shape)
        if t.ndim == 0:
            return float(result)
        return result

    def instantaneous_forward(self, time: ArrayLike) -> NDArray[np.float64] | float:
        """Return the piecewise-constant forward rate implied by log-linear DFs."""
        t = np.asarray(time, dtype=float)
        if np.any(t < 0.0):
            raise ValueError("time cannot be negative")
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 2)
        slopes = np.diff(np.log(self.discount_factors)) / np.diff(self.times)
        result = -slopes[idx]
        if t.ndim == 0:
            return float(result)
        return np.asarray(result, dtype=float)

    def integrated_forward(self, start: float, end: float) -> float:
        """Integral of the instantaneous forward between two nonnegative dates."""
        if start < 0.0 or end < start:
            raise ValueError("require 0 <= start <= end")
        return float(np.log(self.discount(start) / self.discount(end)))
