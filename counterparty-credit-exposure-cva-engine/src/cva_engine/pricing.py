"""Analytic and Monte Carlo swap valuation benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .hull_white import HullWhiteModel, SimulationResult
from .instruments import InterestRateSwap


@dataclass(frozen=True)
class PricingBenchmark:
    analytic_value: float
    monte_carlo_value: float
    standard_error: float
    confidence_interval_low: float
    confidence_interval_high: float
    absolute_error: float


def benchmark_swap_value(
    swap: InterestRateSwap,
    model: HullWhiteModel,
    simulation: SimulationResult,
    confidence_level: float = 0.95,
) -> PricingBenchmark:
    """Compare the curve-valued swap with discounted simulated cash flows."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    fixings = swap.reset_fixings(model, simulation)
    fixed_pv = np.zeros(simulation.short_rates.shape[0], dtype=float)
    float_pv = np.zeros_like(fixed_pv)
    for j, payment_time in enumerate(swap.payment_times):
        idx = simulation.index_at(float(payment_time))
        discount = simulation.discount_factors[:, idx]
        fixed_pv += swap.notional * swap.fixed_rate * swap.accrual * discount
        float_pv += swap.notional * swap.accrual * fixings[:, j] * discount
    path_values = swap.direction * (fixed_pv - float_pv)
    estimate = float(path_values.mean())
    standard_error = float(path_values.std(ddof=1) / np.sqrt(len(path_values)))
    quantile = float(norm.ppf(0.5 + confidence_level / 2.0))
    analytic = swap.analytic_value(model)
    return PricingBenchmark(
        analytic_value=analytic,
        monte_carlo_value=estimate,
        standard_error=standard_error,
        confidence_interval_low=estimate - quantile * standard_error,
        confidence_interval_high=estimate + quantile * standard_error,
        absolute_error=abs(estimate - analytic),
    )
