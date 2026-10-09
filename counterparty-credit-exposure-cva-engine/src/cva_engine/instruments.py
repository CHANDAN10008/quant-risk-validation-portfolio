"""Interest-rate swap representation and conditional mark-to-market valuation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .hull_white import HullWhiteModel, SimulationResult


@dataclass(frozen=True)
class InterestRateSwap:
    """Plain-vanilla single-curve fixed-for-floating swap with regular coupons.

    direction=+1 means receive fixed and pay floating; direction=-1 means
    pay fixed and receive floating. Both legs use the same regular frequency.
    """

    name: str
    notional: float
    fixed_rate: float
    maturity_years: float
    payment_frequency: int = 4
    direction: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("swap name cannot be empty")
        if self.notional <= 0.0:
            raise ValueError("notional must be positive")
        if self.maturity_years <= 0.0:
            raise ValueError("maturity_years must be positive")
        if self.payment_frequency < 1:
            raise ValueError("payment_frequency must be at least one")
        if not np.isclose(self.maturity_years * self.payment_frequency,
                          round(self.maturity_years * self.payment_frequency)):
            raise ValueError("maturity must contain an integer number of regular coupons")
        if self.direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")

    @property
    def payment_times(self) -> NDArray[np.float64]:
        n = int(round(self.maturity_years * self.payment_frequency))
        return np.arange(1, n + 1, dtype=float) / self.payment_frequency

    @property
    def accrual(self) -> float:
        return 1.0 / self.payment_frequency

    def reset_fixings(self, model: HullWhiteModel, simulation: SimulationResult) -> NDArray[np.float64]:
        """Pathwise simple rates fixed at each reset date for the next coupon."""
        times = self.payment_times
        fixings = np.empty((simulation.short_rates.shape[0], len(times)), dtype=float)
        for j, end in enumerate(times):
            start = 0.0 if j == 0 else float(times[j - 1])
            idx = simulation.index_at(start)
            p_end = model.bond_price(start, float(end), simulation.short_rates[:, idx])
            fixings[:, j] = (1.0 / p_end - 1.0) / self.accrual
        return fixings

    def value_at(
        self,
        model: HullWhiteModel,
        time: float,
        short_rate: NDArray[np.float64],
        fixings: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Value the remaining swap cash flows at time t for every path.

        A coupon whose reset date is already past uses its simulated fixing.
        Coupons resetting at or after t are projected from conditional bond
        prices. This supports exposure dates between coupon dates.
        """
        r = np.asarray(short_rate, dtype=float)
        if time < 0.0 or time > self.maturity_years + 1e-10:
            raise ValueError("valuation time must lie within the swap term")
        if fixings.shape != (r.shape[0], len(self.payment_times)):
            raise ValueError("fixings shape does not match paths and coupon schedule")
        fixed_leg = np.zeros_like(r)
        floating_leg = np.zeros_like(r)
        times = self.payment_times
        tol = 1e-10
        for j, end in enumerate(times):
            if end <= time + tol:
                continue
            start = 0.0 if j == 0 else float(times[j - 1])
            p_end = model.bond_price(time, float(end), r)
            fixed_leg += self.notional * self.fixed_rate * self.accrual * p_end
            if start < time - tol:
                rate = fixings[:, j]
            else:
                p_start = model.bond_price(time, float(start), r)
                rate = (p_start / p_end - 1.0) / self.accrual
            floating_leg += self.notional * self.accrual * rate * p_end
        return self.direction * (fixed_leg - floating_leg)

    def analytic_value(self, model: HullWhiteModel) -> float:
        """Analytic time-zero value using the initial discount curve."""
        times = self.payment_times
        fixed_leg = self.notional * self.fixed_rate * self.accrual * np.sum(model.curve.discount(times))
        floating_leg = self.notional * (1.0 - model.curve.discount(self.maturity_years))
        return float(self.direction * (fixed_leg - floating_leg))
