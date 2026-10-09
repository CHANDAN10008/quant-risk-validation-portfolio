"""One-factor Hull-White short-rate model and exact Gaussian path simulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .curve import DiscountCurve


@dataclass(frozen=True)
class HullWhiteParameters:
    mean_reversion: float = 0.10
    volatility: float = 0.01

    def __post_init__(self) -> None:
        if self.mean_reversion <= 0.0:
            raise ValueError("mean_reversion must be positive")
        if self.volatility < 0.0:
            raise ValueError("volatility cannot be negative")


@dataclass(frozen=True)
class SimulationResult:
    times: NDArray[np.float64]
    short_rates: NDArray[np.float64]
    discount_factors: NDArray[np.float64]

    def index_at(self, time: float) -> int:
        idx = int(np.argmin(np.abs(self.times - time)))
        if not np.isclose(self.times[idx], time, atol=1e-10, rtol=0.0):
            raise ValueError(f"time {time} is not on the simulation grid")
        return idx


class HullWhiteModel:
    r"""Hull-White model dr=(theta(t)-a r)dt+sigma dW calibrated to P(0,t).

    The shifted representation r(t)=x(t)+phi(t) is used, where x is an OU
    process and phi(t)=f(0,t)+sigma^2(1-exp(-a t))^2/(2a^2).
    """

    def __init__(self, curve: DiscountCurve, parameters: HullWhiteParameters):
        self.curve = curve
        self.parameters = parameters

    @property
    def a(self) -> float:
        return self.parameters.mean_reversion

    @property
    def sigma(self) -> float:
        return self.parameters.volatility

    def phi(self, time: ArrayLike) -> NDArray[np.float64] | float:
        t = np.asarray(time, dtype=float)
        correction = (self.sigma**2 / (2.0 * self.a**2)) * (-np.expm1(-self.a * t)) ** 2
        result = np.asarray(self.curve.instantaneous_forward(t)) + correction
        if t.ndim == 0:
            return float(result)
        return result

    def _phi_integral(self, start: float, end: float) -> float:
        a = self.a
        # The curve-forward contribution integrates exactly under the curve.
        curve_part = self.curve.integrated_forward(start, end)
        scale = self.sigma**2 / (2.0 * a**2)

        def primitive(t: float) -> float:
            return t + 2.0 * np.exp(-a * t) / a - np.exp(-2.0 * a * t) / (2.0 * a)

        return float(curve_part + scale * (primitive(end) - primitive(start)))

    def bond_price(self, time: float, maturity: float, short_rate: ArrayLike) -> NDArray[np.float64] | float:
        """Conditional zero-coupon price P(t,T) from the affine Hull-White formula."""
        if time < 0.0 or maturity < time:
            raise ValueError("require 0 <= time <= maturity")
        r = np.asarray(short_rate, dtype=float)
        if np.isclose(maturity, time):
            result = np.ones_like(r)
        else:
            a = self.a
            b = -np.expm1(-a * (maturity - time)) / a
            variance_adjustment = (
                self.sigma**2
                * (1.0 - np.exp(-2.0 * a * time))
                * b**2
                / (4.0 * a)
            )
            log_a = (
                np.log(self.curve.discount(maturity) / self.curve.discount(time))
                + b * self.curve.instantaneous_forward(time)
                - variance_adjustment
            )
            result = np.exp(log_a - b * r)
        if r.ndim == 0:
            return float(result)
        return result

    @staticmethod
    def _integrated_ou_variance(a: float, sigma: float, dt: float) -> float:
        x = a * dt
        if abs(x) < 1e-4:
            dimensionless = x**3 / 3.0 - x**4 / 4.0 + 7.0 * x**5 / 60.0 - x**6 / 24.0
            return float(sigma**2 * dimensionless / a**3)
        bracket = dt - 2.0 * (-np.expm1(-x)) / a + (-np.expm1(-2.0 * x)) / (2.0 * a)
        return float(sigma**2 * bracket / a**2)

    def simulate(
        self,
        times: ArrayLike,
        n_paths: int,
        seed: int = 2026,
        antithetic: bool = True,
    ) -> SimulationResult:
        """Simulate exact OU transitions and their correlated rate integrals.

        Exact interval integrals provide pathwise risk-free discount factors
        without Euler discretization. With antithetic sampling, n_paths must be
        even so every Gaussian path has its paired negative.
        """
        grid = np.asarray(times, dtype=float)
        if grid.ndim != 1 or len(grid) < 2 or grid[0] != 0.0 or np.any(np.diff(grid) <= 0.0):
            raise ValueError("times must be a strictly increasing vector starting at zero")
        if n_paths < 2:
            raise ValueError("n_paths must be at least two")
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even when antithetic sampling is enabled")

        n_steps = len(grid) - 1
        rng = np.random.default_rng(seed)
        if antithetic:
            base = rng.standard_normal((n_paths // 2, n_steps, 2))
            z1 = np.concatenate((base[:, :, 0], -base[:, :, 0]), axis=0)
            z2 = np.concatenate((base[:, :, 1], -base[:, :, 1]), axis=0)
        else:
            base = rng.standard_normal((n_paths, n_steps, 2))
            z1, z2 = base[:, :, 0], base[:, :, 1]

        rates = np.empty((n_paths, len(grid)), dtype=float)
        dfs = np.empty_like(rates)
        x = np.zeros(n_paths, dtype=float)
        log_df = np.zeros(n_paths, dtype=float)
        rates[:, 0] = self.phi(0.0)
        dfs[:, 0] = 1.0

        a, sigma = self.a, self.sigma
        for i, dt in enumerate(np.diff(grid)):
            decay = np.exp(-a * dt)
            b = -np.expm1(-a * dt) / a
            var_x = sigma**2 * (-np.expm1(-2.0 * a * dt)) / (2.0 * a)
            var_integral = self._integrated_ou_variance(a, sigma, float(dt))
            cov = sigma**2 * (1.0 - decay) ** 2 / (2.0 * a**2)
            sd_x = np.sqrt(max(var_x, 0.0))
            if sd_x > 0.0:
                loading = cov / sd_x
                residual_sd = np.sqrt(max(var_integral - loading**2, 0.0))
                dx = sd_x * z1[:, i]
                integral_noise = loading * z1[:, i] + residual_sd * z2[:, i]
            else:
                dx = np.zeros(n_paths, dtype=float)
                integral_noise = np.zeros(n_paths, dtype=float)

            integrated_x = x * b + integral_noise
            log_df -= self._phi_integral(float(grid[i]), float(grid[i + 1])) + integrated_x
            x = decay * x + dx
            rates[:, i + 1] = x + self.phi(float(grid[i + 1]))
            dfs[:, i + 1] = np.exp(log_df)

        return SimulationResult(grid, rates, dfs)
