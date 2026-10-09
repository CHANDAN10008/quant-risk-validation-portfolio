"""Exposure aggregation, collateral mechanics, and unilateral CVA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .curve import DiscountCurve
from .hull_white import HullWhiteModel, SimulationResult
from .instruments import InterestRateSwap


@dataclass(frozen=True)
class CollateralAgreement:
    """Symmetric variation-margin agreement; collateral is positive if held."""

    threshold: float = 0.0
    minimum_transfer_amount: float = 0.0
    margin_period_of_risk: float = 0.0

    def __post_init__(self) -> None:
        if self.threshold < 0.0 or self.minimum_transfer_amount < 0.0:
            raise ValueError("threshold and minimum transfer amount cannot be negative")
        if self.margin_period_of_risk < 0.0:
            raise ValueError("margin period of risk cannot be negative")


@dataclass(frozen=True)
class ExposureResult:
    profile: pd.DataFrame
    exposures: NDArray[np.float64]
    cva: float
    epe: float
    netting: bool
    collateralized: bool


def _position_values(
    swaps: list[InterestRateSwap],
    model: HullWhiteModel,
    simulation: SimulationResult,
) -> NDArray[np.float64]:
    n_paths, n_times = simulation.short_rates.shape
    values = np.empty((n_paths, n_times, len(swaps)), dtype=float)
    for j, swap in enumerate(swaps):
        fixings = swap.reset_fixings(model, simulation)
        for i, time in enumerate(simulation.times):
            values[:, i, j] = swap.value_at(
                model, float(time), simulation.short_rates[:, i], fixings
            )
    return values


def _apply_collateral(
    values: NDArray[np.float64],
    times: NDArray[np.float64],
    agreement: CollateralAgreement,
) -> NDArray[np.float64]:
    """Return positive exposure after a pathwise collateral process.

    Calls occur at simulation grid dates. Collateral stays fixed during the
    MPOR, using the latest call on or before t-MPOR.
    """
    if values.ndim == 2:
        positions = values[:, :, None]
    elif values.ndim == 3:
        positions = values
    else:
        raise ValueError("values must have shape paths x dates or paths x dates x trades")

    n_paths, n_times, n_positions = positions.shape
    collateral = np.zeros_like(positions)
    current = np.zeros((n_paths, n_positions), dtype=float)
    for i in range(n_times):
        mark = positions[:, i, :]
        target = np.where(
            mark > agreement.threshold,
            mark - agreement.threshold,
            np.where(mark < -agreement.threshold, mark + agreement.threshold, 0.0),
        )
        transfer = np.abs(target - current) >= agreement.minimum_transfer_amount
        current = np.where(transfer, target, current)
        collateral[:, i, :] = current

    lagged = np.empty_like(positions)
    for i, time in enumerate(times):
        cutoff = time - agreement.margin_period_of_risk
        call_idx = int(np.searchsorted(times, cutoff, side="right") - 1)
        call_idx = max(0, min(call_idx, i))
        lagged[:, i, :] = collateral[:, call_idx, :]
    return np.maximum(positions - lagged, 0.0).sum(axis=2)


def unilateral_cva(
    times: NDArray[np.float64],
    expected_exposure: NDArray[np.float64],
    curve: DiscountCurve,
    hazard_rate: float,
    recovery_rate: float,
) -> float:
    """Discrete unilateral CVA with midpoint discounting and trapezoidal EE."""
    if hazard_rate < 0.0:
        raise ValueError("hazard_rate cannot be negative")
    if not 0.0 <= recovery_rate <= 1.0:
        raise ValueError("recovery_rate must be between zero and one")
    if len(times) != len(expected_exposure) or len(times) < 2:
        raise ValueError("times and expected_exposure must be equal-length vectors")
    survival = np.exp(-hazard_rate * times)
    default_probability = survival[:-1] - survival[1:]
    midpoint = 0.5 * (times[:-1] + times[1:])
    interval_ee = 0.5 * (expected_exposure[:-1] + expected_exposure[1:])
    discounted_lgd = (1.0 - recovery_rate) * curve.discount(midpoint) * interval_ee
    return float(np.sum(discounted_lgd * default_probability))


def build_exposure_profile(
    swaps: list[InterestRateSwap],
    model: HullWhiteModel,
    simulation: SimulationResult,
    hazard_rate: float = 0.02,
    recovery_rate: float = 0.40,
    netting: bool = True,
    collateral: CollateralAgreement | None = None,
    pfe_quantile: float = 0.975,
) -> ExposureResult:
    """Calculate EE, running EPE, PFE, and CVA for a swap portfolio."""
    if not swaps:
        raise ValueError("at least one swap is required")
    if not 0.0 < pfe_quantile < 1.0:
        raise ValueError("pfe_quantile must be between zero and one")
    trade_values = _position_values(swaps, model, simulation)
    if netting:
        values = trade_values.sum(axis=2)
    else:
        values = trade_values

    if collateral is None:
        if values.ndim == 2:
            exposures = np.maximum(values, 0.0)
        else:
            exposures = np.maximum(values, 0.0).sum(axis=2)
    else:
        exposures = _apply_collateral(values, simulation.times, collateral)

    ee = exposures.mean(axis=0)
    pfe = np.quantile(exposures, pfe_quantile, axis=0)
    running_epe = np.empty_like(ee)
    running_epe[0] = ee[0]
    if len(ee) > 1:
        areas = 0.5 * (ee[:-1] + ee[1:]) * np.diff(simulation.times)
        cumulative = np.concatenate(([0.0], np.cumsum(areas)))
        running_epe[1:] = cumulative[1:] / simulation.times[1:]
    cva = unilateral_cva(
        simulation.times, ee, model.curve, hazard_rate, recovery_rate
    )
    profile = pd.DataFrame(
        {
            "time_years": simulation.times,
            "EE": ee,
            "EPE_running_average": running_epe,
            f"PFE_{pfe_quantile:.3f}": pfe,
        }
    )
    horizon = float(simulation.times[-1])
    epe_area = np.sum(0.5 * (ee[:-1] + ee[1:]) * np.diff(simulation.times))
    epe = float(epe_area / horizon) if horizon > 0.0 else float(ee[0])
    return ExposureResult(
        profile=profile,
        exposures=exposures,
        cva=cva,
        epe=epe,
        netting=netting,
        collateralized=collateral is not None,
    )
