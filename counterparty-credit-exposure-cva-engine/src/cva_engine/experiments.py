"""Reproducible portfolio, convergence, and parameter sensitivity workflows."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .curve import DiscountCurve
from .exposure import (
    CollateralAgreement,
    ExposureResult,
    build_exposure_profile,
)
from .hull_white import HullWhiteModel, HullWhiteParameters, SimulationResult
from .instruments import InterestRateSwap
from .pricing import benchmark_swap_value
from .plotting import write_analysis_plots


def example_curve() -> DiscountCurve:
    """Smooth upward-sloping continuously compounded zero curve."""
    return DiscountCurve.from_zero_rates(
        [0.0, 1.0, 2.0, 5.0, 10.0],
        [0.0275, 0.0280, 0.0290, 0.0320, 0.0340],
    )


def example_portfolio() -> list[InterestRateSwap]:
    """A small offsetting portfolio that makes netting visible."""
    return [
        InterestRateSwap(
            name="receive_fixed_01",
            notional=25_000_000.0,
            fixed_rate=0.0340,
            maturity_years=5.0,
            payment_frequency=4,
            direction=1,
        ),
        InterestRateSwap(
            name="pay_fixed_02",
            notional=20_000_000.0,
            fixed_rate=0.0360,
            maturity_years=5.0,
            payment_frequency=4,
            direction=-1,
        ),
    ]


def example_parameters() -> HullWhiteParameters:
    return HullWhiteParameters(mean_reversion=0.10, volatility=0.01)


def example_csa() -> CollateralAgreement:
    return CollateralAgreement(
        threshold=50_000.0,
        minimum_transfer_amount=25_000.0,
        margin_period_of_risk=10.0 / 252.0,
    )


def build_time_grid(
    swaps: list[InterestRateSwap], observation_frequency: int = 52
) -> np.ndarray:
    """Weekly observation dates plus every coupon/reset date and final maturity."""
    if observation_frequency < 1:
        raise ValueError("observation_frequency must be positive")
    horizon = max(swap.maturity_years for swap in swaps)
    step = 1.0 / observation_frequency
    regular = np.arange(0.0, horizon + step * 0.5, step)
    dates = [regular]
    dates.extend(swap.payment_times for swap in swaps)
    grid = np.unique(np.round(np.concatenate(dates), 12))
    grid = grid[grid <= horizon + 1e-10]
    if not np.isclose(grid[0], 0.0):
        grid = np.insert(grid, 0, 0.0)
    if not np.isclose(grid[-1], horizon):
        grid = np.append(grid, horizon)
    return grid


def run_scenarios(
    swaps: list[InterestRateSwap],
    model: HullWhiteModel,
    simulation: SimulationResult,
    hazard_rate: float = 0.02,
    recovery_rate: float = 0.40,
    collateral: CollateralAgreement | None = None,
) -> dict[str, ExposureResult]:
    """Compare netted/gross and collateralized/uncollateralized exposures.

    In the unnetted cases, each trade is treated as its own exposure/collateral
    set; in the netted cases, all trades share one agreement.
    """
    return {
        "netted_uncollateralized": build_exposure_profile(
            swaps, model, simulation, hazard_rate, recovery_rate, netting=True
        ),
        "unnetted_uncollateralized": build_exposure_profile(
            swaps, model, simulation, hazard_rate, recovery_rate, netting=False
        ),
        "netted_collateralized": build_exposure_profile(
            swaps, model, simulation, hazard_rate, recovery_rate,
            netting=True, collateral=collateral,
        ),
        "unnetted_collateralized": build_exposure_profile(
            swaps, model, simulation, hazard_rate, recovery_rate,
            netting=False, collateral=collateral,
        ),
    }


def _subset_simulation(sim: SimulationResult, n_paths: int, antithetic: bool) -> SimulationResult:
    if n_paths > sim.short_rates.shape[0]:
        raise ValueError("requested subset exceeds the simulated path count")
    if antithetic:
        if n_paths % 2:
            raise ValueError("antithetic path subsets must be even")
        pair_count = n_paths // 2
        max_pairs = sim.short_rates.shape[0] // 2
        indices = np.concatenate((np.arange(pair_count), max_pairs + np.arange(pair_count)))
    else:
        indices = np.arange(n_paths)
    return SimulationResult(
        times=sim.times,
        short_rates=sim.short_rates[indices],
        discount_factors=sim.discount_factors[indices],
    )


def _benchmark_as_row(n_paths: int, result: PricingBenchmark, cva: float) -> dict[str, float]:
    return {
        "path_count": n_paths,
        "analytic_swap_value": result.analytic_value,
        "monte_carlo_swap_value": result.monte_carlo_value,
        "swap_standard_error": result.standard_error,
        "swap_ci_95_low": result.confidence_interval_low,
        "swap_ci_95_high": result.confidence_interval_high,
        "absolute_swap_error": result.absolute_error,
        "netted_uncollateralized_cva": cva,
    }


def convergence_analysis(
    primary_swap: InterestRateSwap,
    portfolio: list[InterestRateSwap],
    model: HullWhiteModel,
    maximum_simulation: SimulationResult,
    path_counts: list[int],
    hazard_rate: float,
    recovery_rate: float,
    antithetic: bool = True,
) -> pd.DataFrame:
    """Nested path-count study for pricing and uncollateralized netted CVA."""
    rows: list[dict[str, float]] = []
    for n_paths in sorted(set(path_counts)):
        sub = _subset_simulation(maximum_simulation, n_paths, antithetic)
        pricing = benchmark_swap_value(primary_swap, model, sub)
        exposure = build_exposure_profile(
            portfolio, model, sub, hazard_rate, recovery_rate, netting=True
        )
        rows.append(_benchmark_as_row(n_paths, pricing, exposure.cva))
    frame = pd.DataFrame(rows).sort_values("path_count").reset_index(drop=True)
    max_cva = float(frame.iloc[-1]["netted_uncollateralized_cva"])
    frame["cva_difference_vs_largest_run"] = (
        frame["netted_uncollateralized_cva"] - max_cva
    )
    return frame


def parameter_sensitivity(
    swaps: list[InterestRateSwap],
    curve: DiscountCurve,
    base_simulation: SimulationResult,
    base_model: HullWhiteModel,
    seed: int,
    path_count: int,
    hazard_rate: float,
    recovery_rate: float,
    base_csa: CollateralAgreement,
    antithetic: bool = True,
) -> pd.DataFrame:
    """One-at-a-time sensitivity of unilateral CVA to model and CSA inputs."""
    rows: list[dict[str, float | str]] = []
    base_grid = base_simulation.times

    def add_row(
        parameter: str,
        value: float,
        model: HullWhiteModel,
        simulation: SimulationResult,
        hazard: float,
        recovery: float,
        csa: CollateralAgreement,
    ) -> None:
        uncollateralized = build_exposure_profile(
            swaps, model, simulation, hazard, recovery, netting=True
        )
        collateralized = build_exposure_profile(
            swaps, model, simulation, hazard, recovery,
            netting=True, collateral=csa,
        )
        rows.append(
            {
                "parameter": parameter,
                "value": value,
                "netted_uncollateralized_cva": uncollateralized.cva,
                "netted_collateralized_cva": collateralized.cva,
                "uncollateralized_epe": uncollateralized.epe,
                "collateralized_epe": collateralized.epe,
            }
        )

    for value in [0.05, base_model.a, 0.20]:
        model = HullWhiteModel(curve, HullWhiteParameters(value, base_model.sigma))
        sim = model.simulate(base_grid, path_count, seed=seed, antithetic=antithetic)
        add_row("mean_reversion", value, model, sim, hazard_rate, recovery_rate, base_csa)

    for value in [0.0075, base_model.sigma, 0.015]:
        model = HullWhiteModel(curve, HullWhiteParameters(base_model.a, value))
        sim = model.simulate(base_grid, path_count, seed=seed, antithetic=antithetic)
        add_row("volatility", value, model, sim, hazard_rate, recovery_rate, base_csa)

    for value in [0.01, hazard_rate, 0.04]:
        add_row("hazard_rate", value, base_model, base_simulation, value, recovery_rate, base_csa)

    for value in [0.20, recovery_rate, 0.60]:
        add_row("recovery_rate", value, base_model, base_simulation, hazard_rate, value, base_csa)

    for value in [0.0, base_csa.threshold, 100_000.0]:
        csa = CollateralAgreement(value, base_csa.minimum_transfer_amount,
                                  base_csa.margin_period_of_risk)
        add_row("threshold", value, base_model, base_simulation, hazard_rate, recovery_rate, csa)

    for value in [0.0, base_csa.minimum_transfer_amount, 50_000.0]:
        csa = CollateralAgreement(base_csa.threshold, value, base_csa.margin_period_of_risk)
        add_row("minimum_transfer_amount", value, base_model, base_simulation,
                hazard_rate, recovery_rate, csa)

    for value in [0.0, base_csa.margin_period_of_risk, 20.0 / 252.0]:
        csa = CollateralAgreement(base_csa.threshold, base_csa.minimum_transfer_amount, value)
        add_row("margin_period_of_risk", value, base_model, base_simulation,
                hazard_rate, recovery_rate, csa)

    return pd.DataFrame(rows)


def run_analysis(
    output_dir: str | Path,
    n_paths: int = 5_000,
    seed: int = 2026,
    antithetic: bool = True,
    hazard_rate: float = 0.02,
    recovery_rate: float = 0.40,
) -> dict[str, Any]:
    """Run the complete reproducible example and write CSV, JSON, and SVG files."""
    if n_paths < 200 or (antithetic and n_paths % 2):
        raise ValueError("n_paths must be at least 200 and even for antithetic sampling")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    curve = example_curve()
    parameters = example_parameters()
    model = HullWhiteModel(curve, parameters)
    swaps = example_portfolio()
    csa = example_csa()
    grid = build_time_grid(swaps)
    simulation = model.simulate(grid, n_paths, seed=seed, antithetic=antithetic)

    scenarios = run_scenarios(
        swaps, model, simulation, hazard_rate, recovery_rate, collateral=csa
    )
    scenario_rows = []
    profiles: dict[str, pd.DataFrame] = {}
    for name, result in scenarios.items():
        profile = result.profile.copy()
        profile.insert(0, "scenario", name)
        profiles[name] = profile
        scenario_rows.append(
            {
                "scenario": name,
                "CVA": result.cva,
                "EPE": result.epe,
                "peak_EE": float(result.profile["EE"].max()),
                "peak_PFE_97_5": float(result.profile["PFE_0.975"].max()),
            }
        )
    scenario_summary = pd.DataFrame(scenario_rows)
    profile_table = pd.concat(profiles.values(), ignore_index=True)

    primary = swaps[0]
    counts = sorted(set([max(200, (n_paths // 5) // 2 * 2),
                         max(200, (n_paths // 2) // 2 * 2), n_paths]))
    convergence = convergence_analysis(
        primary, swaps, model, simulation, counts, hazard_rate, recovery_rate,
        antithetic=antithetic,
    )
    sensitivity = parameter_sensitivity(
        swaps, curve, simulation, model, seed, n_paths,
        hazard_rate, recovery_rate, csa, antithetic=antithetic,
    )
    pricing = benchmark_swap_value(primary, model, simulation)
    pricing_frame = pd.DataFrame([asdict(pricing)])

    scenario_summary.to_csv(output / "scenario_summary.csv", index=False)
    profile_table.to_csv(output / "exposure_profiles.csv", index=False)
    convergence.to_csv(output / "path_count_convergence.csv", index=False)
    sensitivity.to_csv(output / "parameter_sensitivity.csv", index=False)
    pricing_frame.to_csv(output / "analytic_vs_monte_carlo.csv", index=False)
    metadata = {
        "seed": seed,
        "path_count": n_paths,
        "antithetic": antithetic,
        "hazard_rate": hazard_rate,
        "recovery_rate": recovery_rate,
        "curve_times": curve.times.tolist(),
        "curve_discount_factors": curve.discount_factors.tolist(),
        "hull_white": asdict(parameters),
        "collateral": asdict(csa),
        "observation_frequency_per_year": 52,
        "model_notes": [
            "Single-curve regular-coupon swap valuation.",
            "Collateral calls occur on the weekly simulation grid.",
            "The MPOR uses the last simulated collateral call on or before default time minus MPOR.",
        ],
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_analysis_plots(profiles, scenario_summary, convergence, sensitivity, output)
    return {
        "scenario_summary": scenario_summary,
        "convergence": convergence,
        "sensitivity": sensitivity,
        "pricing": pricing,
        "output_dir": output,
    }
