"""Counterparty credit exposure and unilateral CVA research engine."""

from .curve import DiscountCurve
from .exposure import (
    CollateralAgreement,
    ExposureResult,
    build_exposure_profile,
    unilateral_cva,
)
from .hull_white import HullWhiteModel, HullWhiteParameters, SimulationResult
from .instruments import InterestRateSwap
from .pricing import PricingBenchmark, benchmark_swap_value

__all__ = [
    "CollateralAgreement",
    "DiscountCurve",
    "ExposureResult",
    "HullWhiteModel",
    "HullWhiteParameters",
    "InterestRateSwap",
    "PricingBenchmark",
    "SimulationResult",
    "benchmark_swap_value",
    "build_exposure_profile",
    "unilateral_cva",
]

__version__ = "0.1.0"
