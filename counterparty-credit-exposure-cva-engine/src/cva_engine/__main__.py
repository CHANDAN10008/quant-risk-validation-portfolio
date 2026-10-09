"""Command-line entry point for the reproducible example."""

from __future__ import annotations

import argparse

from .experiments import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Hull-White counterparty exposure and CVA example."
    )
    parser.add_argument("--output", default="results", help="directory for CSV, JSON, and PNG outputs")
    parser.add_argument("--paths", type=int, default=5_000, help="Monte Carlo path count (default: 5000)")
    parser.add_argument("--seed", type=int, default=2026, help="random seed")
    parser.add_argument("--hazard-rate", type=float, default=0.02, help="constant counterparty hazard rate")
    parser.add_argument("--recovery-rate", type=float, default=0.40, help="counterparty recovery rate")
    parser.add_argument(
        "--no-antithetic",
        action="store_true",
        help="disable antithetic Gaussian paths",
    )
    args = parser.parse_args()

    result = run_analysis(
        output_dir=args.output,
        n_paths=args.paths,
        seed=args.seed,
        antithetic=not args.no_antithetic,
        hazard_rate=args.hazard_rate,
        recovery_rate=args.recovery_rate,
    )
    print("Scenario CVA summary (USD):")
    print(result["scenario_summary"].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print()
    print("Analytic versus Monte Carlo swap valuation:")
    print(result["pricing"])
    print(f"Outputs written to: {result['output_dir'].resolve()}")


if __name__ == "__main__":
    main()
