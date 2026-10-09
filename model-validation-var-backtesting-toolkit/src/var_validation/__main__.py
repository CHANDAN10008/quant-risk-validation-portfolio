"""Run the VaR backtesting example from the command line."""

from __future__ import annotations

import argparse

from .data import load_returns, synthetic_returns
from .forecast import rolling_var_forecasts
from .report import write_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare rolling historical and parametric VaR models")
    parser.add_argument("--input", help="CSV market-data file; omit to run the synthetic demonstration")
    parser.add_argument("--date-column", default="Date")
    parser.add_argument("--column", default="Return", help="CSV column containing decimal returns or closing prices")
    parser.add_argument("--lookback", type=int, default=250)
    parser.add_argument("--confidence", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--observations", type=int, default=1500, help="Synthetic demo sample size")
    parser.add_argument("--output", default="reports/reference_run")
    args = parser.parse_args()

    if args.input:
        returns = load_returns(args.input, date_column=args.date_column, value_column=args.column)
        data_type, seed, input_file = "user_supplied_market_data", None, args.input
    else:
        returns = synthetic_returns(observations=args.observations, seed=args.seed)
        data_type, seed, input_file = "synthetic_demo", args.seed, None
    forecasts = rolling_var_forecasts(returns, lookback=args.lookback, confidence=args.confidence)
    summary = write_run(
        forecasts,
        args.output,
        confidence=args.confidence,
        lookback=args.lookback,
        data_type=data_type,
        seed=seed,
        input_file=input_file,
    )
    print(summary.to_string(index=False))
    print(f"\nWrote report files to {args.output}")


if __name__ == "__main__":
    main()
