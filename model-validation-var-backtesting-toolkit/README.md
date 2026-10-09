# Model Validation and VaR Backtesting Toolkit

A small, reproducible toolkit for comparing one-day 99% historical-simulation and parametric VaR forecasts. It reports exception counts, Kupiec proportion-of-failures and Christoffersen independence tests, and a Basel-style traffic-light result for a 250-day window. NumPy, pandas, and SciPy power the calculations; Matplotlib is available as an optional plotting dependency, with a built-in SVG fallback.

## Quick start

```bash
python -m pip install -e .
python -m var_validation --output reports/reference_run
```

The included reference run uses **synthetic returns** generated with a fixed seed. It is a software demonstration, not a NIFTY 50 result. To run the analysis on market data, provide a CSV with a `Date` column and either a decimal `Return` column or a `Close` column:

```bash
python -m var_validation --input path/to/nifty50.csv --column Close --output reports/nifty50_run
```

`--column Return` expects decimal simple returns (for example, `-0.012` means -1.2%). `--column Close` computes simple percentage returns from closing prices. The CSV is not bundled; users should obtain and review data from an appropriate source and record its provenance.

## Outputs

- `forecasts.csv`: one-step-ahead forecasts formed only from observations available before each forecast date.
- `backtest_summary.csv`: exception rate, average VaR, Kupiec and Christoffersen statistics, and Basel traffic-light result where applicable.
- `var_backtest.svg`: return and forecasted loss-limit chart with exceptions marked.
- `run_metadata.json`: seed, data type, observations, lookback, confidence level, and input details.

## Scope and limitations

This is an educational model-validation example, not a production risk system. It focuses on one-day, univariate return forecasts. It does not model liquidity, transaction costs, portfolio aggregation, volatility clustering, or overlapping horizons. The normal parametric model assumes a normal return distribution; historical simulation is sensitive to the selected window. Backtests have limited power in short samples, and a non-rejection is not proof of model correctness. The 250-day traffic-light bands are a simplified Basel-style presentation (green 0–4, yellow 5–9, red 10+ exceptions) and do not replace supervisory rules or a formal model approval process.

See [`docs/methodology.md`](docs/methodology.md) for definitions and interpretation.
