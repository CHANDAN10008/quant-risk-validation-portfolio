# Methodology and validation notes

## Forecasts

For confidence level `alpha` and a rolling window of `W` prior daily returns, each forecast uses only the preceding `W` observations.

- Historical simulation: `VaR = -Q(1 - alpha)` of the window's returns.
- Parametric normal: `VaR = -(mean + stdev * Phi^-1(1 - alpha))`.

VaR is expressed as a positive loss threshold in decimal return units. An exception occurs when the realized return is below `-VaR`. Volatility uses the sample standard deviation (`ddof=1`).

## Backtests

- Kupiec POF compares the observed exception frequency with the nominal tail probability using a likelihood-ratio statistic with one asymptotic degree of freedom.
- Christoffersen independence compares the first-order exception-transition likelihood under independent exceptions with a two-state Markov alternative. Its statistic also uses one asymptotic degree of freedom.
- The Basel-style traffic-light label is calculated only for exactly 250 out-of-sample forecasts at 99% confidence: 0–4 exceptions green, 5–9 yellow, 10 or more red. This simplified count band is descriptive; it is not a regulatory determination.

Both likelihood-ratio tests can have low power when the backtest sample is short. Their asymptotic p-values should be treated cautiously when exception counts are very small. Passing these tests does not establish that a model is conceptually sound or safe for use.

## Data integrity

The built-in demo series is generated with a fixed random seed and is labelled synthetic in all run metadata. Market data is deliberately not distributed. When running with real data, preserve the original source, date range, index definition, adjustment method, missing-data handling, and any transformations alongside the output report.
