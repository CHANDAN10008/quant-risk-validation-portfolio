# Quant Risk Validation Portfolio

Two reproducible Python projects focused on counterparty credit exposure and market-risk model validation.

## Projects

### [Counterparty Credit Exposure and CVA Engine](counterparty-credit-exposure-cva-engine/)

Monte Carlo swap exposure under a one-factor Hull–White short-rate model, with EE/EPE/PFE profiles, unilateral CVA, netting, collateral scenarios, analytic pricing comparison, convergence, and sensitivities. See its README for assumptions and reference results.

### [Model Validation and VaR Backtesting Toolkit](model-validation-var-backtesting-toolkit/)

Rolling historical and parametric VaR forecasts with Kupiec POF and Christoffersen independence backtests. Its checked-in reference run uses synthetic returns; the CLI accepts a CSV of market returns or closing prices for a separate run.

## Reproducibility and limits

Both projects are educational modelling examples. They document assumptions and limitations and are not production risk systems. No NIFTY 50 market data is redistributed in this repository; a user-supplied dataset must be sourced, checked, and identified separately.
