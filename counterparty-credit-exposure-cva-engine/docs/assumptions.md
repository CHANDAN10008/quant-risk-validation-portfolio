# Modeling conventions and limitations

## Market and swap valuation

- The curve is given directly as continuously compounded zero rates. Discount factors are interpolated linearly in log space; beyond the last knot the final forward rate is held constant.
- A single curve is used for both projection and discounting. The swap has regular coupon periods and a common fixed/floating frequency.
- Floating coupons are projected from the model's conditional bond prices until their reset date, then use the simulated fixing. Valuation includes the full remaining coupon cash flow.
- There are no calendars, business-day adjustments, stubs, day-count conventions, payment delays, notionals exchanges, or basis spreads.
- The Hull-White parameters are user inputs; the workflow does not calibrate them to caps, floors, or swaptions.

## Exposure and CVA

- EE is the pathwise mean of positive exposure. PFE is the selected quantile of positive exposure (97.5% by default). EPE is the time-weighted average of EE over the modeled horizon; the output also contains a running time-average EPE profile.
- Unnetted exposure sums positive trade-level exposure. In the unnetted collateralized case, each trade is treated as a separate collateral set. Netted exposure combines signed trade values before applying the positive part and a shared CSA.
- The CSA is symmetric: positive collateral is held and negative collateral is posted. Threshold, MTA, and margin calls use the same currency units as swap notional.
- Margin calls occur on the weekly simulation grid. The MPOR freezes the last simulated collateral balance at or before default time minus MPOR. Therefore the effective freeze interval is grid-dependent and may be longer than the requested MPOR.
- CVA is unilateral, assumes a constant hazard rate and constant recovery, discounts at the risk-free curve, and uses endpoint EE with midpoint discounting. It excludes bilateral CVA, DVA, funding, wrong-way risk, stochastic recovery, and default closeout disputes.
- There is no explicit default-time simulation. The discrete CVA formula approximates exposure between weekly dates by linear interpolation in expected exposure.

## Numerical interpretation

- The Hull-White Gaussian process and integrated short rate are sampled with exact interval transitions. This removes Euler bias at the observation dates, but does not remove Monte Carlo sampling error.
- The analytic pricing check applies to the sample swap under the same curve and single-curve assumptions. Its Monte Carlo confidence interval is a sampling diagnostic, not a model validation certificate.
- PFE is sensitive to path count, tail sampling, and model parameters. The CSV sensitivity and convergence outputs should be interpreted as diagnostics, not guarantees of stability.
