# Quartile study and slope-test run

Section 5.2 and Table 5 of the paper report this study, which checks the limits of Section 4.4 in finite samples. The single-quantile study fits one quantile to one recentered response law at a time; this experiment uses the same responses for the 25th, 50th, and 75th percentiles. The quantities compared are the sampling variance of one target slope, squared quantile prediction error, and the probability of rejecting equality of the two slopes across the three quartiles.

## Layout

| Path | Role |
| --- | --- |
| `config/quartiles.json`, `config/slope_test.json` | Seeds, sample sizes, levels, error laws, scale slopes, test level and replication counts |
| `design.py` | Covariates, the frozen source fit, target features, true quantile coefficients and the seed streams |
| `estimation.py` | Quantile LP, composite QR with left-quantile intercepts, the slope Wald test |
| `weighting.py` | Slope weights from densities and the analytic variance multipliers, used by the closed-form values |
| `run_quartiles.py`, `run_slope_test.py` | Study runners |
| `validate_quartiles.py`, `validate_slope_test.py` | Checks of the LP formulation and of saved results |
| `reference/quartiles/` | Reference results of this experiment |
| `reference/slope-test/` | Reference results of the slope-test run at 5,000 replications |

## Run it

From the repository root, after `make setup`, `make reproduce` runs this study with the others and compares it with `reference/quartiles/`. To run it alone:

```sh
.venv/bin/python -m multiquantile.run_quartiles --workers 4 --out build/quartiles
.venv/bin/python -m multiquantile.validate_quartiles --results build/quartiles
```

The full run takes about 9 seconds with four workers on an Apple M2 Max. Add `--replicates 4` for a quick run. The output directory is required, so the reference results are never overwritten. The script checks each optimization before writing its files, and it exits on a failed fit or a singular contrast. The replication seed is a function of the master seed, component, and replication index. Worker count and completion order do not change the draws, and a run with R replications reproduces the first R replications of the full run. The reference results were generated with Python 3.12.13, NumPy 2.3.5 and SciPy 1.17.0 on macOS arm64.

## Design

Draw V and B independently uniform on [-1,1]. Write H = |V| − 1/2. The source median is mS = 0.4 + 1.5H + 0.2B. Add independent standard normal noise to 1,000 source draws and fit the source median by linear quantile regression. Freeze this independent fit for the whole experiment and use its centered prediction Ghat = cH H + cB B together with B. The source coefficients appear in metadata.json. The nonlinear basis is known; the target design is a nonsingular change of coordinates from (1,H,B). This isolates the target fit and does not evaluate source model learning.

In the target population, location m = 0.5 + 1.1 mS + 0.6B = 0.94 + 1.65H + 0.82B. Generate Y = m + (1 + lambda B) epsilon with standard normal or unscaled Student-t(3) noise and lambda in {0, .25, .75}. The conditional quantiles are exactly m + (1 + lambda B) z_tau. At lambda=0 all quantiles have the same slopes. At positive lambda only the slope on B changes with tau in the (Ghat,B) coordinates.

Each of 600 target samples is independent across replication indices, conditional on the same source fit and independent fixed 5,000-row evaluation design. Within a replication, fits share outcomes, and scale scenarios share innovations. Under each noise law and scale slope we fit three separate QR models and one equal-weight CQR model with separate intercepts and two common slopes. The unrestricted fit at tau=.50 is median-only regression. No data-dependent pooling rule is used.

## Computations

For a single-level fit, the quantile-regression dual maximizes y'v subject to D'v=0 and tau−1 ≤ v_i ≤ tau. For CQR, stack one block of observations for each quantile, use one intercept indicator per block plus the two common features, and change the dual box limits by block. SciPy minimizes −y'v. Negating its equality multipliers gives the fitted coefficients. Every fit checks equality between its check-loss objective and the dual optimum (relative tolerance 1e−7 with unit offset). validate_quartiles.py independently solves primal programs on fixed test data for all individual fits and a composite fit. Squared prediction errors use e'Qeval e, where e is the coefficient error and Qeval is the independent evaluation sample's empirical second moment of (1,Ghat,B). This is exactly the average of the 5,000 squared prediction errors.

The composite fit keeps the LP's common slopes b and sets the intercept at level tau to the left empirical quantile of the residuals y − Xb, the residual of one-based rank ceil(n tau). With the slopes fixed the loss separates by level, and every point between the residuals of ranks ceil(n tau) and floor(n tau)+1 minimizes it. Here n tau is 75, 150 and 225, so that set is a whole interval and the point the LP returns depends on the platform. The rank is computed exactly from the decimal level. The selected intercepts must reach the LP optimum under the same 1e−7 rule, and `solver_diagnostics.json` records the largest gaps.

For level j, estimate H_j by the average of D_i D_i' φ(r_ij / h_j)/h_j. The bandwidth is 1.06 × IQR(r_ij)/1.349 × n_T^(−1/5) using that level's unrestricted fitted residuals. Pair j,k has score covariance (min(tau_j,tau_k)−tau_j tau_k) times the design second moment. Stack the sandwich blocks and apply all four adjacent slope differences. Adjacent contrasts and differences from the first level span the same four restrictions and give the same Wald statistic. Compute a chi-squared(4) p-value. The kernel step does not access the known simulation density.

A prediction MCSE is the across-replication standard deviation divided by sqrt(R). For a difference, take the within-replication composite-minus-separate difference first. For a variance ratio, use the standard error of its paired delta-method influence values: ((C-mean C)^2 − ratio × (S-mean S)^2)/mean((S-mean S)^2). The binomial plug-in MCSE is zero at a sample proportion of one, which happens in the two lambda = 0.75 rows.

| File in an output directory | Content |
| --- | --- |
| replications.csv | One row per replication, noise law, scale slope, and level. The commonly fitted slope and joint Wald statistic intentionally repeat across that replication's three levels. |
| summary.csv | Each method's MSE, its MCSE, and the paired difference and MCSE. |
| efficiency.csv | Conditional-null variance ratio for the target coefficient on B, a paired MCSE, and its analytic limit. |
| test_rates.csv | Rejection counts, denominators, proportions and binomial MCSEs. |
| report_table.tex | Generated LaTeX table of 1,000 × the paired prediction differences, with MCSEs, and the rejection percentages. |
| metadata.json | Complete effective configuration, fitted source coefficients and package versions. |
| solver_diagnostics.json | The intercept rule, the largest relative objective gaps of the LP and selected fits, the largest dual equality residual, and how often the LP returned a different intercept. That last count depends on the platform. |

## Slope-test run at 5,000 replications

`run_slope_test.py` reruns the slope test of this experiment alone, for replications 0 to 4,999, so that its rejection rates have a smaller Monte Carlo error than the 600 replications above. `config/slope_test.json` shares every value except the replication count with `config/quartiles.json`, and a test checks that. Each replication draws the target design from stream 3 and the innovations from streams 4 and 5, as above, fits the three separate quartile regressions on the raw regressors (1, H, B) and computes the Wald statistic. The quartile study fits on (1, Ghat, B), a nonsingular linear change of the same regressors, which leaves the statistic unchanged up to rounding, so no source fit is needed. The first 600 statistics are the quartile study's: `validate_slope_test.py` requires every one of them to agree with the `wald` column of `reference/quartiles/replications.csv` within a relative 1e-8, with the same rejection decision.

```sh
.venv/bin/python -m multiquantile.run_slope_test --workers 4 --out build/slope-test
.venv/bin/python -m multiquantile.validate_slope_test --results build/slope-test
```

| File in an output directory | Content |
| --- | --- |
| statistics.csv | One row per replication, noise law and scale slope: the Wald statistic and its chi-squared(4) p-value. |
| test_rates.csv | Rejection counts at level 0.05, denominators, proportions and binomial MCSEs, as in the quartile study. |
| metadata.json | Effective configuration, workers and package versions. |

The reference output is `reference/slope-test/`.

All observations are synthetic. The design distinguishes conditional outcome quantiles across covariates and slopes across quantiles.
