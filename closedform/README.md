# Closed-form values

These files hold the numbers of the thesis that come from formulas rather than simulation: constants of the three quartiles under a location model, the limiting power of the 5% slope test at the departure where sharing slopes stops helping prediction, a local approximation to the quartile study, the population error of the transfer class in the single-quantile design, and the bound on the inputs of the paper's Example 2.1 that Remark 3.3 uses. Nothing is random. `local.py` holds the formulas and uses `multiquantile/weighting.py` for the covariance and the optimal weights.

## Run it

```sh
.venv/bin/python -m closedform.run_closedform --out build/closed-form
.venv/bin/python -m closedform.validate_closedform --results build/closed-form
```

The validator recomputes every file and compares it under the rule of `scripts/tolerance.py`. It also requires the variance ratio to equal the analytic limit in `multiquantile/reference/quartiles/efficiency.csv`, and each power curve to start at 0.05, rise, and equal the boundary power at relative size 1.

## Notation

Levels tau_j = 0.25, 0.50, 0.75. For an error law with quantile function F^-1 and density f (standard normal, Student t with 3 degrees of freedom, standard Cauchy, all unscaled):

- v_j = F^-1(tau_j) and f_j = f(v_j);
- K_jk = min(tau_j, tau_k) − tau_j tau_k, and Omega_jk = K_jk / (f_j f_k), the covariance multiplier of the three separate slope estimates;
- c_j = Omega_jj, the variance multiplier of the separate slope at level j; c_C = sum_jk K_jk / (sum_j f_j)^2, that of the composite slope; c_* = the minimum of w'Omega w over weights w >= 0 that sum to 1, with the minimizing weights;
- v_bar = sum_j f_j v_j / sum_j f_j, the density-weighted mean quantile;
- Gamma = (R v)' (R Omega R')^-1 (R v), with R the 2 × 3 matrix of adjacent differences;
- power(nu, lambda) = P(chi-squared with nu degrees of freedom and noncentrality lambda > its 0.95 quantile at lambda = 0), which is 0.05 at lambda = 0.

With d inputs the slope test has nu = 2d degrees of freedom. The boundary noncentrality at level j is Lambda_j(d) = d Gamma (c_j − c_C) / (v_j − v_bar)^2, and with equal weights on the three levels it is d Gamma sum_j (c_j − c_C) / sum_j (v_j − v_bar)^2. A departure of relative size r (r = 1 at the boundary) has noncentrality r^2 Lambda_j(d).

For the quartile study (n = 300, d = 2, scale slope b, squared departure S = n b^2 / 3), the approximate difference in prediction error, composite minus separate, times 1,000, is (1000 / n) ((c_C − c_j) d + (v_j − v_bar)^2 S), and the approximate rejection rate is 100 power(4, Gamma S). The outer-quartile difference changes sign at b_cross = (3 (c_0.25 − c_C) d / (n v_0.75^2))^(1/2).

In the single-quantile design at level tau = 0.90, the target quantile departs from the transfer class by omega G with G uniform on [−1, 1], and the error is centered at its tau-quantile and scaled to unit interquartile range. Let a be the tau-quantile of omega G plus the error. The population check-loss fit of the transfer class misses the target quantile by a − omega G, so its mean squared error is omega^2 / 3 + a^2.

## Files

| File | One row per | Content |
| --- | --- | --- |
| `constants.csv` | error law | v_j, f_j, c_j, c_C, c_*, the optimal weights, v_bar and Gamma |
| `boundary_power.csv` | law (normal, t3) × d (1, 2, 5, 10) × criterion | nu = 2d, the boundary noncentrality at an outer quartile (`outer_quartile`) or with equal weights (`equal_weights`), and the power there |
| `least_squares.csv` | nu (2, 4, 10, 20) | power(nu, nu), the classical boundary for nu restrictions in least squares |
| `boundary_curves.csv` | relative size r = 0, 0.02, ..., 3.00 | power(2d, r^2 Lambda_0.25(d)) for each law and d, the outer-quartile curves |
| `quartile_approximation.csv` | law × scale slope (0, 0.25, 0.75) | The approximate differences at the three quartiles and the approximate rejection percentage, n = 300 and d = 2 |
| `crossover.csv` | law | b_cross, the power there, the approximation cost 0.75^2 v_0.75^2 / 3 and the variance ratio c_C / c_0.50 |
| `example_bound.csv` | one row | M^2, the largest squared norm of the ideal transfer inputs (1, q_S, z1, z2, x1) of Example 2.1, the coefficient radius A = 100, and A M |
| `single_quantile_population.csv` | law × omega (0.5, 1) | The shift a and the squared error omega^2 / 3 + a^2 at the 0.90 level |
| `metadata.json` | | Levels, alpha, n, d, the grid and package versions |

The figure `figures/reference/boundary_power.pdf` (and `.png`), drawn by `python -m figures.make_boundary_power`, plots `boundary_curves.csv` with the boundary powers of `boundary_power.csv` marked at r = 1.
