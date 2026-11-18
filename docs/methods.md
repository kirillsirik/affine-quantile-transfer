# Methods and statistical conventions

This page collects the conventions the code relies on. The study READMEs give the full designs: `singlequantile/README.md`, `singlequantile/grid/README.md`, `multiquantile/README.md`, `closedform/README.md` and `acs/README.md`.

## Quantile regression

The check loss is rho_tau(u) = max(tau u, (tau − 1) u). Fits minimize its sum over observations. The grid's code divides by n, which does not change the minimizer.

- **Quartile study and slope-test run** (`multiquantile/estimation.py`). Each fit solves the dual linear program: maximize y'v subject to D'v = 0 and tau − 1 <= v_i <= tau, with the HiGHS dual simplex in SciPy. The coefficients are the negated equality multipliers. A fit is accepted only if the check loss of those coefficients equals the dual optimum within 1e-7 × (1 + loss). Otherwise the run stops.
- **Single-quantile grid** (`singlequantile/grid/run_simulations.py`). Each fit solves the primal program with split residuals and HiGHS, with feasibility tolerances of 1e-8. Source-only, target-only and pooled fits bound their dictionary coefficients by 10. Affine transfer must lie in a Euclidean ball of radius 100. If the unrestricted fit leaves the ball, a cutting-plane solver enforces the radius and certifies a primal-dual gap of at most 1e-7. The maximum LP equality residual is recorded in every row.
- **Single-quantile study at 2,000 replications** (`singlequantile/`). The grid's functions, with two variants. `bounded` keeps the grid's constraints: dictionary coefficients in [−10, 10] and the transfer coefficients in the ball of radius 100. `unbounded` removes both: the source, target-only and pooled fits are unconstrained quantile regressions, and the transfer fit is the unrestricted fit even when its norm exceeds 100. The omitted effect 4|Z2| − 1 = 8 (Z2)+ − 4 Z2 − 1 lies in the span of the target dictionary, and quantile regression is equivariant to adding a combination of its regressors to the response, so without the bound the target-only predictions move exactly with the departure and their error is the same at every departure.
- **ACS study** (`acs/estimation.py`). The quartile study's dual program, with per-row levels and the conventions of the next section.

## Composite quantile regression and its intercepts

Composite QR stacks one copy of the data for each level, gives each copy its own intercept, and shares the slopes. With the common slopes b fixed, the loss separates by level. At level tau, every point between the residuals of one-based ranks ceil(n tau) and floor(n tau) + 1 of y − Xb minimizes it. When n tau is not an integer those ranks coincide and the intercept is unique. When n tau is an integer k, the whole interval between the k-th and (k+1)-th residuals is optimal. A simplex solver returns one end or the other, and which end depends on last-bit differences in the inputs, so it varies by platform. In the quartile study n = 300 and n tau is 75, 150 and 225, so this happens at every level.

`fit_composite` keeps the slopes from the LP and sets each intercept to the left end of its interval, the left empirical quantile of the residuals: the smallest residual r with #{r_i <= r} >= n tau. The rank is computed exactly from the decimal level, since in floating point 25 × 0.28 is 7.000000000000001, which would round up to rank 8. The selected point has the same loss as the LP point, so it is also optimal. The code still checks the loss against the dual optimum under the 1e-7 rule and records the largest gap in `solver_diagnostics.json`. For increasing levels, the intercepts are nondecreasing. Shifting y by c shifts every intercept by c. Adding X gamma to y adds gamma to the slopes and leaves the intercepts unchanged. `tests/test_intercepts.py` checks these properties and the `[0, 2, 4, 8]` median, whose interval of minimizers is [2, 4] and whose selected intercept is 2.

## The ACS study: ties, columns and uniqueness

Real survey data differ from the simulated data in two ways: every covariate is discrete, and incomes are reported at round values. The minimizer of the check loss is then often not unique, and the vertex a simplex solver returns can differ between platforms. `acs/estimation.py` fixes four conventions.

- **Row levels.** Every record of an extract carries a fixed tie number u in [0, 1), drawn once from `SeedSequence([20261114, 3, FIPS code, 0])` in the row order of the committed file. A fit at nominal level tau solves the program with row levels tau_i = tau − 1e-5 (1 + u_i). The levels move by at most 2e-5. For an intercept-only fit with n tau an integer, as in every labeled sample of the study, this selects the left empirical quantile, the rule of the composite intercepts above; `acs/validate_acs.py` checks it on 200 records of each target. In a sample of several thousand records the shift can move a fitted quantile by one order statistic. A record keeps its tie number in every fit, including the pooled fit and the stacked composite program, where each block repeats the tie numbers of its rows. Losses, coverage and the slope test always use the nominal level.
- **Columns.** If the design of a fit has lower rank than its number of columns (for example an indicator that is zero for every labeled record), the fit keeps the leftmost linearly independent columns, by successive rank checks, and sets the other coefficients to 0.
- **Uniqueness certificate.** At a fitted beta, the rows with |y_i − x_i'beta| <= 1e-7 (1 + |y_i|) are the zero-residual rows. The fit is the unique minimizer exactly when those rows span the columns and there are dual values u_i for them, with tau_i − 1 + t <= u_i <= tau_i − t for some t > 0, that balance the subgradient of the other rows: sum over zero-residual rows of u_i x_i = −(sum over positive residuals of tau_i x_i + sum over negative residuals of (tau_i − 1) x_i). A small linear program maximizes t (capped at 0.5) for every fit, and `solver_diagnostics.json` records the smallest t over the run and every fit with t below 1e-6. A value of −1 means the zero-residual rows do not span the columns. Every fit in the reference run has t > 0.
- **Coverage.** A held-out response y counts as covered by a prediction q when y <= q + 1e-8, so that a record identical to one that determines the fit counts as covered despite rounding.

The fits are unconstrained and unweighted apart from the row levels: no coefficient bounds, no survey weights, no smoothing, and no transformation of incomes beyond the logarithm in Study B. The scores of the transfer fit are principal-component scores of the eight covariates, computed from the held-out records' covariates only (covariance with that number of records as divisor, each loading vector signed so that its entry of largest absolute value is positive), and scaled by 1/sqrt(8) as in the grid's code.

## Tests and densities

- **Slope Wald test** (quartile study). The test uses the separate fits at the three quartiles. Each level's bread is a Gaussian kernel estimate with bandwidth 1.06 × IQR / 1.349 × n^(−1/5) of that level's residuals. The meat is (min(tau_j, tau_k) − tau_j tau_k) times the design second moment. All four adjacent slope differences are tested jointly against chi-squared(4) at level 0.05. The known error density is never used.

## Monte Carlo standard errors

- Means: the across-replication standard deviation over sqrt(R).
- Paired comparisons: the difference is formed within each replication first, then summarized the same way.
- Ratios (the null slope-variance ratio and the MSE ratios): the standard error of the paired delta-method influence values.
- Rejection rates: the binomial plug-in sqrt(p(1 − p)/R), which is zero when every replication rejects.

These standard errors describe simulation noise in the synthetic experiments, including the randomness of the evaluation samples. They are not statements about an empirical population.

In the ACS study the randomness is different: the records are fixed, and each replication draws a new random split of a state's records into labeled and held-out samples. The same formulas apply over the 200 splits. They measure how much a result varies across splits of the same records, not sampling variation of the state's population and not survey error. The closed-form values have no Monte Carlo error.

## Oracle inputs and feasible estimates

Every study scores its estimates against simulated truth, which no estimator sees.

- **Grid, known scores.** All methods use the population principal scores, so the scores are oracle inputs. `transfer` fits the source quantile, while `oracle_transfer` uses the true source quantile.
- **Grid, estimated scores.** The scores come from N unlabeled covariates, so `transfer` is fully feasible. `oracle_transfer` keeps the true source quantile and population scores.
- **Quartile study.** The source coefficients are estimated once and frozen, but the nonlinear basis (H, B) is known. `qr` and `cqr` are feasible given that basis. The Wald test estimates its densities. `theory_ratio` in `efficiency.csv` is an analytic limit.
- **Single-quantile study at 2,000 replications.** The grid's known-score setting: every method uses the population scores, which are oracle inputs, and `transfer` uses the estimated source fit.
- **ACS study.** No oracle input. The source fits use only the source records, the scores use only the held-out covariates of the target, and every evaluation uses held-out responses that no fit has seen. The benchmark L_ref is a cross-validated fit on all of a target's records, which a real analysis with a small labeled sample would not have; it serves only as a scale for the held-out losses.

## Closed-form values

`closedform/local.py` computes, for the levels tau_j = 0.25, 0.50, 0.75 and an error law with quantile function F^-1 and density f (standard normal, t with 3 degrees of freedom, standard Cauchy):

- v_j = F^-1(tau_j), f_j = f(v_j), K_jk = min(tau_j, tau_k) − tau_j tau_k and Omega_jk = K_jk / (f_j f_k);
- the variance multipliers c_j = Omega_jj of the separate slopes, c_C = sum_jk K_jk / (sum_j f_j)^2 of the composite slope, and c_*, the minimum of w'Omega w over nonnegative weights that sum to 1;
- v_bar = sum_j f_j v_j / sum_j f_j and Gamma = (R v)'(R Omega R')^-1 (R v), with R the 2 × 3 matrix of adjacent differences;
- power(nu, lambda) = P(a noncentral chi-squared with nu degrees of freedom and noncentrality lambda exceeds the 0.95 quantile of the central one).

With d inputs the slope test has nu = 2d degrees of freedom. Sharing the slopes stops improving prediction at level j when the noncentrality reaches Lambda_j(d) = d Gamma (c_j − c_C) / (v_j − v_bar)^2, or d Gamma sum_j (c_j − c_C) / sum_j (v_j − v_bar)^2 for the error averaged over the three levels with equal weights; a departure of relative size r has noncentrality r^2 Lambda_j(d). For the quartile study (n = 300, d = 2, scale slope b, S = n b^2 / 3) the local approximation to 1,000 × (composite minus separate prediction error) at level j is (1000 / n) ((c_C − c_j) d + (v_j − v_bar)^2 S), its approximate rejection rate is 100 power(4, Gamma S), and the outer-quartile difference changes sign at b_cross = sqrt(3 (c_0.25 − c_C) d / (n v_0.75^2)). For the single-quantile design at level 0.90, the population error of the transfer class is omega^2 / 3 + a^2, where a is the 0.90 quantile of omega G plus the grid's error, G uniform on [−1, 1]. For Remark 3.3, M^2 is the largest squared norm of the ideal inputs (1, q_S, z1, z2, x1) of Example 2.1 over the support of the covariates, and B is at least A M with A = 100. `closedform/README.md` lists the files.

## Comparing reruns

Reruns are compared with saved results under one rule, fixed on 2026-11-06 before the first rerun and not changed since. Floats agree when |a − b| <= max(1e-9 × max(|a|, |b|), 1e-10). Integers, labels and join keys must match exactly, missing values must stay missing, and generated text must match exactly. The absolute floor is the tolerance of the exact-seed reruns that validated the grid when it was first run. The relative part is orders of magnitude above the rounding differences that different BLAS libraries and processors introduce (the README lists the largest ones observed), and orders of magnitude below the precision of any reported value. Byte equality is reported separately and is not required. `scripts/tolerance.py` implements the rule and `tests/test_tolerance.py` tests it.
