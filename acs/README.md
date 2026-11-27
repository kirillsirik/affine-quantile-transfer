# ACS income study

This study applies the two parts of the thesis to incomes in the 2024 American Community Survey (ACS). California is the large source population, and the eight least populous states are small target populations. Study A compares an affine transfer fit at the 0.90 level with fits that do not reuse the source, or reuse it differently. Study B compares separate and composite quartile regressions, and the test of equal slopes, in the same target samples. The data are public-use microdata of the U.S. Census Bureau; `data/README.md` is the data notice.

## Layout

| Path | Role |
| --- | --- |
| `config/acs.json` | Seed, states, sample sizes, levels, specifications, replication count and the constants of the fitting conventions |
| `build_data.py` | Downloads the Census files and writes the committed extracts in `data/` |
| `variables.py` | The outcome, the schooling map, the covariates and the 16-term dictionary, as functions of the extract columns; descriptive statistics |
| `estimation.py` | Quantile-regression fits with tie numbers, the column rule and the uniqueness certificate |
| `run_acs.py` | The runner: source fits, Study A, its sensitivity and alternate-source fits, the fit without the source prediction, Study B, the benchmark, full-sample fits and descriptive statistics |
| `validate_acs.py` | Checks of the extracts, the conventions and a saved results directory |
| `data/` | The ten extracts, their counts and manifest, the top codes, and statistics of the full California and Texas files |
| `reference/` | The outputs of the full run |

## Run it

From the repository root, after `make setup`, `make reproduce` runs this study with the others and compares it with `reference/`. To run it alone:

```sh
.venv/bin/python -m acs.run_acs --workers 4 --out build/acs
.venv/bin/python -m acs.validate_acs --results build/acs
```

`--replicates R` runs the first R replications and `--targets WY VT` runs only the listed targets; `make smoke` uses `--replicates 2 --targets WY`. The output directory is required, so the reference results are never overwritten. Every command reads only `data/`, and none needs the network. `make acs-data` downloads the Census files into `build/acs-raw/`, rebuilds the extracts into `build/acs-data/`, and fails unless every download matches the SHA-256 recorded in `data/manifest.json` and every rebuilt extract equals the committed one. The full run takes about 10 minutes with 4 workers on an Apple M2 Max.

## Data, outcome and covariates

The extracts hold the person records of the 2024 ACS 1-Year Public Use Microdata Sample (PUMS) of ten states, filtered as in the ACSIncome task of Ding, Hardt, Miller and Schmidt (2021): age above 16, total income above 100 dollars, positive usual hours worked, person weight at least 1, a class of worker from 1 to 7, and no missing schooling, weeks worked, sex, marital status or wage income. California and Texas are subsampled to 20,000 records each; the eight targets keep every filtered record. `data/README.md` lists the columns and the counts after each filter.

- **Outcome.** income = PINCP × (ADJINC / 10^6) / 1000, total person income in thousands of 2024 dollars. Study B also uses log(1000 × income), the logarithm of income in dollars. Incomes are used as reported: no trimming, winsorizing, jitter or other transformation.
- **Schooling** S in years from SCHL: 0 for codes 1-3, SCHL − 3 for codes 4-14, 11 for 15, 12 for 16 and 17, 12.5 for 18, 13 for 19, 14 for 20, 16 for 21, 18 for 22, 19 for 23 and 20 for 24.
- **Covariates** x, eight entries: age = (AGEP − 40)/10, school = (S − 13)/3, hours = (WKHP − 40)/10, weeks = (WKWN − 52)/10, and the indicators female (SEX = 2), selfemp (COW 6 or 7), gov (COW 3, 4 or 5) and married (MAR = 1).
- **Dictionary** Phi(x), 16 columns: 1, age, (AGEP − 30)+/10, (AGEP − 50)+/10, school, (S − 12)+/3, (S − 16)+/3, hours, (WKHP − 40)+/10, weeks, female, selfemp, gov, married, female × school and school × hours, where (t)+ = max(t, 0).

Person weights are not used in any fit: the filtered records of a state are treated as that state's population.

## Splits

Every random draw comes from `numpy.random.SeedSequence([20261114, stream, f, i])`, where f is the state's FIPS code. Stream 1 draws the California and Texas subsamples (in `build_data.py`), stream 2 the splits, stream 3 one tie number per record, and stream 4 the cross-validation folds. In replication b = 0, ..., 199 of a target with N records, `perm = rng(2, f, b).permutation(N)`. The labeled samples of sizes n = 100, 200, 400 and 800 are `perm[:n]`, so they are nested. The held-out set is `perm[800:]` for every n. Principal-component scores come from the covariates of the held-out set only: the center and the loadings of the first four components (covariance with the number of held-out records as divisor, each loading vector signed so that its entry of largest absolute value is positive), and scores (x − center) × loadings / sqrt(8). No response and no labeled record enters the scores. Both studies use the same splits.

## Study A: transfer at the 0.90 level

The source fit is the dictionary quantile regression at 0.90 on the 20,000 California records; its prediction is s(x) = Phi(x)'b_S. In each target, replication and labeled size n:

| Method | Rows | Columns |
| --- | --- | --- |
| `transfer` | the labeled sample | 1, s(x), the first two scores, school: five coefficients |
| `target_only` | the labeled sample | Phi(x): 16 coefficients |
| `pooled` | the 20,000 source records followed by the labeled sample | Phi(x) |
| `source_only` | no fit | s(x) as it is |

Each method is evaluated on the held-out records: the mean check loss at 0.90, in thousands of dollars, and the coverage, the share of held-out incomes at or below the prediction. The check loss is rho_tau(u) = max(tau u, (tau − 1) u) of the response minus the prediction.

- **Sensitivity** (Wyoming, California source): the transfer fit with r = 0, 2 or 4 scores and the retained covariates none, school, or age, school and hours, nine specifications with columns 1, s(x), the r scores, then the retained covariates. r = 2 with school is the main specification, and its rows equal the `transfer` rows.
- **Alternate source** (Wyoming): the four methods with the Texas source fit and the Texas records in the pooled fit, written with `source` = TX. The target-only rows repeat the California ones.
- **Reduced fit** (each target, California source), the fit without the source prediction: the transfer columns without s(x), that is 1, the first two scores and school, four coefficients, fitted to the labeled sample and evaluated on the held-out records like the four methods. It separates the contribution of the source prediction from the reduction to fewer coefficients. Its rows are in the three `inputs_only_*` files, and each of its linear programs has a positive uniqueness margin, recorded in the `margin` column.
- **Benchmark** (each target): ten-fold cross-validation of the dictionary fit at 0.90 on all N records, with folds `numpy.array_split(rng(4, f, 0).permutation(N), 10)`. L_ref is the mean out-of-fold check loss and `coverage_ref` the out-of-fold coverage. `excess_pct` = 100 × (mean held-out loss / L_ref − 1) measures a method against this benchmark.
- **Full sample** (each target): the dictionary fit and the transfer fit on all N records, with scores from all N records.

## Study B: common versus separate slopes at the quartiles

On each scale, `level` (income in thousands of dollars) and `log` (the logarithm of income in dollars), the source median fit is the dictionary fit at 0.50 on the California records, with prediction s_M(x). The inputs are W = (s_M(x), school) and Z = (1, W). In each labeled sample the study fits the three separate quartile regressions of the response on Z, the composite fit with one intercept per quartile and common slopes on W, and the Wald test that the slopes are equal at the three quartiles (`multiquantile.estimation.slope_wald`, 4 restrictions, rejection when p < 0.05). The held-out loss at each quartile is the mean check loss at that level, in the units of the scale. `outer_worse` is the share of splits in which the composite fit's summed held-out loss at 0.25 and 0.75 exceeds that of the separate fits; `all_worse` sums over the three quartiles. The full-sample fits repeat this on all N records of each target.

## Fitting conventions

Every covariate is discrete and incomes are reported at round values, so the minimizer of the check loss is often not unique, and a simplex solver can return a different vertex on another platform. The study therefore fixes these conventions, which `docs/methods.md` also states.

- **Tie numbers and row levels.** Each record carries a fixed number u in [0, 1) from stream 3, in the row order of its extract. A fit at nominal level tau solves the linear program with row levels tau − 10^-5 (1 + u_i). For an intercept-only fit at the sample sizes used here this selects the left empirical quantile, the rule the simulation studies use for composite intercepts. A record keeps its tie number in every fit, including the pooled fit and the stacked composite program. Losses, coverage and the test always use the nominal level.
- **Columns.** When a labeled sample makes the design rank deficient, the fit keeps the leftmost linearly independent columns and sets the other coefficients to 0. `dropped_columns` records how many.
- **Uniqueness certificate.** After every fit, a small linear program finds the largest margin t such that dual values strictly inside their bounds, by at least t, balance the rows with nonzero residuals. The fit is the unique minimizer exactly when t > 0 and the rows with zero residual span the columns. `solver_diagnostics.json` records the smallest margin and every program with a margin below 10^-6.
- **Coverage.** A held-out response counts as covered when it is at most the prediction plus 10^-8, so that a record identical to one that determines the fit counts as covered despite rounding.
- **No constraints, no weights, no smoothing.** Every fit goes through `multiquantile.estimation.quantile_lp` and is accepted only if its check loss equals the dual optimum within 10^-7 × (1 + loss). If the slope test raises an error (zero bandwidth or a singular covariance), its statistic and p-value are left missing and the failure is counted in `solver_diagnostics.json`.

## Monte Carlo standard errors

The randomness here is the random split of a fixed, finite set of records into labeled and held-out samples, not a draw from a model. A mean's Monte Carlo standard error is the standard deviation over the 200 replications divided by sqrt(200); a paired difference is formed within each replication first; a rejection rate uses sqrt(p(1 − p)/R). These standard errors describe the variation over splits of the same records. They say nothing about sampling from the state's population or about survey error.

## Output files

Floats are written with their full `repr`; missing values are empty cells.

| File | One row per | Content |
| --- | --- | --- |
| `metadata.json` | | The effective configuration (targets and replication count actually run), the source coefficient vectors, the records per sample, workers and package versions |
| `solver_diagnostics.json` | | Number of linear programs, smallest uniqueness margin and its program, the programs with a margin below 10^-6, fits with dropped columns, failed tests |
| `descriptives.csv` | sample | Records; mean and standard deviation (divisor N − 1) of income; quantiles at 0.10-0.99 (left inverse); maximum; kurtosis (divisor N); 0.99 quantile over the median; Hill tail-index estimates from the largest 2% and 5%; shares of PINCP at multiples of 1,000 and 5,000 dollars and of wages at or above the top-code threshold; share of records whose eight covariates and PINCP equal another record's; covariate means |
| `topcode_shares.csv` | sample | The wage top-code threshold in thousands of dollars of the file, the share of wages at or above it, and the share of predictions above it: the full-sample transfer fit for a target, the source fit for California and Texas |
| `benchmark.csv` | target | `records`, `L_ref`, `coverage_ref` |
| `fullsample_coefficients.csv` | target × scale × fit × level × term | Coefficients of the full-sample dictionary, transfer, separate and composite fits |
| `fullsample_fits.csv` | target × scale × fit × level | In-sample coverage of each full-sample fit, and the slope test as a row with `fit` = slope_test |
| `transfer_replications.csv.gz` | source × target × n × replication × method | Held-out loss and coverage, coefficient norm, the five transfer coefficients, dropped columns, held-out size |
| `transfer_summary.csv` | source × target × n × method | Mean loss and coverage with MCSEs, `excess_pct` |
| `transfer_paired.csv` | source × target × n × contrast | Transfer minus each other method: mean, MCSE, share of splits with a negative difference |
| `transfer_coefficients.csv` | source × target × n × term | Mean and standard deviation of the transfer coefficients |
| `inputs_only_replications.csv.gz` | target × n × replication | The fit without the source prediction: held-out loss and coverage, coefficient norm, dropped columns, uniqueness margin, held-out size |
| `inputs_only_summary.csv` | target × n | Its mean loss and coverage with MCSEs, `excess_pct` |
| `inputs_only_paired.csv` | source × target × n × contrast | Transfer minus this fit, and this fit minus the target-only fit: mean, MCSE, share of splits with a negative difference |
| `sensitivity_replications.csv`, `sensitivity_summary.csv` | n × replication × specification; n × specification | Held-out loss and coverage of the nine specifications, and the paired difference from the target-only fit |
| `sharing_replications.csv.gz` | scale × target × n × replication × level | Held-out losses of the composite and separate fits, their slopes, the Wald statistic and p-value (the composite slopes and the test repeat across the three levels) |
| `sharing_summary.csv` | scale × target × n × level | Mean losses with MCSEs and the paired difference, composite minus separate |
| `sharing_tests.csv` | scale × target × n | Rejections, rate and its MCSE, `outer_worse`, `all_worse` |
| `sharing_crosstab.csv` | scale × target × n × reject × outer_worse | Number of splits in each cell |

`sharing_tests.csv` divides rejections by the number of splits with a test statistic; R counts every split.

## Figure

`python -m figures.make_acs_loss --output FILE` draws `excess_pct` of the four methods against n, one panel per target with the primary target first, from `transfer_summary.csv` and `benchmark.csv`; bars span two Monte Carlo standard errors either side. The script sets the common vertical range from the data, so every point and bar is shown and no value is cut. The reference copies are `figures/reference/acs_loss.pdf` and `acs_loss.png`.

## Limits

The fits are unweighted, use one survey year, and treat each state's filtered records as its population. Incomes are top-coded by state and reported at round values; `descriptives.csv` and `topcode_shares.csv` report how often. The extracts are not linked to any other data.
