# Data

## External data

The only data from outside the repository are extracts of the U.S. Census Bureau's 2024 American Community Survey 1-Year Public Use Microdata Sample, person records of ten states. `acs/data/README.md` is their data notice: source, citation, terms of use, columns, filters, and how `make acs-data` rebuilds them from the Census files. `acs/data/manifest.json` records the SHA-256 of every Census file and of every extract.

| File | State | Role | Records | Census file |
| --- | --- | --- | ---: | --- |
| `acs/data/ca.csv.gz` | California | source | 20,000 of 202,998 | `csv_pca.zip` |
| `acs/data/tx.csv.gz` | Texas | alternate source | 20,000 of 147,511 | `csv_ptx.zip` |
| `acs/data/wy.csv.gz` | Wyoming | primary target | 3,198 | `csv_pwy.zip` |
| `acs/data/vt.csv.gz` | Vermont | target | 3,835 | `csv_pvt.zip` |
| `acs/data/ak.csv.gz` | Alaska | target | 3,496 | `csv_pak.zip` |
| `acs/data/nd.csv.gz` | North Dakota | target | 4,505 | `csv_pnd.zip` |
| `acs/data/sd.csv.gz` | South Dakota | target | 5,130 | `csv_psd.zip` |
| `acs/data/de.csv.gz` | Delaware | target | 4,896 | `csv_pde.zip` |
| `acs/data/ri.csv.gz` | Rhode Island | target | 5,604 | `csv_pri.zip` |
| `acs/data/mt.csv.gz` | Montana | target | 5,772 | `csv_pmt.zip` |

Every extract has 13 columns. The other files of `acs/data/` are `manifest.json`; `sample_counts.csv`, the records of each state after each filter; `topcodes.csv`, the rows of the Census top-code table for the ten states; and `population.csv`, descriptive statistics of all filtered records of California and Texas before subsampling.

## Saved results

Every other data set is simulated by the code here from recorded seeds. `make check` validates each file below, and `make reproduce` regenerates it and compares it with the saved copy.

### Single-quantile study (`singlequantile/`)

| File | Rows | What it holds |
| --- | ---: | --- |
| `reference/replications.csv.gz` | 36,000 | Squared errors of the transfer, target-only and pooled fits, bound counts, fallback flag and transfer norm in each replication |
| `reference/summary.csv` | 54 | Mean, MCSE, median and maximum squared error, with and without coefficient bounds |
| `reference/paired.csv` | 18 | Transfer minus target-only and transfer minus pooled, paired within replication |
| `reference/activity.csv` | 9 | Percent of replications with an active coefficient bound or ball |
| `reference/batches.csv` | 180 | Means of transfer minus target-only over blocks of 200 replications |
| `reference/metadata.json` | | Configuration and package versions |
| `grid/reference/raw_results.csv.gz` | 28,000 | The grid: one row per scenario, method and replication |
| `grid/reference/summary.csv` | 140 | The grid: means, standard deviations, medians and MCSEs of each outcome, by scenario and method |
| `grid/reference/paired_comparisons.csv` | 28 | The grid: differences between methods, paired within replication, by scenario |
| `grid/reference/config.json` | | The grid's design as the run recorded it |

### Quartile study and slope-test run (`multiquantile/`)

| File | Rows | What it holds |
| --- | ---: | --- |
| `reference/quartiles/replications.csv` | 10,800 | Prediction errors and slopes of the composite and separate fits, and the slope test, in each replication |
| `reference/quartiles/summary.csv` | 18 | Prediction error of the composite and separate fits and their paired difference, by error law, scale slope and quartile |
| `reference/quartiles/efficiency.csv` | 2 | Null slope-variance ratio of the composite and median fits, with its analytic limit |
| `reference/quartiles/test_rates.csv` | 6 | Rejection rates of the slope test in the 600 replications of the quartile study |
| `reference/quartiles/report_table.tex` | | The study's summary table as a LaTeX table body |
| `reference/quartiles/metadata.json`, `solver_diagnostics.json` | | Configuration, source coefficients and versions; objective gaps of the composite fits |
| `reference/slope-test/statistics.csv` | 30,000 | Wald statistic and p-value for each replication, error law and scale slope |
| `reference/slope-test/test_rates.csv` | 6 | Rejection counts, rates and MCSEs of the 5% test in 5,000 replications |
| `reference/slope-test/metadata.json` | | Configuration and versions |

### Closed-form values (`closedform/reference/`)

| File | Rows | What it holds |
| --- | ---: | --- |
| `constants.csv` | 3 | Quantiles, densities, variance multipliers, optimal weights, v_bar and Gamma of the three quartiles |
| `boundary_power.csv` | 16 | Boundary noncentrality and limiting power of the 5% slope test, outer quartile and equal weights |
| `least_squares.csv` | 4 | Power of the least-squares test at noncentrality nu |
| `boundary_curves.csv` | 151 | Power curves against the relative size of the departure |
| `quartile_approximation.csv` | 6 | Local approximation to the quartile study |
| `crossover.csv` | 2 | Crossover of the local approximation |
| `single_quantile_population.csv` | 6 | Population error of the transfer class in the single-quantile design |
| `example_bound.csv` | 1 | The input bound of Example 2.1 used in Remark 3.3 |
| `metadata.json` | | Levels, laws, dimensions, grid and versions |

### ACS study (`acs/reference/`)

| File | Rows | What it holds |
| --- | ---: | --- |
| `descriptives.csv` | 10 | Descriptive statistics of the samples |
| `topcode_shares.csv` | 10 | Shares of wages at or above the top code, and of predictions above it |
| `benchmark.csv` | 8 | Ten-fold cross-validated benchmark of each target |
| `transfer_replications.csv.gz` | 28,800 | Study A: held-out loss and coverage of the four methods in each split |
| `transfer_summary.csv` | 144 | Study A: mean loss and coverage with MCSEs, and the excess over the benchmark |
| `transfer_paired.csv` | 108 | Study A: transfer minus each other method, paired within split |
| `transfer_coefficients.csv` | 180 | Study A: mean and standard deviation of the transfer coefficients |
| `inputs_only_replications.csv.gz` | 6,400 | Study A: held-out loss and coverage of the fit without the source prediction in each split |
| `inputs_only_summary.csv` | 32 | Study A: that fit's mean loss and coverage with MCSEs, and the excess over the benchmark |
| `inputs_only_paired.csv` | 64 | Study A: transfer minus that fit, and that fit minus the target-only fit, paired within split |
| `sensitivity_replications.csv` | 7,200 | Study A sensitivity: the nine transfer specifications of the primary target in each split |
| `sensitivity_summary.csv` | 36 | Study A sensitivity: the nine specifications, summarized |
| `sharing_replications.csv.gz` | 38,400 | Study B: composite and separate quartile fits and the slope test in each split |
| `sharing_summary.csv` | 192 | Study B: mean losses with MCSEs and the paired difference, composite minus separate |
| `sharing_tests.csv` | 64 | Study B: rejections of the slope test, their rate and MCSE, and the shares of splits in which the composite fit lost |
| `sharing_crosstab.csv` | 256 | Study B: number of splits in each cell of test outcome by loss comparison |
| `fullsample_coefficients.csv` | 456 | Coefficients of the full-sample fits of each target |
| `fullsample_fits.csv` | 128 | In-sample coverage of each full-sample fit, and the slope test |
| `metadata.json`, `solver_diagnostics.json` | | Configuration, source coefficients, records per sample and versions; linear programs, uniqueness margins, dropped columns and failed tests |

### Figures (`figures/reference/`)

`boundary_power`, `sample_size` and `acs_loss` (PDF and PNG) are Figures 1, 2 and 3 of the paper, drawn with Matplotlib by the scripts in `figures/` from the files above. The figures in the PDF were typeset with the paper from the same data.
