# Single-quantile grid

Figure 2 of the paper comes from this run: the mean squared prediction error of the transfer and target-only fits under t3 errors at the 0.90 level, with known scores and 1,000 source observations, at n = 50, 100, 200 and 400 target observations and departures ω = 0 and 1. The single-quantile study one folder up (Tables 3 and 4) runs the same functions for 2,000 replications at n = 100, and its replications 0 to 199 are this run's rows at n = 100.

## Design

The run has 28 scenarios with 200 replications each, and five methods in every replication: source-only, target-only, pooled, transfer, and transfer with the true source quantile and true scores (oracle inputs).

- **Core grid (18 scenarios):** levels 0.5 and 0.9; normal, t3 and Cauchy errors; departures 0, 0.5 and 1; 1,000 source and 100 target observations.
- **Target sample size (6):** n = 50, 200 and 400 at the 0.9 level with t3 errors and departures 0 and 1. With the core cells at n = 100 they make Figure 2.
- **Estimated scores (4):** scores estimated from N = 100 or 1,000 unlabeled covariates, at the 0.9 level, t3 errors, n = 100, departures 0 and 1.

The covariates are 20 bounded, correlated coordinates whose two leading principal-component scores are the paper's z1 and z2, and the retained coordinate is x1. The conditional quantiles are those of the paper's Example 2.1, and each response adds an error centered at the level's quantile and scaled to unit interquartile range. The dictionary of the source, target-only and pooled fits has eight functions (the paper's Section 5.1), with coefficients bounded by 10. The transfer fit has five coefficients in a Euclidean ball of radius 100. Each scenario evaluates every fit on 5,000 independent covariates, and the master seed is 2026. `docs/methods.md` describes the linear programs and the Monte Carlo standard errors.

Each scenario fits one level to one response law. The error is recentered so that the chosen level has quantile zero, and the level is part of the seed, so the 0.5 rows and the 0.9 rows are different outcome laws drawn with different seeds.

## Files

| File | Content |
| --- | --- |
| `run_simulations.py` | The design, the fits and the summaries. It writes its outputs next to itself, so run a copy of it. |
| `reference/raw_results.csv.gz` | 28,000 rows: one per scenario, method and replication, with the squared prediction error, the centered check loss, coverage, source and score errors, coefficient norms, bound counts and LP diagnostics |
| `reference/summary.csv` | Means, standard deviations, medians and Monte Carlo standard errors of each outcome, by scenario and method (140 rows) |
| `reference/paired_comparisons.csv` | Transfer minus target-only and transfer minus pooled, paired within replication, by scenario (28 rows) |
| `reference/config.json` | The design as the run recorded it |

## Run it

`make reproduce` copies the script into `build/reproduce/grid/` and runs `python run_simulations.py --replicates 200` there (about 100 seconds on an Apple M2 Max). The script uses min(4, CPU count) worker processes. The seeds depend only on the replication, the scenario and the sample sizes, so the worker count does not change the results.
