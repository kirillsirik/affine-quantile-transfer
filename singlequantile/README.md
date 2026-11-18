# Single-quantile study (Section 5.1)

Tables 3 and 4 of the paper, and the numbers of Section 5.1 that quote them, come from this folder: the design of the grid in `grid/` at the 0.90 level with known scores, run for 2,000 replications, with the grid's coefficient bounds and without them. Figure 2 comes from the grid itself; `grid/README.md` describes it.

## Design

Each replication draws 1,000 source observations, 100 target observations and 5,000 evaluation points from the grid's covariate law, with normal, t3 or Cauchy errors scaled to unit interquartile range and centered so that the 0.90 quantile of the error is zero. The target quantile adds 1 + Z1 and an omitted nonlinear effect omega (4|Z2| − 1) to the source quantile, with omega (`departure`, written ω in the thesis) equal to 0, 0.5 or 1. The scores are the population scores (known scores). Three fits are compared on the evaluation points:

- **transfer**: quantile regression of the target responses on 1, the source fit's prediction, the two scores and W;
- **target-only**: the source dictionary of 8 functions fitted to the 100 target observations;
- **pooled**: the dictionary fitted to the source and target observations together.

The error of a fit is the mean, over the 5,000 evaluation points, of the squared difference between its prediction and the true target 0.90 quantile.

Two variants differ only in the constraints:

- **bounded**, the grid's settings: every dictionary coefficient of the source, target-only and pooled fits in [−10, 10], and the transfer coefficients in the Euclidean ball of radius 100, enforced by the grid's cutting-plane fit when the unrestricted fit leaves the ball;
- **unbounded**: no bound on any coefficient and no ball.

## Seeds and the link to the grid

`trial.py` follows `fit_trial` of `grid/run_simulations.py` draw for draw: the generator of replication r is `SeedSequence([2026, r, e, 1, 100, 0, 0])`, where e is 0, 1 or 2 for normal, t3 or Cauchy errors; it is the grid's seed for the 0.90 level, 100 target observations and known scores. Called with the bound 10 and the ball, `trial` returns for the three fits exactly what `fit_trial` returns. Replications 0 to 199 of the bounded variant are therefore the grid's rows in `grid/reference/raw_results.csv.gz`, and replications 200 to 1,999 extend them. The validator checks that link on the saved rows, and `tests/test_single.py` checks replication 0 against `fit_trial` itself. Both variants use the same draws in each replication.

## Run it

From the repository root, after `make setup`, `make reproduce` runs this study with the others. To run it alone:

```sh
.venv/bin/python -m singlequantile.run_single --workers 4 --out build/single
.venv/bin/python -m singlequantile.validate_single --results build/single
```

`--replicates R` runs replications 0 to R − 1. The grid's fits ask HiGHS for one thread, and HiGHS keeps one thread pool per process, so the trials always run in worker processes that run nothing else, even with one worker. The worker count does not change any draw. The full run takes about 8 minutes with 4 workers on an Apple M2 Max.

## Output files

| File | One row per | Content |
| --- | --- | --- |
| `replications.csv.gz` | variant × error law × replication × departure | The three squared errors, the transfer coefficient norm, the number of coefficients at their bound in the source, target-only and pooled fits, and whether the transfer fit needed the ball |
| `summary.csv` | variant × error law × departure × method | Mean, Monte Carlo standard error, median and maximum of the squared error |
| `paired.csv` | variant × error law × departure | Mean and MCSE of transfer minus target-only and of transfer minus pooled, formed within each replication |
| `activity.csv` | error law × departure, bounded variant | Percent of replications with a coefficient at its bound in each fit, and with the ball active for the transfer fit |
| `batches.csv` | variant × error law × departure × batch | Mean of transfer minus target-only in each block of 200 consecutive replications |
| `metadata.json` | | Configuration, workers and package versions |

A mean's Monte Carlo standard error is the standard deviation over replications divided by the square root of their number. With Cauchy errors the squared errors have very heavy tails; `summary.csv` keeps the median and maximum, and `batches.csv` the block means, so that the spread across replications can be read directly.
