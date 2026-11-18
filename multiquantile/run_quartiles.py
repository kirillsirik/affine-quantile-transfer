"""Quartile study: separate and composite quantile regressions fitted to the same target responses.

Usage, from the repository root:
    python -m multiquantile.run_quartiles --out DIR [--workers 4] [--replicates R] [--config FILE]

Input: multiquantile/config/quartiles.json (seed, sample sizes, levels, error laws, scale slopes,
test level). Writes seven files to DIR: replications.csv (one row per replication, error law, scale
slope and level), summary.csv, efficiency.csv, test_rates.csv, report_table.tex, metadata.json and
solver_diagnostics.json. Composite intercepts are left empirical quantiles of the residuals at the
fitted common slopes (see multiquantile/estimation.py).
Draws depend only on the seed and the replication index, so the worker count does not change them,
and a run with R replications reproduces the first R replications of a longer run.
"""
import argparse
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import scipy

from .common import standard_error, write_csv
from .design import (INNOVATION_STREAMS, LAWS, TARGET_LOCATION, fixed_inputs, raw_design, rng_at,
                     target_features, target_quantile_coefficients)
from .estimation import fit_composite, quantile_lp, slope_wald, summarize_fits

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "quartiles.json"


def run_replicate(r, *, cfg, source, eval_gram):
    """Fit replication r under every error law and scale slope.

    For each design, three separate QR fits and one composite QR fit share the responses. Each
    row records the joint Wald statistic, the fitted slopes on B, and the prediction error
    e' Q e of each method's coefficients, where Q is the evaluation second moment of (1, Ghat, B).
    Returns the rows and the composite fits.
    """
    n = cfg["n_target"]
    levels = np.asarray(cfg["quantiles"])
    raw = raw_design(rng_at(cfg["seed"], 3, r), n)
    design = target_features(raw, source)
    location = raw @ TARGET_LOCATION
    output, fits = [], []
    for family in cfg["noise"]:
        innovation = LAWS[family].rvs(size=n, random_state=rng_at(cfg["seed"], INNOVATION_STREAMS[family], r))
        error_quantiles = LAWS[family].ppf(levels)
        for scale_slope in cfg["scale_slopes"]:
            y = location + (1 + scale_slope * raw[:, 2]) * innovation
            separate = np.vstack([quantile_lp(y, design, tau) for tau in levels])
            shared = fit_composite(y, design[:, 1:], levels)
            fits.append(shared)
            wald, p_value = slope_wald(y, design, levels, separate)
            for j, tau in enumerate(levels):
                truth = target_quantile_coefficients(source, error_quantiles[j], scale_slope)
                row = {"replicate": r, "noise": family, "scale_slope": scale_slope,
                       "tau": tau, "wald": wald, "p_value": p_value,
                       "slope_truth": truth[-1]}
                for name, fitted in (("cqr", shared.coefficients(j)), ("qr", separate[j])):
                    error = fitted - truth
                    row[f"mse_{name}"] = float(error @ eval_gram @ error)
                    row[f"slope_{name}"] = fitted[-1]
                output.append(row)
    return output, fits


def summarize(rows, cfg):
    """Prediction summaries, the null slope-variance ratio and Wald rejection rates.

    Monte Carlo standard errors: sample standard deviation over sqrt(R); paired differences are
    formed within replication first; the variance ratio uses its paired delta-method influence
    values; rejection rates use the binomial plug-in standard error.
    """
    prediction, efficiencies, tests = [], [], []
    levels = np.asarray(cfg["quantiles"])
    bridge = np.minimum.outer(levels, levels) - np.outer(levels, levels)
    for family in cfg["noise"]:
        law = LAWS[family]
        densities = law.pdf(law.ppf(levels))
        c_pooled = bridge.sum() / np.sum(densities) ** 2
        for scale in cfg["scale_slopes"]:
            for j, tau in enumerate(levels):
                group = [row for row in rows if (row["noise"], row["scale_slope"], row["tau"]) == (family, scale, tau)]
                cqr = np.array([row["mse_cqr"] for row in group])
                qr = np.array([row["mse_qr"] for row in group])
                diff = cqr - qr
                prediction.append({"noise": family, "scale_slope": scale, "tau": tau, "R": len(diff),
                                   "mse_cqr": cqr.mean(), "mcse_cqr": standard_error(cqr),
                                   "mse_qr": qr.mean(), "mcse_qr": standard_error(qr),
                                   "paired_difference": diff.mean(), "paired_mcse": standard_error(diff)})
                if tau == levels[0]:
                    rates = np.array([row["p_value"] < cfg["alpha"] for row in group])
                    tests.append({"noise": family, "scale_slope": scale, "R": len(rates),
                                  "rejections": int(rates.sum()), "rate": rates.mean(),
                                  "mcse": np.sqrt(rates.mean() * (1 - rates.mean()) / len(rates))})
                if tau == 0.5 and scale == 0:
                    pooled = np.array([row["slope_cqr"] for row in group])
                    single = np.array([row["slope_qr"] for row in group])
                    pooled_sq = (pooled - pooled.mean()) ** 2
                    single_sq = (single - single.mean()) ** 2
                    ratio = pooled_sq.mean() / single_sq.mean()
                    ratio_influence = (pooled_sq - ratio * single_sq) / single_sq.mean()
                    efficiencies.append({"noise": family, "R": len(single),
                                         "variance_cqr": pooled.var(ddof=1), "variance_median": single.var(ddof=1),
                                         "variance_ratio": ratio, "paired_ratio_mcse": standard_error(ratio_influence),
                                         "theory_ratio": c_pooled * densities[j] ** 2 / bridge[j, j]})
    return prediction, efficiencies, tests


def table_text(prediction, tests):
    """LaTeX table of 1000 x (composite minus separate prediction error) and Wald rejection rates.

    Each prediction cell is the paired difference with its Monte Carlo standard error in
    parentheses; the last column is the rejection percentage with its binomial standard error.
    """
    lines = [r"\begin{tabular}{llrrrr}\toprule",
             r"Errors & $\lambda$ & $\Delta_{.25}$ & $\Delta_{.50}$ & $\Delta_{.75}$ & Reject (\%) \\ \midrule"]
    for test in tests:
        family, scale = test["noise"], test["scale_slope"]
        pred = [v for v in prediction if (v["noise"], v["scale_slope"]) == (family, scale)]
        cells = [f"{1000*v['paired_difference']:.1f} ({1000*v['paired_mcse']:.1f})" for v in pred]
        cells.append(f"{100*test['rate']:.1f} ({100*test['mcse']:.1f})")
        lines.append(" & ".join(["Normal" if family == "normal" else "$t_3$", f"{scale:g}"] + cells) + r" \\")
    lines.append(r"\bottomrule\end{tabular}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--replicates", type=int, help="override the replication count for a smoke run")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.replicates is not None:
        cfg["replicates"] = args.replicates
    if cfg["replicates"] < 2 or args.workers < 1:
        parser.error("use at least two replications and one worker")
    if cfg["quantiles"] != [0.25, 0.5, 0.75] or any(abs(v) >= 1 for v in cfg["scale_slopes"]):
        parser.error("this quartile experiment requires three quartiles and strictly positive scales")
    source, evaluation = fixed_inputs(cfg)
    test_design = target_features(evaluation, source)
    fit_one = partial(run_replicate, cfg=cfg, source=source, eval_gram=test_design.T @ test_design / len(test_design))
    ids = range(cfg["replicates"])
    if args.workers == 1:
        results = [fit_one(r) for r in ids]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(fit_one, ids, chunksize=4))
    rows = [row for block, _ in results for row in block]
    fits = [fit for _, block in results for fit in block]
    prediction, efficiencies, tests = summarize(rows, cfg)
    args.out.mkdir(exist_ok=True, parents=True)
    for name, records in (("replications", rows), ("summary", prediction), ("efficiency", efficiencies), ("test_rates", tests)):
        write_csv(args.out / f"{name}.csv", records)
    (args.out / "report_table.tex").write_text(table_text(prediction, tests))
    metadata = {"configuration": cfg, "source_coefficients": list(source), "workers": args.workers,
                "python": platform.python_version(), "scipy": scipy.__version__, "numpy": np.__version__}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out / "solver_diagnostics.json").write_text(json.dumps(summarize_fits(fits), indent=2) + "\n")
    print(f"Completed {len(ids)} target draws per design; {len(rows)} level-specific records in {args.out}")


if __name__ == "__main__":
    main()
