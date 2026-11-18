"""Checks of the quartile study: LP formulation, target law, and a saved results directory.

Usage, from the repository root:
    python -m multiquantile.validate_quartiles [--results DIR] [--design-only]

The design checks solve the separate and composite programs on a fixed test sample and compare
each check loss, including the composite loss at the left-quantile intercepts, with an independent
primal LP solved by interior point, then confirm the induced conditional quantiles. The results
checks confirm complete, unique design cells, finite values, every summary, test rate and variance
ratio recomputed from replications.csv, report_table.tex regenerated from the summaries, and the
solver diagnostics when solver_diagnostics.json is present.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack

from .common import read_rows
from .design import LAWS, TARGET_LOCATION, fixed_inputs, raw_design, rng_at, target_features
from .estimation import (OBJECTIVE_RTOL, check_loss, composite_design, fit_composite, left_empirical_quantile,
                         quantile_lp)
from .run_quartiles import table_text

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "reference" / "quartiles"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def independent_primal_solution(y, x, tau):
    """Primal LP with split residuals, solved by interior point: (minimal check loss, coefficients)."""
    n, p = x.shape
    tau = np.broadcast_to(tau, (n,))
    equality = hstack((csr_matrix(x), eye(n), -eye(n)), format="csr")
    fit = linprog(np.r_[np.zeros(p), tau, 1-tau], A_eq=equality, b_eq=y,
                  bounds=[(None, None)]*p+[(0, None)]*(2*n), method="highs-ipm")
    require(fit.success, "independent primal LP failed")
    return fit.fun, fit.x[:p]


def independent_primal(y, x, tau):
    """Minimal check loss from the primal LP with split residuals, solved by interior point."""
    return independent_primal_solution(y, x, tau)[0]


def verify_design(cfg):
    """Check the LP fits and the target conditional quantiles on a fixed 61-row test sample."""
    source, _ = fixed_inputs(cfg)
    raw = raw_design(rng_at(1234, 11), 61)
    x = target_features(raw, source)
    y = raw @ TARGET_LOCATION + rng_at(1234, 12).standard_normal(len(x))
    levels = np.array(cfg["quantiles"])
    designs = [(y, x, tau) for tau in levels]
    designs.append((np.tile(y, 3), composite_design(x[:, 1:], 3), np.repeat(levels, len(y))))
    for response, regressors, tau in designs:
        beta = quantile_lp(response, regressors, tau)
        loss = check_loss(response - regressors @ beta, tau)
        np.testing.assert_allclose(loss, independent_primal(response, regressors, tau), rtol=1e-8)
    composite = fit_composite(y, x[:, 1:], levels)
    np.testing.assert_allclose(composite.loss, independent_primal(*designs[-1]), rtol=1e-8)
    residual = y - x[:, 1:] @ composite.slopes
    require(all(composite.intercepts[j] == left_empirical_quantile(residual, tau) for j, tau in enumerate(levels)),
            "composite intercepts are not the left empirical quantiles")
    require(np.all(np.diff(composite.intercepts) >= 0), "composite intercepts are not ordered")
    # 61 tau is not an integer at any quartile, so each intercept is unique and the LP must find it too.
    require(composite.moved() == 0, "the LP intercepts differ from the unique minimizers")
    for family in cfg["noise"]:
        for slope in cfg["scale_slopes"]:
            for tau in levels:
                q = LAWS[family].ppf(tau)
                direct = raw @ TARGET_LOCATION + (1+slope*raw[:, 2])*q
                induced = x @ [0.94+q, 1.65/source[1], 0.82-1.65*source[2]/source[1]+slope*q]
                np.testing.assert_allclose(induced, direct, rtol=0, atol=1e-12)
    return source


def verify_results(folder):
    """Validate one results directory written by run_quartiles. Raises AssertionError on failure."""
    metadata = json.loads((folder/"metadata.json").read_text())
    cfg = metadata["configuration"]
    source = verify_design(cfg)
    np.testing.assert_allclose(metadata["source_coefficients"], source, rtol=1e-10)
    raw = read_rows(folder/"replications.csv")
    expected = {(fam, slope, tau, i) for fam in cfg["noise"] for slope in cfg["scale_slopes"]
                for tau in cfg["quantiles"] for i in range(cfg["replicates"])}
    keys = [(v["noise"], float(v["scale_slope"]), float(v["tau"]), int(v["replicate"])) for v in raw]
    require(len(raw) == len(expected) and set(keys) == expected, "missing or duplicated experiment cells")
    require(all(np.isfinite(float(value)) for row in raw for key, value in row.items() if key != "noise"),
            "non-finite value in replications.csv")
    groups = {}
    for row in raw:
        groups.setdefault((row["noise"], float(row["scale_slope"]), float(row["tau"])), []).append(row)
        require(min(float(row[k]) for k in ("wald", "mse_qr", "mse_cqr", "p_value")) >= 0, "negative statistic")
        require(float(row["p_value"]) <= 1, "p-value above one")
    summary = read_rows(folder/"summary.csv")
    require(len(summary) == len(groups), "summary.csv does not cover every design cell")
    for row in summary:
        key = row["noise"], float(row["scale_slope"]), float(row["tau"])
        g = groups[key]
        require(int(row["R"]) == len(g), "summary replication count")
        for method in ("cqr", "qr"):
            mse = np.array([float(r["mse_"+method]) for r in g])
            np.testing.assert_allclose([float(row["mse_"+method]), float(row["mcse_"+method])],
                                       [mse.mean(), mse.std(ddof=1)/np.sqrt(len(mse))])
        delta = np.array([float(r["mse_cqr"])-float(r["mse_qr"]) for r in g])
        np.testing.assert_allclose([float(row["paired_difference"]), float(row["paired_mcse"])],
                                   [delta.mean(), delta.std(ddof=1)/np.sqrt(len(g))])
    tests = read_rows(folder/"test_rates.csv")
    require(len(tests) == len(cfg["noise"]) * len(cfg["scale_slopes"]), "test_rates.csv row count")
    for row in tests:
        family, scale = row["noise"], float(row["scale_slope"])
        g = groups[family, scale, .5]
        rejected = sum(float(r["p_value"]) < cfg["alpha"] for r in g)
        require(rejected == int(row["rejections"]), "rejection count")
        np.testing.assert_allclose(float(row["rate"]), rejected/len(g))
        for other in (.25, .75):
            matches = zip(g, groups[family, scale, other], strict=True)
            require(all(r["replicate"] == s["replicate"] and r["wald"] == s["wald"] and
                        r["slope_cqr"] == s["slope_cqr"] for r, s in matches),
                    "Wald statistic or common slope differs across levels of one replication")
    efficiency = read_rows(folder/"efficiency.csv")
    require(len(efficiency) == len(cfg["noise"]), "efficiency.csv row count")
    for row in efficiency:
        g = groups[row["noise"], 0, .5]
        cq = np.array([float(r["slope_cqr"]) for r in g])
        qr = np.array([float(r["slope_qr"]) for r in g])
        np.testing.assert_allclose([float(row["variance_cqr"]), float(row["variance_median"]), float(row["variance_ratio"])],
                                   [cq.var(ddof=1), qr.var(ddof=1), cq.var()/qr.var()])
    prediction = [{**row, "scale_slope": float(row["scale_slope"]), "paired_difference": float(row["paired_difference"]),
                   "paired_mcse": float(row["paired_mcse"])} for row in summary]
    rates = [{**row, "scale_slope": float(row["scale_slope"]), "rate": float(row["rate"]), "mcse": float(row["mcse"])}
             for row in tests]
    require(table_text(prediction, rates) == (folder/"report_table.tex").read_text(),
            "report_table.tex does not match summary.csv and test_rates.csv")
    diagnostics = folder/"solver_diagnostics.json"
    if diagnostics.exists():
        diag = json.loads(diagnostics.read_text())
        fits = cfg["replicates"] * len(cfg["noise"]) * len(cfg["scale_slopes"])
        require(diag["composite_fits"] == fits and diag["level_fits"] == fits * len(cfg["quantiles"]),
                "solver diagnostics do not cover every composite fit")
        require(max(diag["max_relative_gap_lp"], diag["max_relative_gap_selected"]) <= OBJECTIVE_RTOL,
                "a composite fit misses the LP optimum")
    print(f"Validated the two LP formulations, conditional quantiles, {len(raw)} rows, and summary statistics in {folder}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--design-only", action="store_true", help="check the LP formulation only")
    parser.add_argument("--config", type=Path, default=HERE / "config" / "quartiles.json")
    args = parser.parse_args()
    if args.design_only:
        verify_design(json.loads(args.config.read_text()))
        print("Validated the two LP formulations and the conditional quantiles.")
    else:
        verify_results(args.results)


if __name__ == "__main__":
    main()
