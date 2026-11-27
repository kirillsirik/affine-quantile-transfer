"""Compare a run directory with the reference outputs, under the rule in scripts/tolerance.py.

Usage, from the repository root:
    python -m scripts.compare_runs --run-dir DIR [--mode full|smoke] [--quartiles-reference DIR] [--json FILE]

Full mode expects the layout written by make reproduce: DIR/grid, DIR/single, DIR/quartiles,
DIR/slope-test, DIR/closed-form and DIR/acs.
It checks row counts, unique keys, design cells, method names, finite values and solver
diagnostics, and compares every table, generated text and configuration with its reference.
Smoke mode expects the layout of make smoke and compares each small run
with the reference rows that use the same seeds; the closed-form tables, which do not depend on a
replication count, are compared in full. Solver diagnostics are checked, not compared. Exits
nonzero if any check fails.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from acs.validate_acs import expected_programs
from scripts.tolerance import ATOL, RTOL, compare_tables, json_differences, read_table

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "singlequantile" / "grid" / "reference"
REFERENCES = {
    "quartiles": ROOT / "multiquantile" / "reference" / "quartiles",
    "single": ROOT / "singlequantile" / "reference",
    "slope-test": ROOT / "multiquantile" / "reference" / "slope-test",
    "closed-form": ROOT / "closedform" / "reference",
    "acs": ROOT / "acs" / "reference",
}
RAW_KEY = ["experiment", "rep", "noise", "tau", "delta", "nT", "N", "method"]
SUMMARY_KEY = ["experiment", "noise", "tau", "delta", "nS", "nT", "N", "method"]
PAIRED_KEY = ["experiment", "noise", "tau", "delta", "nS", "nT", "N"]
FIVE_METHODS = {"transfer", "target_only", "source_only", "pooled", "oracle_transfer"}
QUARTILE_KEY = ["replicate", "noise", "scale_slope", "tau"]
SINGLE_KEY = ["variant", "noise", "replicate", "departure"]
SLOPE_KEY = ["replicate", "noise", "scale_slope"]
ACS_METHODS = {"transfer", "target_only", "pooled", "source_only"}
# (file, key) of the tables of the new studies; the first group of each study is also compared in smoke mode.
NEW_TABLES = {
    "single": ([("replications.csv.gz", SINGLE_KEY)],
               [("summary.csv", ["variant", "noise", "departure", "method"]), ("paired.csv", ["variant", "noise", "departure"]),
                ("activity.csv", ["noise", "departure"]), ("batches.csv", ["variant", "noise", "departure", "batch"])]),
    "slope-test": ([("statistics.csv", SLOPE_KEY)], [("test_rates.csv", ["noise", "scale_slope"])]),
    "acs": ([("transfer_replications.csv.gz", ["source", "target", "n", "replicate", "method"]),
             ("sensitivity_replications.csv", ["target", "n", "replicate", "components", "retained"]),
             ("sharing_replications.csv.gz", ["scale", "target", "n", "replicate", "tau"]),
             ("benchmark.csv", ["target"]), ("descriptives.csv", ["sample"]), ("topcode_shares.csv", ["sample"]),
             ("fullsample_coefficients.csv", ["target", "scale", "fit", "tau", "term"]),
             ("fullsample_fits.csv", ["target", "scale", "fit", "tau"]),
             ("inputs_only_replications.csv.gz", ["target", "n", "replicate"])],
            [("transfer_summary.csv", ["source", "target", "n", "method"]),
             ("transfer_paired.csv", ["source", "target", "n", "contrast"]),
             ("transfer_coefficients.csv", ["source", "target", "n", "term"]),
             ("sensitivity_summary.csv", ["target", "n", "components", "retained"]),
             ("sharing_summary.csv", ["scale", "target", "n", "tau"]), ("sharing_tests.csv", ["scale", "target", "n"]),
             ("sharing_crosstab.csv", ["scale", "target", "n", "reject", "outer_worse"]),
             ("inputs_only_summary.csv", ["target", "n"]),
             ("inputs_only_paired.csv", ["source", "target", "n", "contrast"])]),
}
CLOSED_FORM_TABLES = [("constants.csv", ["law"]), ("boundary_power.csv", ["law", "d", "criterion"]),
                      ("least_squares.csv", ["nu"]), ("boundary_curves.csv", ["relative_size"]),
                      ("quartile_approximation.csv", ["law", "scale_slope"]), ("crossover.csv", ["law"]),
                      ("single_quantile_population.csv", ["law", "omega"]), ("example_bound.csv", ["radius"])]


def new_study_pairs(run, mode):
    """(label, new file, reference file, key, reference filter, subset) for the tables of the new studies."""
    pairs = []
    for study, (rows, summaries) in NEW_TABLES.items():
        for name, key in rows + (summaries if mode == "full" else []):
            label = f"{study} {name.split('.')[0]}"
            pairs.append((label, run / study / name, REFERENCES[study] / name, key, None, mode == "smoke"))
    for name, key in CLOSED_FORM_TABLES:
        pairs.append((f"closed-form {name.split('.')[0]}", run / "closed-form" / name, REFERENCES["closed-form"] / name,
                      key, None, False))
    return pairs


def table_pairs(run, mode, quartiles_ref):
    """(label, new file, reference file, key, reference filter, subset) for every compared table."""
    return [(*pair, mode == "smoke") for pair in existing_pairs(run, mode, quartiles_ref)] + new_study_pairs(run, mode)


def existing_pairs(run, mode, quartiles_ref):
    """(label, new file, reference file, key, reference filter) for the tables of the grid and the quartile study."""
    grid = run / "grid"
    if mode == "smoke":
        def core_reps(frame):
            return frame[(frame.experiment == "known") & (frame.nT == 100) & (frame.rep < 2)]
        return [
            ("grid smoke raw rows", grid / "smoke_raw.csv", GRID / "raw_results.csv.gz", RAW_KEY, core_reps),
            ("quartile replications", run / "quartiles/replications.csv", quartiles_ref / "replications.csv", QUARTILE_KEY, None),
        ]
    pairs = [
        ("grid raw_results", grid / "raw_results.csv", GRID / "raw_results.csv.gz", RAW_KEY, None),
        ("grid summary", grid / "summary.csv", GRID / "summary.csv", SUMMARY_KEY, None),
        ("grid paired", grid / "paired_comparisons.csv", GRID / "paired_comparisons.csv", PAIRED_KEY, None),
        ("quartile replications", run / "quartiles/replications.csv", quartiles_ref / "replications.csv", QUARTILE_KEY, None),
        ("quartile summary", run / "quartiles/summary.csv", quartiles_ref / "summary.csv", ["noise", "scale_slope", "tau"], None),
        ("quartile efficiency", run / "quartiles/efficiency.csv", quartiles_ref / "efficiency.csv", ["noise"], None),
        ("quartile test_rates", run / "quartiles/test_rates.csv", quartiles_ref / "test_rates.csv", ["noise", "scale_slope"], None),
    ]
    return pairs


def text_pairs(run, quartiles_ref):
    return [("quartile report_table.tex", run / "quartiles/report_table.tex", quartiles_ref / "report_table.tex")]


def json_pairs(run, quartiles_ref):
    return [
        ("grid config.json", run / "grid/config.json", GRID / "config.json"),
        ("quartile metadata.json", run / "quartiles/metadata.json", quartiles_ref / "metadata.json"),
        ("single metadata.json", run / "single/metadata.json", REFERENCES["single"] / "metadata.json"),
        ("slope-test metadata.json", run / "slope-test/metadata.json", REFERENCES["slope-test"] / "metadata.json"),
        ("closed-form metadata.json", run / "closed-form/metadata.json", REFERENCES["closed-form"] / "metadata.json"),
        ("acs metadata.json", run / "acs/metadata.json", REFERENCES["acs"] / "metadata.json"),
    ]


def design_checks(run, mode):
    """Row counts, unique keys, design cells and method names, from each run's own settings."""
    grid = run / "grid"
    grid_cfg = json.loads((grid / "config.json").read_text())
    checks = []
    if mode == "smoke":
        raw = read_table(grid / "smoke_raw.csv")
        cells = raw.groupby(["noise", "tau", "delta", "method"]).size()
        checks += [("grid smoke: 180 unique rows", len(raw) == 180 and not raw.duplicated(RAW_KEY).any()),
                   ("grid smoke: 90 cells of 2 replications", len(cells) == 90 and set(cells) == {2}),
                   ("grid smoke: method names", set(raw.method) == FIVE_METHODS)]
    else:
        raw = read_table(grid / "raw_results.csv")
        cells = raw.groupby(["experiment", "noise", "tau", "delta", "nT", "N", "method"]).size()
        checks += [
            ("grid: 28,000 unique rows", len(raw) == 28000 and not raw.duplicated(RAW_KEY).any()),
            (f"grid: 140 cells of {grid_cfg['replicates']}", len(cells) == 140 and set(cells) == {grid_cfg["replicates"]}),
            ("grid: method names", set(raw.method) == FIVE_METHODS),
            ("grid: finite outcomes", bool(np.isfinite(raw[["mse", "centered_check", "coverage", "source_mse", "score_mse"]]).all().all())),
            ("grid: LP equality residual below 1e-7", bool(raw.max_equality_residual.max() < 1e-7)),
        ]
    meta = json.loads((run / "quartiles/metadata.json").read_text())
    cfg = meta["configuration"]
    quartile = read_table(run / "quartiles/replications.csv")
    expected = cfg["replicates"] * len(cfg["noise"]) * len(cfg["scale_slopes"]) * len(cfg["quantiles"])
    checks += [
        (f"quartiles: {expected:,} unique rows", len(quartile) == expected and not quartile.duplicated(QUARTILE_KEY).any()),
        ("quartiles: design cells", set(map(tuple, quartile[["noise", "scale_slope", "tau"]].drop_duplicates().to_numpy()))
         == {(n, s, t) for n in cfg["noise"] for s in cfg["scale_slopes"] for t in cfg["quantiles"]}),
        ("quartiles: finite values", bool(np.isfinite(quartile.drop(columns="noise").to_numpy(float)).all())),
    ]
    diagnostics = run / "quartiles/solver_diagnostics.json"
    if diagnostics.exists():
        diag = json.loads(diagnostics.read_text())
        checks.append(("quartiles: selected composite intercepts attain the LP optimum",
                       diag["max_relative_gap_selected"] <= diag["objective_rtol"]))
    return checks + new_study_checks(run, mode)


def finite(frame, columns):
    return bool(np.isfinite(frame[columns].to_numpy(float)).all())


def new_study_checks(run, mode):
    """Row counts, unique keys, design cells, names, finite values and solver diagnostics of the new studies."""
    checks = []
    cfg = json.loads((run / "single/metadata.json").read_text())["configuration"]
    rows = read_table(run / "single/replications.csv.gz")
    expected = len(cfg["variants"]) * len(cfg["noise"]) * cfg["replicates"] * len(cfg["departures"])
    errors = ["mse_transfer", "mse_target_only", "mse_pooled"]
    checks += [(f"single: {expected:,} unique rows", len(rows) == expected and not rows.duplicated(SINGLE_KEY).any()),
               ("single: design cells", set(map(tuple, rows[["variant", "noise", "departure"]].drop_duplicates().to_numpy()))
                == {(v, n, d) for v in cfg["variants"] for n in cfg["noise"] for d in cfg["departures"]}),
               ("single: finite errors", finite(rows, errors + ["transfer_norm"]))]
    if mode == "full":
        checks.append(("single: method names", set(read_table(run / "single/summary.csv").method)
                       == {"transfer", "target_only", "pooled"}))
    cfg = json.loads((run / "slope-test/metadata.json").read_text())["configuration"]
    rows = read_table(run / "slope-test/statistics.csv")
    expected = cfg["replicates"] * len(cfg["noise"]) * len(cfg["scale_slopes"])
    checks += [(f"slope-test: {expected:,} unique rows", len(rows) == expected and not rows.duplicated(SLOPE_KEY).any()),
               ("slope-test: finite statistics, p-values in [0, 1]",
                finite(rows, ["wald", "p_value"]) and bool(rows.p_value.between(0, 1).all()))]
    curves = read_table(run / "closed-form/boundary_curves.csv")
    constants = read_table(run / "closed-form/constants.csv")
    checks.append(("closed-form: 3 laws, 151 grid points, finite values",
                   list(constants.law) == ["normal", "t3", "cauchy"] and len(curves) == 151
                   and finite(curves, list(curves.columns)) and finite(constants, list(constants.columns[1:]))))
    cfg = json.loads((run / "acs/metadata.json").read_text())["configuration"]
    sizes, replicates, targets = cfg["labeled_sizes"], cfg["replicates"], cfg["targets"]
    primary = cfg["primary_target"] in targets
    transfer = read_table(run / "acs/transfer_replications.csv.gz")
    expected = (len(targets) + primary) * len(sizes) * replicates * len(ACS_METHODS)
    sharing = read_table(run / "acs/sharing_replications.csv.gz")
    expected_sharing = len(cfg["scales"]) * len(targets) * len(sizes) * replicates * len(cfg["quartiles"])
    diagnostics = json.loads((run / "acs/solver_diagnostics.json").read_text())
    checks += [
        (f"acs: {expected:,} unique transfer rows", len(transfer) == expected
         and not transfer.duplicated(["source", "target", "n", "replicate", "method"]).any()),
        ("acs: transfer design cells and method names", set(transfer.target) == set(targets)
         and set(transfer.n) == set(sizes) and set(transfer.method) == ACS_METHODS),
        ("acs: finite losses, coverage in [0, 1]", finite(transfer, ["loss", "coverage"])
         and bool(transfer.coverage.between(0, 1).all())),
        (f"acs: {expected_sharing:,} unique sharing rows", len(sharing) == expected_sharing
         and not sharing.duplicated(["scale", "target", "n", "replicate", "tau"]).any()),
        ("acs: finite sharing losses", finite(sharing, ["loss_composite", "loss_separate"])),
        ("acs: every fit is the unique minimizer (margin > 0)", diagnostics["smallest_uniqueness_margin"] > 0),
        ("acs: solver diagnostics cover every program", diagnostics["linear_programs"] == expected_programs(cfg)),
    ]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    parser.add_argument("--quartiles-reference", type=Path, default=REFERENCES["quartiles"])
    parser.add_argument("--json", type=Path, help="also write the results as JSON")
    args = parser.parse_args()
    run, mode, quartiles_ref = args.run_dir.resolve(), args.mode, args.quartiles_reference.resolve()
    report = {"mode": mode, "tolerance": {"rtol": RTOL, "atol": ATOL}, "design": [], "tables": [],
              "texts": [], "json": []}
    failed = 0
    for label, passed in design_checks(run, mode):
        failed += not passed
        report["design"].append({"check": label, "passed": bool(passed)})
        print(f"design  {label:<66}{'PASS' if passed else 'FAIL'}")
    print(f"\ntolerance: |a-b| <= max({RTOL:g} * max(|a|,|b|), {ATOL:g})\n")
    print(f"{'comparison':<40}{'rows':>7}  {'bytes':<7}{'largest gap in':<24}{'max abs gap':>12}{'max rel gap':>12}  status")
    for label, new_path, old_path, key, ref_filter, subset in table_pairs(run, mode, quartiles_ref):
        result = compare_tables(label, new_path, old_path, key, subset=subset, reference_filter=ref_filter)
        floats = [m for m in result["metrics"] if m[1] == "float"]
        worst = max(floats, key=lambda m: m[3]) if floats else ("-", "", 0, 0.0, 0.0, 0)
        status = "PASS" if not result["failures"] else "FAIL"
        failed += status == "FAIL"
        report["tables"].append({**result, "metrics": [dict(zip(("column", "kind", "unequal", "max_abs_gap", "max_rel_gap",
                                                                 "outside_tolerance"), m)) for m in result["metrics"]],
                                 "status": status})
        shown = "-" if result["bytes_equal"] is None else str(result["bytes_equal"])
        print(f"{label:<40}{result['rows']:>7}  {shown:<7}{worst[0]:<24}{worst[3]:>12.3g}{worst[4]:>12.3g}  {status}")
    if mode == "full":
        print()
        for label, new_path, old_path in text_pairs(run, quartiles_ref):
            same = new_path.read_bytes() == old_path.read_bytes()
            failed += not same
            report["texts"].append({"label": label, "identical": same})
            print(f"{label:<28}text {'identical' if same else 'DIFFERENT'}")
        print()
        for label, new_path, old_path in json_pairs(run, quartiles_ref):
            differences = json_differences(json.loads(new_path.read_text()), json.loads(old_path.read_text()))
            failed += bool(differences)
            report["json"].append({"label": label, "differences": differences})
            print(f"{label:<40}{'matches (python and workers not compared)' if not differences else differences}")
    print("\nLargest gap by metric (metrics with any nonzero gap):")
    for table in report["tables"]:
        for m in table["metrics"]:
            if m["unequal"]:
                print(f"  {table['label']:<40}{m['column']:<26}{m['kind']:<6}{m['unequal']:>7} unequal"
                      f"{m['max_abs_gap']:>12.3g}{m['max_rel_gap']:>12.3g}{m['outside_tolerance']:>7} outside")
    for table in report["tables"]:
        for failure in table["failures"]:
            print(f"FAIL {table['label']}: {failure}")
    report["failed"] = failed
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=float) + "\n")
    print(f"\n{failed} check(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
