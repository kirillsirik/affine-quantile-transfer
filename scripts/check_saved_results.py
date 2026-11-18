"""Validate every saved reference output without rerunning a study (part of make check).

Usage, from the repository root:
    python -m scripts.check_saved_results

The single-quantile grid: complete, unique raw rows in every design cell, the five method names,
finite outcomes, LP equality residuals, the affine-ball radius and fallback certificates, and both
saved summaries recomputed from the raw rows. The other studies use their own validators, which
also recompute each summary from the saved draws: the single-quantile, quartile, slope-test,
closed-form and ACS studies. The ACS check also verifies the committed extracts in acs/data/ and
the fitting conventions. Recomputed and saved values must agree under the comparison rule in
scripts/tolerance.py. No fit runs in this process. Exits nonzero on the first failure.
"""
import json
import sys
from pathlib import Path

import numpy as np

from acs import validate_acs
from closedform import validate_closedform
from multiquantile import validate_quartiles, validate_slope_test
from scripts.tolerance import compare_frames, read_table
from singlequantile import validate_single

ROOT = Path(__file__).resolve().parents[1]
GRID = ROOT / "singlequantile" / "grid" / "reference"
FIVE_METHODS = {"transfer", "target_only", "source_only", "pooled", "oracle_transfer"}
RAW_KEY = ["experiment", "rep", "noise", "tau", "delta", "nT", "N", "method"]
SUMMARY_KEY = ["experiment", "noise", "tau", "delta", "nS", "nT", "N", "method"]
OUTCOMES = ["mse", "centered_check", "coverage", "source_mse", "score_mse"]
RADIUS = 100.0


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def agree(label, recomputed, saved, key):
    """Require two frames to agree column by column under the comparison rule."""
    _, failures = compare_frames(recomputed[saved.columns], saved, key)
    require(not failures, f"{label}: {failures}")


def check_raw(frame, label, rows, cells, replicates):
    require(len(frame) == rows and not frame.duplicated(RAW_KEY).any(), f"{label}: expected {rows} unique rows")
    counts = frame.groupby(["experiment", "noise", "tau", "delta", "nT", "N", "method"]).size()
    require(len(counts) == cells and set(counts) == {replicates}, f"{label}: expected {cells} cells of {replicates}")
    require(set(frame.method) == FIVE_METHODS, f"{label}: method names")
    require(frame[OUTCOMES].notna().all().all() and np.isfinite(frame[OUTCOMES]).all().all(), f"{label}: missing outcomes")
    require(frame.mse.ge(0).all() and frame.coverage.between(0, 1).all(), f"{label}: outcome ranges")
    require(frame.coef_norm.max() <= RADIUS + 1e-6, f"{label}: affine norm above the radius")
    require(frame.max_equality_residual.max() < 1e-7, f"{label}: LP equality residual")
    fallback = frame[frame.affine_fallback == 1]
    require(fallback.affine_certified_gap.notna().all() and fallback.affine_certified_gap.le(1e-7).all(),
            f"{label}: fallback certificates")
    require(frame.loc[frame.affine_fallback == 0, "affine_certified_gap"].isna().all(), f"{label}: uncomputed gaps must stay missing")


def check_grid():
    cfg = json.loads((GRID / "config.json").read_text())
    raw = read_table(GRID / "raw_results.csv.gz")
    check_raw(raw, "grid raw_results", 28000, 140, cfg["replicates"])
    # summary.csv, as written by run_simulations.summarize
    aggs = {"mse_mean": ("mse", "mean"), "mse_sd": ("mse", "std"), "mse_median": ("mse", "median"),
            "replicates": ("mse", "size"), "centered_check_mean": ("centered_check", "mean"),
            "centered_check_sd": ("centered_check", "std"), "coverage_mean": ("coverage", "mean"),
            "coverage_sd": ("coverage", "std"), "source_mse_mean": ("source_mse", "mean"),
            "score_mse_mean": ("score_mse", "mean"), "max_coef_norm": ("coef_norm", "max"),
            "max_equality_residual": ("max_equality_residual", "max"),
            "source_box_active_total": ("source_box_active", "sum"), "target_box_active_total": ("target_box_active", "sum"),
            "pooled_box_active_total": ("pooled_box_active", "sum")}
    summary = raw.groupby(SUMMARY_KEY, dropna=False).agg(**aggs).reset_index()
    for metric in ("mse", "centered_check", "coverage"):
        summary[metric + "_mcse"] = summary[metric + "_sd"] / np.sqrt(summary.replicates)
    agree("grid summary.csv", summary, read_table(GRID / "summary.csv"), SUMMARY_KEY)
    # paired_comparisons.csv
    keys = SUMMARY_KEY[:-1]
    wide = raw.pivot(index=keys + ["rep"], columns="method", values="mse").reset_index()
    wide["d_target"] = wide.transfer - wide.target_only
    wide["d_pooled"] = wide.transfer - wide.pooled
    paired = wide.groupby(keys).agg(mean_difference=("d_target", "mean"), sd_difference=("d_target", "std"),
                                    replicates=("rep", "size"), transfer_win_fraction=("d_target", lambda s: np.mean(s < 0)),
                                    transfer_minus_pooled=("d_pooled", "mean")).reset_index()
    paired["paired_mcse"] = paired.sd_difference / np.sqrt(paired.replicates)
    agree("grid paired_comparisons.csv", paired, read_table(GRID / "paired_comparisons.csv"), keys)
    fallback = raw[raw.affine_fallback == 1]
    print(f"Validated the grid: {len(raw):,} rows, two summaries recomputed, "
          f"{len(fallback)} certified fallback fit(s), largest LP equality residual {raw.max_equality_residual.max():.3g}.")


def main():
    check_grid()
    validate_quartiles.verify_results(ROOT / "multiquantile" / "reference" / "quartiles")
    validate_single.verify_results(ROOT / "singlequantile" / "reference")
    validate_slope_test.verify_results(ROOT / "multiquantile" / "reference" / "slope-test")
    validate_closedform.verify_results(ROOT / "closedform" / "reference")
    validate_acs.verify_design()
    validate_acs.verify_results(ROOT / "acs" / "reference")
    print("All saved reference outputs passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
