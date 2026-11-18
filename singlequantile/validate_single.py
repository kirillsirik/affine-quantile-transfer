"""Checks of a saved results directory of the single-quantile study.

Usage, from the repository root:
    python -m singlequantile.validate_single [--results DIR]

Confirms complete and unique design cells, finite values and valid bound counts, recomputes every
summary from replications.csv.gz, and links the bounded variant to the grid: for every
replication below 200 in the run, the three squared errors, the three bound counts, the fallback
flag and the transfer coefficient norm must equal the rows of singlequantile/grid/reference/raw_results.csv.gz
(known scores, nT 100, level 0.9) under the comparison rule of scripts/tolerance.py. No fit is run.
Raises AssertionError on the first failure.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.tolerance import read_table, within

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "reference"
ARCHIVE = HERE / "grid" / "reference" / "raw_results.csv.gz"
METHODS = ("transfer", "target_only", "pooled")
COUNTS = ("source_bound_active", "target_bound_active", "pooled_bound_active", "transfer_fallback")
ARCHIVED_REPLICATES = 200


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare(label, recomputed, saved, keys):
    require(len(recomputed) == len(saved), f"{label}: {len(saved)} rows, expected {len(recomputed)}")
    merged = saved.merge(recomputed, on=keys, how="outer", suffixes=("", "_recomputed"), indicator=True)
    require(bool((merged["_merge"] == "both").all()), f"{label}: design cells differ")
    for column in recomputed.columns:
        if column not in keys:
            np.testing.assert_allclose(merged[column].to_numpy(float), merged[column + "_recomputed"].to_numpy(float),
                                       err_msg=f"{label}: {column}")


def mcse(values):
    return values.std(ddof=1) / np.sqrt(len(values))


def verify_results(folder):
    """Validate one results directory written by run_single. Raises AssertionError on failure."""
    folder = Path(folder)
    cfg = json.loads((folder / "metadata.json").read_text())["configuration"]
    rows = read_table(folder / "replications.csv.gz")
    keys = ["variant", "noise", "replicate", "departure"]
    expected = {(v, n, r, d) for v in cfg["variants"] for n in cfg["noise"] for r in range(cfg["replicates"])
                for d in cfg["departures"]}
    found = list(map(tuple, rows[keys].itertuples(index=False)))
    require(len(found) == len(expected) and set(found) == expected, "missing or duplicated design cells")
    errors = [f"mse_{m}" for m in METHODS]
    require(np.isfinite(rows[errors + ["transfer_norm"]]).all().all() and rows[errors].ge(0).all().all(),
            "non-finite or negative squared error")
    require(rows[list(COUNTS)].ge(0).all().all() and rows.transfer_fallback.isin([0, 1]).all(), "bound counts")
    for variant, settings in cfg["variants"].items():
        mine = rows[rows.variant == variant]
        if settings["coefficient_box"] is None:
            require(mine[list(COUNTS[:3])].eq(0).all().all(), f"{variant}: bound counts without a bound")
        if not settings["ball"]:
            require(mine.transfer_fallback.eq(0).all(), f"{variant}: fallback without a ball")
    cells = ["variant", "noise", "departure"]
    long = rows.melt(id_vars=keys, value_vars=errors, var_name="method", value_name="error")
    long["method"] = long.method.str.removeprefix("mse_")
    summary = long.groupby(cells + ["method"]).error.agg(R="size", mean="mean", mcse=mcse, median="median",
                                                         maximum="max").reset_index()
    compare("summary", summary, read_table(folder / "summary.csv"), cells + ["method"])
    rows["d_target"] = rows.mse_transfer - rows.mse_target_only
    rows["d_pooled"] = rows.mse_transfer - rows.mse_pooled
    paired = rows.groupby(cells).agg(R=("d_target", "size"), mean_minus_target_only=("d_target", "mean"),
                                     mcse_minus_target_only=("d_target", mcse), mean_minus_pooled=("d_pooled", "mean"),
                                     mcse_minus_pooled=("d_pooled", mcse)).reset_index()
    compare("paired", paired, read_table(folder / "paired.csv"), cells)
    bounded = rows[rows.variant == "bounded"]
    activity = bounded.groupby(["noise", "departure"]).agg(
        R=("replicate", "size"), source_pct=("source_bound_active", lambda s: 100 * np.mean(s > 0)),
        target_pct=("target_bound_active", lambda s: 100 * np.mean(s > 0)),
        pooled_pct=("pooled_bound_active", lambda s: 100 * np.mean(s > 0)),
        transfer_pct=("transfer_fallback", lambda s: 100 * np.mean(s > 0))).reset_index()
    compare("activity", activity, read_table(folder / "activity.csv"), ["noise", "departure"])
    ordered = rows.sort_values(keys).assign(batch=lambda f: f.replicate // cfg["batch_size"])
    batches = ordered.groupby(cells + ["batch"]).agg(first_replicate=("replicate", "min"), R=("d_target", "size"),
                                                     mean=("d_target", "mean")).reset_index()
    compare("batches", batches, read_table(folder / "batches.csv"), cells + ["batch"])
    linked = bounded[bounded.replicate < ARCHIVED_REPLICATES]
    if len(linked):
        archive = read_table(ARCHIVE)
        archive = archive[(archive.experiment == "known") & (archive.nT == 100) & (archive.tau == cfg["level"])
                          & (archive.rep < cfg["replicates"])]
        wide = archive.pivot(index=["rep", "noise", "delta"], columns="method", values="mse")
        transfer = archive[archive.method == "transfer"].set_index(["rep", "noise", "delta"])
        index = list(zip(linked.replicate, linked.noise, linked.departure))
        pairs = [(linked[f"mse_{m}"], wide.loc[index, m]) for m in METHODS]
        pairs += [(linked["transfer_norm"], transfer.loc[index, "coef_norm"]),
                  (linked["transfer_fallback"], transfer.loc[index, "affine_fallback"])]
        pairs += [(linked[f"{k}_bound_active"], transfer.loc[index, f"{k}_box_active"]) for k in ("source", "target", "pooled")]
        for new, old in pairs:
            require(bool(np.all(within(new.to_numpy(float), old.to_numpy(float)))),
                    f"{new.name} differs from the grid")
    print(f"Validated {len(rows):,} rows and every summary in {folder}; {len(linked):,} bounded rows match the "
          f"grid.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    verify_results(args.results)


if __name__ == "__main__":
    main()
