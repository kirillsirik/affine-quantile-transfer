"""Checks of a saved results directory of the slope-test run.

Usage, from the repository root:
    python -m multiquantile.validate_slope_test [--results DIR]

Confirms complete and unique design cells, finite statistics and p-values in [0, 1], recomputes the
rejection rates from statistics.csv, and links the run to the quartile study: for every replication
below 600 in the run, the Wald statistic must agree with the wald column of
multiquantile/reference/quartiles/replications.csv within a relative 1e-8, with the same rejection
decision. Raises AssertionError on the first failure.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .common import read_rows

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "reference" / "slope-test"
QUARTILES = HERE / "reference" / "quartiles" / "replications.csv"
LINK_RTOL = 1e-8


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def verify_results(folder):
    """Validate one results directory written by run_slope_test. Raises AssertionError on failure."""
    folder = Path(folder)
    cfg = json.loads((folder / "metadata.json").read_text())["configuration"]
    rows = read_rows(folder / "statistics.csv")
    keys = [(int(r["replicate"]), r["noise"], float(r["scale_slope"])) for r in rows]
    expected = {(i, n, s) for i in range(cfg["replicates"]) for n in cfg["noise"] for s in cfg["scale_slopes"]}
    require(len(keys) == len(expected) and set(keys) == expected, "missing or duplicated design cells")
    wald = np.array([float(r["wald"]) for r in rows])
    p_value = np.array([float(r["p_value"]) for r in rows])
    require(np.isfinite(wald).all() and (wald >= 0).all(), "non-finite or negative statistic")
    require(np.isfinite(p_value).all() and ((p_value >= 0) & (p_value <= 1)).all(), "p-value outside [0, 1]")
    saved = read_rows(folder / "test_rates.csv")
    require(len(saved) == len(cfg["noise"]) * len(cfg["scale_slopes"]), "test_rates.csv row count")
    for row in saved:
        cell = [p for k, p in zip(keys, p_value) if (k[1], k[2]) == (row["noise"], float(row["scale_slope"]))]
        rejected = sum(p < cfg["alpha"] for p in cell)
        require(int(row["R"]) == len(cell) and int(row["rejections"]) == rejected, "rejection count")
        rate = rejected / len(cell)
        np.testing.assert_allclose([float(row["rate"]), float(row["mcse"])], [rate, np.sqrt(rate * (1 - rate) / len(cell))])
    quartile = {(int(r["replicate"]), r["noise"], float(r["scale_slope"])): (float(r["wald"]), float(r["p_value"]))
                for r in read_rows(QUARTILES) if float(r["tau"]) == 0.5}
    linked = 0
    for key, statistic, p in zip(keys, wald, p_value):
        if key in quartile:
            other, other_p = quartile[key]
            require(abs(statistic - other) <= LINK_RTOL * abs(other), f"Wald statistic differs from the quartile study at {key}")
            require((p < cfg["alpha"]) == (other_p < cfg["alpha"]), f"rejection decision differs at {key}")
            linked += 1
    print(f"Validated {len(rows):,} statistics and the rejection rates in {folder}; {linked:,} match the quartile study.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    verify_results(args.results)


if __name__ == "__main__":
    main()
