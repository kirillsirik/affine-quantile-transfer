"""Checks of a saved results directory of the closed-form values.

Usage, from the repository root:
    python -m closedform.validate_closedform [--results DIR]

Recomputes every file and compares it with the saved one under the comparison rule of
scripts/tolerance.py, then checks the link to the quartile study: the variance ratio equals the
analytic limit in multiquantile/reference/quartiles/efficiency.csv. Each power curve must equal 0.05 at
relative size 0, be nondecreasing, and equal the outer-quartile boundary power at relative size 1.
Raises AssertionError on the first failure.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.tolerance import compare_frames, read_table, within

from .local import ALPHA
from .run_closedform import DIMENSIONS, TEST_LAWS, metadata, tables

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_RESULTS = HERE / "reference"
KEYS = {"constants": ["law"], "boundary_power": ["law", "d", "criterion"], "least_squares": ["nu"],
        "boundary_curves": ["relative_size"], "quartile_approximation": ["law", "scale_slope"], "crossover": ["law"],
        "single_quantile_population": ["law", "omega"], "example_bound": ["radius"]}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(a, b):
    return bool(np.all(within(a, b)))


def verify_results(folder):
    """Validate one results directory written by run_closedform. Raises AssertionError on failure."""
    folder = Path(folder)
    saved = {}
    for name, rows in tables().items():
        recomputed = pd.DataFrame(rows).astype({"relative_size": float} if name == "boundary_curves" else {})
        saved[name] = read_table(folder / f"{name}.csv")
        _, failures = compare_frames(recomputed, saved[name], KEYS[name])
        require(not failures, f"{name}.csv: {failures}")
    stored = json.loads((folder / "metadata.json").read_text())
    expected = metadata()
    require({k: v for k, v in stored.items() if k not in ("python", "numpy", "scipy")}
            == {k: v for k, v in expected.items() if k not in ("python", "numpy", "scipy")}, "metadata.json")
    efficiency = read_table(ROOT / "multiquantile" / "reference" / "quartiles" / "efficiency.csv").set_index("noise")
    crossover = saved["crossover"].set_index("law")
    for law in TEST_LAWS:
        require(close(crossover.loc[law, "variance_ratio"], efficiency.loc[law, "theory_ratio"]),
                f"{law}: variance ratio differs from efficiency.csv")
    curves = saved["boundary_curves"].sort_values("relative_size")
    power = saved["boundary_power"].set_index(["law", "d", "criterion"])
    at_one = curves[curves.relative_size == 1.0]
    require(len(curves) == 151 and len(at_one) == 1, "boundary_curves.csv grid")
    for law in TEST_LAWS:
        for d in DIMENSIONS:
            column = curves[f"{law}_d{d}"].to_numpy()
            require(column[0] == ALPHA and np.all(np.diff(column) >= 0), f"{law}_d{d}: curve must start at 0.05 and rise")
            require(close(at_one[f"{law}_d{d}"].iloc[0], power.loc[(law, d, "outer_quartile"), "power"]),
                    f"{law}_d{d}: curve at 1 differs from the boundary power")
    print(f"Validated the closed-form tables in {folder} and their link to the quartile reference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    verify_results(args.results)


if __name__ == "__main__":
    main()
