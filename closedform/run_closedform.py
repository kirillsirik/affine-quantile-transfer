"""Closed-form values: quartile constants, boundary noncentralities and powers, the local approximation
to the quartile study, the population error of the single-quantile design, and the input bound of
the paper's Example 2.1 (Remark 3.4).

Usage, from the repository root:
    python -m closedform.run_closedform --out DIR

No input data and no random draws. Uses multiquantile/weighting.py through closedform/local.py and
writes constants.csv, boundary_power.csv, least_squares.csv, boundary_curves.csv,
quartile_approximation.csv, crossover.csv, single_quantile_population.csv, example_bound.csv and
metadata.json to DIR.
closedform/README.md gives the formula behind each file.
"""
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from multiquantile.common import write_csv
from multiquantile.weighting import LAWS, LEVELS

from .local import (ALPHA, boundary_noncentrality, boundary_noncentrality_equal_weights, constants, crossover,
                    population_shift, power, quartile_approximation)

N_TARGET, INPUTS = 300, 2
DIMENSIONS = (1, 2, 5, 10)
SCALE_SLOPES = (0.0, 0.25, 0.75)
LEAST_SQUARES_NU = (2, 4, 10, 20)
GRID_STEPS, GRID_STOP = 150, 3.0
OMEGAS, SINGLE_LEVEL = (0.5, 1.0), 0.9
TEST_LAWS = ("normal", "t3")
SUFFIX = ("25", "50", "75")
EXAMPLE_RADIUS = 100.0  # the coefficient radius A of the transfer fit in Example 2.1 and the single-quantile study


def example_bound():
    """M^2, the largest squared norm of the ideal transfer inputs (1, q_S, z1, z2, x1) of Example 2.1, and A M.

    q_S = 1 + z1 + z2 + (z1)_+ + x1 with z1, x1 in [-1, 1] and z2 in [-1/2, 1/2]. On each half of the box
    split at z1 = 0, q_S is affine, so the squared norm is convex there and its maximum is at a corner.
    """
    largest = max(1 + (1 + z1 + z2 + max(z1, 0.0) + x1) ** 2 + z1 ** 2 + z2 ** 2 + x1 ** 2
                  for z1 in (-1.0, 0.0, 1.0) for z2 in (-0.5, 0.5) for x1 in (-1.0, 1.0))
    return [{"m_squared": largest, "radius": EXAMPLE_RADIUS, "radius_times_m": EXAMPLE_RADIUS * largest ** 0.5}]


def tables():
    """Every output table as a list of rows, keyed by file stem."""
    k = {name: constants(law) for name, law in LAWS.items()}
    out = {"constants": [], "boundary_power": [], "least_squares": [], "boundary_curves": [],
           "quartile_approximation": [], "crossover": [], "single_quantile_population": [],
           "example_bound": example_bound()}
    for name, c in k.items():
        row = {"law": name}
        for label, key in (("v", "v"), ("f", "f"), ("c", "c")):
            row.update({f"{label}_{s}": float(value) for s, value in zip(SUFFIX, c[key])})
        row.update({"c_composite": float(c["c_composite"]), "c_optimal": float(c["c_optimal"])})
        row.update({f"weight_{s}": float(value) for s, value in zip(SUFFIX, c["weights"])})
        row.update({"v_bar": float(c["v_bar"]), "gamma": float(c["gamma"])})
        out["constants"].append(row)
    for name in TEST_LAWS:
        for d in DIMENSIONS:
            for criterion, rule in (("outer_quartile", boundary_noncentrality),
                                    ("equal_weights", boundary_noncentrality_equal_weights)):
                lam = float(rule(k[name], d))
                out["boundary_power"].append({"law": name, "d": d, "criterion": criterion, "nu": 2 * d,
                                              "noncentrality": lam, "power": power(2 * d, lam)})
    out["least_squares"] = [{"nu": nu, "power": power(nu, nu)} for nu in LEAST_SQUARES_NU]
    for step in range(GRID_STEPS + 1):
        r = step * GRID_STOP / GRID_STEPS
        row = {"relative_size": f"{r:.2f}"}
        for name in TEST_LAWS:
            for d in DIMENSIONS:
                row[f"{name}_d{d}"] = power(2 * d, r ** 2 * float(boundary_noncentrality(k[name], d)))
        out["boundary_curves"].append(row)
    for name in TEST_LAWS:
        for b in SCALE_SLOPES:
            differences, rejection = quartile_approximation(k[name], N_TARGET, INPUTS, b)
            out["quartile_approximation"].append({"law": name, "scale_slope": b,
                                                  **{f"difference_{s}": float(v) for s, v in zip(SUFFIX, differences)},
                                                  "rejection_pct": float(rejection)})
        cross = crossover(k[name], N_TARGET, INPUTS)
        out["crossover"].append({"law": name, "scale_slope_cross": cross,
                                 "power": power(2 * INPUTS, float(k[name]["gamma"]) * N_TARGET * cross ** 2 / 3),
                                 "approximation_cost": float(0.75 ** 2 * k[name]["v"][2] ** 2 / 3),
                                 "variance_ratio": float(k[name]["c_composite"] / k[name]["c"][1])})
    for name, law in LAWS.items():
        for omega in OMEGAS:
            shift = population_shift(law, SINGLE_LEVEL, omega)
            out["single_quantile_population"].append({"law": name, "omega": omega, "shift": shift,
                                                      "squared_error": omega ** 2 / 3 + shift ** 2})
    return out


def metadata():
    return {"levels": LEVELS.tolist(), "alpha": ALPHA, "n": N_TARGET, "d": INPUTS, "dimensions": list(DIMENSIONS),
            "scale_slopes": list(SCALE_SLOPES), "least_squares_nu": list(LEAST_SQUARES_NU),
            "grid": {"start": 0.0, "stop": GRID_STOP, "step": GRID_STOP / GRID_STEPS, "points": GRID_STEPS + 1},
            "single_quantile": {"level": SINGLE_LEVEL, "omega": list(OMEGAS)},
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in tables().items():
        write_csv(args.out / f"{name}.csv", rows)
    (args.out / "metadata.json").write_text(json.dumps(metadata(), indent=2) + "\n")
    print(f"Wrote the closed-form tables and metadata.json to {args.out}")


if __name__ == "__main__":
    main()
