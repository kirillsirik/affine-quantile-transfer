"""Slope test of the quartile study alone, at 5,000 replications.

Usage, from the repository root:
    python -m multiquantile.run_slope_test --out DIR [--workers 4] [--replicates R] [--config FILE]

Input: multiquantile/config/slope_test.json, which shares every value except the replication count
with quartiles.json. Each replication draws the quartile study's target design (stream 3) and
innovations (streams 4 and 5), fits the three separate quartile regressions on the raw regressors
(1, H, B) and computes the Wald test of equal slopes. The quartile study uses (1, Ghat, B), a
nonsingular linear change of these regressors, so the first 600 statistics are the quartile
study's up to rounding. No source fit is needed. Writes statistics.csv, test_rates.csv and
metadata.json to DIR. Draws depend only on the seed and the replication index, so the worker count
does not change them, and a run with R replications reproduces the first R of a longer run.
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

from .common import write_csv
from .design import INNOVATION_STREAMS, LAWS, TARGET_LOCATION, raw_design, rng_at
from .estimation import quantile_lp, slope_wald

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "slope_test.json"


def run_replicate(r, *, cfg):
    """Wald statistic and p-value of replication r under every error law and scale slope."""
    levels = np.asarray(cfg["quantiles"])
    raw = raw_design(rng_at(cfg["seed"], 3, r), cfg["n_target"])
    rows = []
    for family in cfg["noise"]:
        innovation = LAWS[family].rvs(size=cfg["n_target"], random_state=rng_at(cfg["seed"], INNOVATION_STREAMS[family], r))
        for scale_slope in cfg["scale_slopes"]:
            y = raw @ TARGET_LOCATION + (1 + scale_slope * raw[:, 2]) * innovation
            separate = np.vstack([quantile_lp(y, raw, tau) for tau in levels])
            wald, p_value = slope_wald(y, raw, levels, separate)
            rows.append({"replicate": r, "noise": family, "scale_slope": scale_slope,
                         "wald": float(wald), "p_value": float(p_value)})
    return rows


def rates(rows, cfg):
    """Rejection counts and rates at level alpha, with the binomial plug-in standard error."""
    result = []
    for family in cfg["noise"]:
        for scale_slope in cfg["scale_slopes"]:
            rejected = np.array([row["p_value"] < cfg["alpha"] for row in rows
                                 if (row["noise"], row["scale_slope"]) == (family, scale_slope)])
            rate = rejected.mean()
            result.append({"noise": family, "scale_slope": scale_slope, "R": len(rejected),
                           "rejections": int(rejected.sum()), "rate": rate,
                           "mcse": np.sqrt(rate * (1 - rate) / len(rejected))})
    return result


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
    one = partial(run_replicate, cfg=cfg)
    ids = range(cfg["replicates"])
    if args.workers == 1:
        blocks = [one(r) for r in ids]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            blocks = list(pool.map(one, ids, chunksize=50))
    rows = [row for block in blocks for row in block]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "statistics.csv", rows)
    write_csv(args.out / "test_rates.csv", rates(rows, cfg))
    metadata = {"configuration": cfg, "workers": args.workers, "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {len(ids):,} replications per design; {len(rows):,} statistics in {args.out}")


if __name__ == "__main__":
    main()
