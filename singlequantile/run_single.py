"""Single-quantile study at the 0.90 level with known scores, 2,000 replications, with and without bounds.

Usage, from the repository root:
    python -m singlequantile.run_single --out DIR [--workers N] [--replicates R] [--config FILE]

Input: singlequantile/config/single.json. Each trial is singlequantile/trial.py on the functions of
singlequantile/grid/run_simulations.py, with the grid's seeds, so replications 0-199 of the
bounded variant are the grid's rows at n = 100. The bounded variant keeps the grid's coefficient
bounds and transfer ball; the unbounded variant removes both. Writes replications.csv.gz,
summary.csv, paired.csv, activity.csv, batches.csv and metadata.json to DIR.

The grid's fits ask HiGHS for one thread, and HiGHS keeps one thread pool per process, so the
trials always run in worker processes that run nothing else. Draws depend only on the seed and
the replication index, so the worker count does not change them.
"""
import argparse
import csv
import gzip
import importlib.util
import io
import json
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

from multiquantile.common import standard_error, write_csv

from . import trial as trial_module

HERE = Path(__file__).resolve().parent
GRID = HERE / "grid" / "run_simulations.py"
DEFAULT_CONFIG = HERE / "config" / "single.json"
METHODS = ("transfer", "target_only", "pooled")
GRID_MODULE = None


def load_grid():
    """Worker initializer: load the grid script as a module, once per process."""
    global GRID_MODULE
    spec = importlib.util.spec_from_file_location("grid_run_simulations", GRID)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    GRID_MODULE = module


def run_trial(task):
    variant, noise, replicate, coefficient_box, ball = task
    rows = trial_module.trial(GRID_MODULE, replicate, noise, coefficient_box, ball)
    return [{"variant": variant, **row} for row in rows]


def write_rows_gz(path, rows):
    """The same CSV text as write_csv, gzip-compressed without a file name or timestamp."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    with path.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as packed:
        packed.write(buffer.getvalue().encode())


def summarize(rows, cfg):
    """summary, paired, activity and batches from the replication rows."""
    cells = {}
    for row in rows:
        cells.setdefault((row["variant"], row["noise"], row["departure"]), []).append(row)
    summary, paired, activity, batches = [], [], [], []
    for (variant, noise, departure), group in cells.items():
        group = sorted(group, key=lambda r: r["replicate"])
        errors = {m: np.array([r[f"mse_{m}"] for r in group]) for m in METHODS}
        for method in METHODS:
            values = errors[method]
            summary.append({"variant": variant, "noise": noise, "departure": departure, "method": method,
                            "R": len(values), "mean": float(values.mean()), "mcse": float(standard_error(values)),
                            "median": float(np.median(values)), "maximum": float(values.max())})
        versus_target, versus_pooled = errors["transfer"] - errors["target_only"], errors["transfer"] - errors["pooled"]
        paired.append({"variant": variant, "noise": noise, "departure": departure, "R": len(group),
                       "mean_minus_target_only": float(versus_target.mean()),
                       "mcse_minus_target_only": float(standard_error(versus_target)),
                       "mean_minus_pooled": float(versus_pooled.mean()),
                       "mcse_minus_pooled": float(standard_error(versus_pooled))})
        if variant == "bounded":
            share = {k: 100 * float(np.mean([r[k] > 0 for r in group]))
                     for k in ("source_bound_active", "target_bound_active", "pooled_bound_active", "transfer_fallback")}
            activity.append({"noise": noise, "departure": departure, "R": len(group),
                             "source_pct": share["source_bound_active"], "target_pct": share["target_bound_active"],
                             "pooled_pct": share["pooled_bound_active"], "transfer_pct": share["transfer_fallback"]})
        size = cfg["batch_size"]
        for batch, start in enumerate(range(0, len(group), size)):
            part = versus_target[start:start + size]
            batches.append({"variant": variant, "noise": noise, "departure": departure, "batch": batch,
                            "first_replicate": group[start]["replicate"], "R": len(part), "mean": float(part.mean())})
    return summary, paired, activity, batches


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
    constants = {"seed": trial_module.SEED, "level": trial_module.TAU, "n_source": trial_module.N_SOURCE,
                 "n_target": trial_module.N_TARGET, "n_evaluation": trial_module.N_EVALUATION,
                 "departures": list(trial_module.DEPARTURES)}
    differing = [key for key, value in constants.items() if cfg[key] != value]
    if differing:
        parser.error(f"the configuration disagrees with singlequantile/trial.py: {differing}")
    if set(cfg["noise"]) - {"normal", "t3", "cauchy"}:
        parser.error("noise laws must be among the grid's laws normal, t3 and cauchy")
    tasks = [(variant, noise, replicate, settings["coefficient_box"], settings["ball"])
             for variant, settings in cfg["variants"].items() for noise in cfg["noise"]
             for replicate in range(cfg["replicates"])]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=load_grid) as pool:
        rows = [row for block in pool.map(run_trial, tasks, chunksize=4) for row in block]
    summary, paired, activity, batches = summarize(rows, cfg)
    args.out.mkdir(parents=True, exist_ok=True)
    write_rows_gz(args.out / "replications.csv.gz", rows)
    for name, records in (("summary", summary), ("paired", paired), ("activity", activity), ("batches", batches)):
        write_csv(args.out / f"{name}.csv", records)
    metadata = {"configuration": cfg, "workers": args.workers, "python": platform.python_version(),
                "numpy": np.__version__, "scipy": scipy.__version__}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Completed {len(tasks):,} trials ({cfg['replicates']} replications per variant and law); "
          f"{len(rows):,} rows in {args.out}")


if __name__ == "__main__":
    main()
