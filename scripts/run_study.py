"""Run studies into an explicit output directory.

Usage, from the repository root:
    python -m scripts.run_study STUDY [STUDY ...] --mode {smoke,full} --out DIR [--workers N]

STUDY is grid, single, quartiles, slope-test, closed-form, acs or all. Smoke mode runs each
study at a few replications: 2 grid, 2 single-quantile, 4 quartile and 20 slope-test
replications, and 2 ACS replications of the primary target only; the closed-form values are the
same in both modes. Each study runs in
DIR/<study>.partial, and is renamed to DIR/<study> only after its runner and its validator
succeed, so DIR/<study>/run.json exists only for a complete study. run.json records the commands,
settings, runtimes and environment. A partial directory left by an interrupted or failed run is
kept for inspection and replaced when the study runs again, from its beginning.

The grid script, singlequantile/grid/run_simulations.py, writes next to itself, so it is copied
into the study directory and every grid output there is produced by the run.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.environment import collect

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ("grid", "single", "quartiles", "slope-test", "closed-form", "acs")
GRID_SCRIPT = ROOT / "singlequantile" / "grid" / "run_simulations.py"
SMOKE_REPLICATES = {"quartiles": 4, "single": 2, "slope-test": 20, "acs": 2}
SMOKE_OPTIONS = {"acs": ("--targets", "WY")}
MODULES = {"single": ("singlequantile.run_single", "singlequantile.validate_single"),
           "slope-test": ("multiquantile.run_slope_test", "multiquantile.validate_slope_test"),
           "acs": ("acs.run_acs", "acs.validate_acs")}
THREADS = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "VECLIB_MAXIMUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def steps(study, mode, work, workers):
    """Commands for one study as (argv, cwd, log name); a None argv names files that must exist."""
    py = sys.executable
    if study == "grid":
        if mode == "smoke":
            return [([py, "run_simulations.py", "--smoke"], work, None),
                    (None, work, ("smoke_raw.csv",))]
        return [([py, "run_simulations.py", "--replicates", "200"], work, None),
                (None, work, ("raw_results.csv", "summary.csv", "paired_comparisons.csv"))]
    if study == "closed-form":
        return [([py, "-m", "closedform.run_closedform", "--out", str(work)], ROOT, None),
                ([py, "-m", "closedform.validate_closedform", "--results", str(work)], ROOT, None)]
    smoke = ["--replicates", str(SMOKE_REPLICATES[study]), *SMOKE_OPTIONS.get(study, ())] if mode == "smoke" else []
    if study in MODULES:
        runner, validator = MODULES[study]
        return [([py, "-m", runner, "--out", str(work), "--workers", str(workers), *smoke], ROOT, None),
                ([py, "-m", validator, "--results", str(work)], ROOT, None)]
    return [([py, "-m", "multiquantile.run_quartiles", "--out", str(work), "--workers", str(workers), *smoke], ROOT, None),
            ([py, "-m", "multiquantile.validate_quartiles", "--results", str(work)], ROOT, None)]


def relative(value):
    text = str(value)
    if text == sys.executable:
        return "python"
    try:
        return str(Path(text).resolve().relative_to(ROOT))
    except ValueError:
        return text


def execute(argv, cwd, log_path):
    """Run one command, echoing its output; keep a copy in log_path when given."""
    env = {**os.environ, **{k: os.environ.get(k, v) for k, v in THREADS.items()}}
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1)
    lines = []
    for line in process.stdout:
        sys.stdout.write(line)
        lines.append(line)
    if process.wait():
        raise SystemExit(f"command failed with exit status {process.returncode}: {' '.join(map(relative, argv))}")
    if log_path is not None:
        log_path.write_text("".join(lines))


def run_study(study, mode, out, workers):
    final = out / study
    work = out / f"{study}.partial"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    if study == "grid":
        shutil.copy2(GRID_SCRIPT, work / GRID_SCRIPT.name)
    started = time.time()
    record = {"study": study, "mode": mode, "workers": workers,
              "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "steps": []}
    print(f"== {study} ({mode})", flush=True)
    for argv, cwd, extra in steps(study, mode, work, workers):
        if argv is None:
            missing = [name for name in extra if not (cwd / name).is_file() or (cwd / name).stat().st_size == 0]
            if missing:
                raise SystemExit(f"{study}: expected output missing or empty: {', '.join(missing)}")
            continue
        began = time.time()
        execute(argv, cwd, cwd / extra if extra else None)
        record["steps"].append({"command": " ".join(relative(a) for a in argv), "cwd": relative(cwd),
                                "seconds": round(time.time() - began, 1)})
    record["seconds"] = round(time.time() - started, 1)
    if study == "grid":
        written = json.loads((work / "config.json").read_text())
        record["workers"] = written["workers"]
        record["note"] = "the grid script uses min(4, CPU count) worker processes regardless of --workers"
    record["environment"] = collect(record["workers"])
    (work / "run.json").write_text(json.dumps(record, indent=2) + "\n")
    shutil.rmtree(final, ignore_errors=True)
    work.rename(final)
    print(f"== {study} complete in {record['seconds']} s: {relative(final)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("studies", nargs="+", choices=STUDIES + ("all",))
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("use at least one worker")
    studies = STUDIES if "all" in args.studies else tuple(dict.fromkeys(args.studies))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for study in studies:
        run_study(study, args.mode, out, args.workers)


if __name__ == "__main__":
    main()
