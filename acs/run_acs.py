"""ACS income study: transfer at the 0.90 level (Study A) and common versus separate quartile slopes (Study B).
Study A also fits the target records on the transfer inputs without the source prediction (inputs_only).

Usage, from the repository root:
    python -m acs.run_acs --out DIR [--workers N] [--replicates R] [--targets WY ...] [--config FILE]

Input: acs/config/acs.json and the committed extracts in acs/data/. The source fits use the 20,000
California records (the alternate source the 20,000 Texas records); each target state is split
into nested labeled samples of 100, 200, 400 and 800 records and a held-out set of the remaining
records, redrawn in every replication. Writes to DIR: metadata.json, solver_diagnostics.json,
descriptives.csv, topcode_shares.csv, benchmark.csv, fullsample_coefficients.csv,
fullsample_fits.csv, transfer_replications.csv.gz, transfer_summary.csv, transfer_paired.csv,
transfer_coefficients.csv, sensitivity_replications.csv, sensitivity_summary.csv,
sharing_replications.csv.gz, sharing_summary.csv, sharing_tests.csv, sharing_crosstab.csv,
inputs_only_replications.csv.gz, inputs_only_summary.csv and inputs_only_paired.csv.
acs/README.md describes every file.

Every random draw comes from SeedSequence([seed, stream, FIPS code, index]): stream 2 the splits,
3 the tie numbers of the records, 4 the cross-validation folds (stream 1, the source subsamples,
is used by acs/build_data.py). No result depends on the worker count or the order of completion,
and a run with R replications reproduces the first R replications of a longer run.
"""
import argparse
import csv
import gzip
import io
import json
import math
import multiprocessing
import platform
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy

from multiquantile.common import standard_error, write_csv
from multiquantile.estimation import slope_wald

from . import estimation
from .estimation import coverage, fit, fit_composite, mean_check_loss, score_map
from .variables import COVARIATES, DICTIONARY, STATES, covariates, describe, dictionary, income, log_income, read_extract

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config" / "acs.json"
METHODS = ("transfer", "target_only", "pooled", "source_only")
CONTRASTS = ("target_only", "pooled", "source_only")
TRANSFER_TERMS = ("intercept", "source", "score1", "score2", "school")
SHARING_TERMS = ("intercept", "source_median", "school")
SCORE_COMPONENTS = 4
MARGIN_THRESHOLD = 1e-6
SAMPLES = {}
CONFIG = {}


def rng(seed, stream, fips, index):
    return np.random.default_rng(np.random.SeedSequence([seed, stream, fips, index]))


def load_samples(cfg):
    """Read every extract the run needs, with its dictionary, covariates, responses and tie numbers."""
    samples = {}
    for state in [cfg["source"], cfg["alternate_source"], *cfg["targets"]]:
        frame = read_extract(state)
        fips = STATES[state][1]
        samples[state] = {"frame": frame, "fips": fips, "phi": dictionary(frame), "x": covariates(frame),
                          "level": income(frame), "log": log_income(frame),
                          "ties": rng(cfg["seed"], 3, fips, 0).random(len(frame))}
    return samples


def initialize(cfg):
    """Worker initializer: load the extracts once per process."""
    CONFIG.clear()
    CONFIG.update(cfg)
    SAMPLES.clear()
    SAMPLES.update(load_samples(cfg))


def scores_of(x, center, loadings, p):
    return (x - center) @ loadings / math.sqrt(p)


def transfer_design(source_prediction, scores, x, components, retained):
    """Columns 1, the source prediction, the first `components` scores, then the retained covariates."""
    columns = [np.ones(len(x)), source_prediction, *scores[:, :components].T]
    columns += [x[:, COVARIATES.index(name)] for name in retained]
    return np.column_stack(columns)


class Programs:
    """Collects the uniqueness margin and dropped columns of every linear program, with its identity."""

    def __init__(self):
        self.records = []

    def fit(self, identity, y, design, tau, ties):
        diagnostics = []
        beta, dropped = fit(y, design, tau, ties, diagnostics)
        self.records.append({**identity, "rows": diagnostics[0][0], "columns": diagnostics[0][1],
                             "margin": diagnostics[0][2], "dropped": dropped})
        return beta, dropped

    def composite(self, identity, y, slope_design, taus, ties):
        diagnostics = []
        intercepts, slopes = fit_composite(y, slope_design, taus, ties, diagnostics)
        self.records.append({**identity, "rows": diagnostics[0][0], "columns": diagnostics[0][1],
                             "margin": diagnostics[0][2], "dropped": 0})
        return intercepts, slopes


def identity(study, target, fit_name, source=None, scale=None, n=None, replicate=None):
    return {"study": study, "source": source, "scale": scale, "target": target, "n": n, "replicate": replicate,
            "fit": fit_name}


def wald_test(y, design, levels, separate):
    """slope_wald with the nominal levels; (nan, nan, True) if it raises ArithmeticError."""
    try:
        statistic, p_value = slope_wald(y, design, np.asarray(levels, dtype=float), separate)
        return float(statistic), float(p_value), False
    except ArithmeticError:
        return math.nan, math.nan, True


def replicate_task(target, b, coefficients):
    """Study A, the sensitivity and alternate-source fits (primary target only) and Study B for one split."""
    cfg, s = CONFIG, SAMPLES[target]
    seed, tau, tolerance = cfg["seed"], cfg["level"], cfg["coverage_tolerance"]
    n_records = len(s["level"])
    perm = rng(seed, 2, s["fips"], b).permutation(n_records)
    held = perm[max(cfg["labeled_sizes"]):]
    center, loadings, p = score_map(s["x"][held], SCORE_COMPONENTS)
    scores = scores_of(s["x"], center, loadings, p)
    programs = Programs()
    transfer_rows, sensitivity_rows, sharing_rows, failed = [], [], [], []
    y, y_held = s["level"], s["level"][held]
    # the transfer inputs without the source prediction, fitted with their own programs
    inputs_design = np.column_stack((np.ones(n_records), scores[:, :cfg["components"]],
                                     *(s["x"][:, COVARIATES.index(name)] for name in cfg["retained"])))
    inputs_programs, inputs_rows = Programs(), []
    for n in cfg["labeled_sizes"]:
        labeled = perm[:n]
        beta_i, dropped_i = inputs_programs.fit(identity("transfer", target, "inputs_only", n=n, replicate=b),
                                                y[labeled], inputs_design[labeled], tau, s["ties"][labeled])
        predicted = inputs_design[held] @ beta_i
        inputs_rows.append({"target": target, "n": n, "replicate": b, "loss": mean_check_loss(y_held, predicted, tau),
                            "coverage": coverage(y_held, predicted, tolerance), "coef_norm": float(np.linalg.norm(beta_i)),
                            "dropped_columns": dropped_i, "margin": inputs_programs.records[-1]["margin"],
                            "n_eval": len(held)})
    sources = [cfg["source"]] + ([cfg["alternate_source"]] if target == cfg["primary_target"] else [])
    target_only = {}
    for source in sources:
        src = SAMPLES[source]
        prediction = s["phi"] @ coefficients[(source, "level", tau)]
        design = transfer_design(prediction, scores, s["x"], cfg["components"], cfg["retained"])
        for n in cfg["labeled_sizes"]:
            labeled = perm[:n]
            key = dict(source=source, n=n, replicate=b)
            beta, dropped = programs.fit(identity("transfer", target, "transfer", **key),
                                         y[labeled], design[labeled], tau, s["ties"][labeled])
            fits = {"transfer": (design[held] @ beta, beta, dropped)}
            if n not in target_only:
                beta_t, dropped_t = programs.fit(identity("transfer", target, "target_only", **key),
                                                 y[labeled], s["phi"][labeled], tau, s["ties"][labeled])
                target_only[n] = (s["phi"][held] @ beta_t, beta_t, dropped_t)
            fits["target_only"] = target_only[n]
            beta_p, dropped_p = programs.fit(identity("transfer", target, "pooled", **key),
                                             np.r_[src["level"], y[labeled]], np.vstack((src["phi"], s["phi"][labeled])),
                                             tau, np.r_[src["ties"], s["ties"][labeled]])
            fits["pooled"] = (s["phi"][held] @ beta_p, beta_p, dropped_p)
            fits["source_only"] = (prediction[held], None, 0)
            for method in METHODS:
                predicted, beta_m, dropped_m = fits[method]
                coefs = dict(zip(TRANSFER_TERMS, beta_m)) if method == "transfer" else {}
                transfer_rows.append({
                    "source": source, "target": target, "n": n, "replicate": b, "method": method,
                    "loss": mean_check_loss(y_held, predicted, tau), "coverage": coverage(y_held, predicted, tolerance),
                    "coef_norm": float(np.linalg.norm(beta_m)) if beta_m is not None else math.nan,
                    **{f"coef_{term}": float(coefs.get(term, math.nan)) for term in TRANSFER_TERMS},
                    "dropped_columns": dropped_m, "n_eval": len(held)})
        if target == cfg["primary_target"] and source == cfg["source"]:
            for components in cfg["sensitivity_components"]:
                for retained in cfg["sensitivity_retained"]:
                    spec = transfer_design(prediction, scores, s["x"], components, retained)
                    label = "+".join(retained) if retained else "none"
                    for n in cfg["labeled_sizes"]:
                        labeled = perm[:n]
                        beta, _ = programs.fit(identity("sensitivity", target, f"r{components}/{label}", source=source,
                                                        n=n, replicate=b),
                                               y[labeled], spec[labeled], tau, s["ties"][labeled])
                        predicted = spec[held] @ beta
                        sensitivity_rows.append({"target": target, "n": n, "replicate": b, "components": components,
                                                 "retained": label, "loss": mean_check_loss(y_held, predicted, tau),
                                                 "coverage": coverage(y_held, predicted, tolerance)})
    taus = cfg["quartiles"]
    for scale in cfg["scales"]:
        response = s[scale]
        median = s["phi"] @ coefficients[(cfg["source"], scale, 0.5)]
        slopes_design = np.column_stack((median, s["x"][:, COVARIATES.index("school")]))
        design = np.column_stack((np.ones(n_records), slopes_design))
        for n in cfg["labeled_sizes"]:
            labeled = perm[:n]
            key = dict(scale=scale, n=n, replicate=b)
            separate = np.vstack([programs.fit(identity("sharing", target, f"separate/{level}", **key),
                                               response[labeled], design[labeled], level, s["ties"][labeled])[0]
                                  for level in taus])
            intercepts, slopes = programs.composite(identity("sharing", target, "composite", **key),
                                                    response[labeled], slopes_design[labeled], taus, s["ties"][labeled])
            wald, p_value, failure = wald_test(response[labeled], design[labeled], taus, separate)
            if failure:
                failed.append(identity("sharing", target, "slope_test", **key))
            for j, level in enumerate(taus):
                sharing_rows.append({
                    "scale": scale, "target": target, "n": n, "replicate": b, "tau": level,
                    "loss_composite": mean_check_loss(response[held], intercepts[j] + slopes_design[held] @ slopes, level),
                    "loss_separate": mean_check_loss(response[held], design[held] @ separate[j], level),
                    "slope_source_composite": float(slopes[0]), "slope_school_composite": float(slopes[1]),
                    "slope_source_separate": float(separate[j, 1]), "slope_school_separate": float(separate[j, 2]),
                    "wald": wald, "p_value": p_value})
    return {"transfer": transfer_rows, "sensitivity": sensitivity_rows, "sharing": sharing_rows,
            "inputs_only": inputs_rows, "programs": programs.records, "failed": failed}


def benchmark_task(target):
    """Ten-fold cross-validation of the dictionary fit at the 0.90 level on every record of a target."""
    cfg, s = CONFIG, SAMPLES[target]
    tau, n_records = cfg["level"], len(s["level"])
    folds = np.array_split(rng(cfg["seed"], 4, s["fips"], 0).permutation(n_records), cfg["folds"])
    programs = Programs()
    predicted = np.empty(n_records)
    for k, held in enumerate(folds):
        train = np.sort(np.concatenate([fold for j, fold in enumerate(folds) if j != k]))
        beta, _ = programs.fit(identity("benchmark", target, f"fold{k}", n=len(train)),
                               s["level"][train], s["phi"][train], tau, s["ties"][train])
        predicted[held] = s["phi"][held] @ beta
    row = {"target": target, "records": n_records, "L_ref": mean_check_loss(s["level"], predicted, tau),
           "coverage_ref": coverage(s["level"], predicted, cfg["coverage_tolerance"])}
    return {"benchmark": [row], "programs": programs.records, "failed": []}


def fullsample_task(target, coefficients, threshold):
    """Dictionary, transfer, separate and composite fits and the slope test on every record of a target."""
    cfg, s = CONFIG, SAMPLES[target]
    tau, tolerance, n_records = cfg["level"], cfg["coverage_tolerance"], len(s["level"])
    programs, coefficient_rows, fit_rows, failed = Programs(), [], [], []

    def keep(scale, name, level, terms, beta, predicted, response):
        coefficient_rows.extend({"target": target, "scale": scale, "fit": name, "tau": level, "term": term,
                                 "estimate": float(value)} for term, value in zip(terms, beta))
        fit_rows.append({"target": target, "scale": scale, "fit": name, "tau": level, "records": n_records,
                         "coverage": coverage(response, predicted, tolerance), "wald": math.nan, "p_value": math.nan})

    everyone = np.arange(n_records)
    beta, _ = programs.fit(identity("fullsample", target, "dictionary", scale="level", n=n_records),
                           s["level"], s["phi"], tau, s["ties"])
    keep("level", "dictionary", tau, DICTIONARY, beta, s["phi"] @ beta, s["level"])
    center, loadings, p = score_map(s["x"][everyone], cfg["components"])
    prediction = s["phi"] @ coefficients[(cfg["source"], "level", tau)]
    design = transfer_design(prediction, scores_of(s["x"], center, loadings, p), s["x"], cfg["components"], cfg["retained"])
    beta, _ = programs.fit(identity("fullsample", target, "transfer", source=cfg["source"], scale="level", n=n_records),
                           s["level"], design, tau, s["ties"])
    transfer_prediction = design @ beta
    keep("level", "transfer", tau, TRANSFER_TERMS, beta, transfer_prediction, s["level"])
    taus = cfg["quartiles"]
    for scale in cfg["scales"]:
        response = s[scale]
        median = s["phi"] @ coefficients[(cfg["source"], scale, 0.5)]
        slopes_design = np.column_stack((median, s["x"][:, COVARIATES.index("school")]))
        design = np.column_stack((np.ones(n_records), slopes_design))
        separate = []
        for level in taus:
            beta, _ = programs.fit(identity("fullsample", target, f"separate/{level}", scale=scale, n=n_records),
                                   response, design, level, s["ties"])
            separate.append(beta)
            keep(scale, "separate", level, SHARING_TERMS, beta, design @ beta, response)
        intercepts, slopes = programs.composite(identity("fullsample", target, "composite", scale=scale, n=n_records),
                                                response, slopes_design, taus, s["ties"])
        for j, level in enumerate(taus):
            keep(scale, "composite", level, SHARING_TERMS, np.r_[intercepts[j], slopes],
                 intercepts[j] + slopes_design @ slopes, response)
        wald, p_value, failure = wald_test(response, design, taus, np.vstack(separate))
        if failure:
            failed.append(identity("fullsample", target, "slope_test", scale=scale, n=n_records))
        fit_rows.append({"target": target, "scale": scale, "fit": "slope_test", "tau": math.nan, "records": n_records,
                         "coverage": math.nan, "wald": wald, "p_value": p_value})
    shares = {"sample": target, "wage_threshold": threshold / 1000,
              "share_wage_at_or_above": float(np.mean(s["frame"]["WAGP"].to_numpy() >= threshold)),
              "share_prediction_above": float(np.mean(transfer_prediction > threshold / 1000))}
    return {"coefficients": coefficient_rows, "fits": fit_rows, "topcodes": [shares], "programs": programs.records,
            "failed": failed}


def run_task(task):
    kind, arguments = task
    return {"replicate": replicate_task, "benchmark": benchmark_task, "fullsample": fullsample_task}[kind](*arguments)


def clean(rows):
    """Replace NaN by None, which csv writes as an empty cell."""
    return [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()} for row in rows]


def write_rows(path, rows):
    write_csv(path, clean(rows))


def write_rows_gz(path, rows):
    """The same CSV text as write_csv, gzip-compressed without a file name or timestamp."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(clean(rows))
    with path.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as packed:
        packed.write(buffer.getvalue().encode())


def mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(standard_error(values))


def summarize_transfer(rows, benchmark, cfg):
    """transfer_summary, transfer_paired and transfer_coefficients from the replication rows."""
    groups = {}
    for row in rows:
        groups.setdefault((row["source"], row["target"], row["n"]), {}).setdefault(row["method"], []).append(row)
    summary, paired, coefficient_rows = [], [], []
    for (source, target, n), methods in groups.items():
        losses = {}
        for method in METHODS:
            group = sorted(methods[method], key=lambda r: r["replicate"])
            losses[method] = np.array([r["loss"] for r in group])
            mean_loss, mcse_loss = mean_mcse(losses[method])
            mean_cov, mcse_cov = mean_mcse([r["coverage"] for r in group])
            summary.append({"source": source, "target": target, "n": n, "method": method, "R": len(group),
                            "mean_loss": mean_loss, "mcse_loss": mcse_loss, "mean_coverage": mean_cov,
                            "mcse_coverage": mcse_cov, "excess_pct": 100 * (mean_loss / benchmark[target] - 1)})
            if method == "transfer":
                for term in TRANSFER_TERMS:
                    values = np.array([r[f"coef_{term}"] for r in group])
                    coefficient_rows.append({"source": source, "target": target, "n": n, "term": term, "R": len(values),
                                             "mean": float(values.mean()), "sd": float(values.std(ddof=1))})
        for other in CONTRASTS:
            difference = losses["transfer"] - losses[other]
            mean, mcse = mean_mcse(difference)
            paired.append({"source": source, "target": target, "n": n, "contrast": f"transfer_minus_{other}",
                           "R": len(difference), "mean": mean, "mcse": mcse,
                           "share_negative": float(np.mean(difference < 0))})
    return summary, paired, coefficient_rows


def summarize_inputs_only(rows, transfer_rows, benchmark, cfg):
    """inputs_only_summary and inputs_only_paired: the fit on the transfer inputs without the source prediction."""
    loss = {(r["target"], r["n"], r["replicate"], r["method"]): r["loss"] for r in transfer_rows
            if r["source"] == cfg["source"]}
    groups = {}
    for row in rows:
        groups.setdefault((row["target"], row["n"]), []).append(row)
    summary, paired = [], []
    for (target, n), group in groups.items():
        group = sorted(group, key=lambda r: r["replicate"])
        own = np.array([r["loss"] for r in group])
        mean_loss, mcse_loss = mean_mcse(own)
        mean_cov, mcse_cov = mean_mcse([r["coverage"] for r in group])
        summary.append({"target": target, "n": n, "R": len(group), "mean_loss": mean_loss, "mcse_loss": mcse_loss,
                        "mean_coverage": mean_cov, "mcse_coverage": mcse_cov,
                        "excess_pct": 100 * (mean_loss / benchmark[target] - 1)})
        other = {m: np.array([loss[(target, n, r["replicate"], m)] for r in group]) for m in ("transfer", "target_only")}
        for contrast, difference in (("transfer_minus_inputs_only", other["transfer"] - own),
                                     ("inputs_only_minus_target_only", own - other["target_only"])):
            mean, mcse = mean_mcse(difference)
            paired.append({"source": cfg["source"], "target": target, "n": n, "contrast": contrast, "R": len(difference),
                           "mean": mean, "mcse": mcse, "share_negative": float(np.mean(difference < 0))})
    return summary, paired


def summarize_sensitivity(rows, transfer_rows, cfg):
    base = {(r["n"], r["replicate"]): r["loss"] for r in transfer_rows
            if r["source"] == cfg["source"] and r["target"] == cfg["primary_target"] and r["method"] == "target_only"}
    groups = {}
    for row in rows:
        groups.setdefault((row["target"], row["n"], row["components"], row["retained"]), []).append(row)
    summary = []
    for (target, n, components, retained), group in groups.items():
        group = sorted(group, key=lambda r: r["replicate"])
        loss = np.array([r["loss"] for r in group])
        difference = loss - np.array([base[n, r["replicate"]] for r in group])
        mean_loss, mcse_loss = mean_mcse(loss)
        mean_diff, mcse_diff = mean_mcse(difference)
        summary.append({"target": target, "n": n, "components": components, "retained": retained, "R": len(group),
                        "mean_loss": mean_loss, "mcse_loss": mcse_loss,
                        "mean_coverage": float(np.mean([r["coverage"] for r in group])),
                        "difference_from_target_only": mean_diff, "mcse_difference": mcse_diff})
    return summary


def summarize_sharing(rows, cfg):
    """sharing_summary, sharing_tests and sharing_crosstab from the replication rows."""
    cells, splits = {}, {}
    for row in rows:
        cells.setdefault((row["scale"], row["target"], row["n"], row["tau"]), []).append(row)
        splits.setdefault((row["scale"], row["target"], row["n"]), {}).setdefault(row["replicate"], {})[row["tau"]] = row
    summary = []
    for (scale, target, n, tau), group in cells.items():
        group = sorted(group, key=lambda r: r["replicate"])
        composite = np.array([r["loss_composite"] for r in group])
        separate = np.array([r["loss_separate"] for r in group])
        c_mean, c_mcse = mean_mcse(composite)
        s_mean, s_mcse = mean_mcse(separate)
        d_mean, d_mcse = mean_mcse(composite - separate)
        summary.append({"scale": scale, "target": target, "n": n, "tau": tau, "R": len(group),
                        "loss_composite": c_mean, "mcse_composite": c_mcse, "loss_separate": s_mean,
                        "mcse_separate": s_mcse, "paired_difference": d_mean, "paired_mcse": d_mcse})
    tests, crosstab = [], []
    low, high = cfg["quartiles"][0], cfg["quartiles"][-1]
    for (scale, target, n), by_replicate in splits.items():
        replicates = sorted(by_replicate)
        p_values = np.array([by_replicate[b][low]["p_value"] for b in replicates])
        valid = ~np.isnan(p_values)
        reject = p_values[valid] < cfg["alpha"]
        outer = np.array([sum(by_replicate[b][t]["loss_composite"] - by_replicate[b][t]["loss_separate"]
                              for t in (low, high)) > 0 for b in replicates])
        every = np.array([sum(by_replicate[b][t]["loss_composite"] for t in cfg["quartiles"])
                          > sum(by_replicate[b][t]["loss_separate"] for t in cfg["quartiles"]) for b in replicates])
        rate = float(reject.mean()) if valid.any() else math.nan
        tests.append({"scale": scale, "target": target, "n": n, "R": len(replicates), "rejections": int(reject.sum()),
                      "rate": rate, "mcse": math.sqrt(rate * (1 - rate) / valid.sum()) if valid.any() else math.nan,
                      "outer_worse": float(outer.mean()), "all_worse": float(every.mean())})
        for rejected in (0, 1):
            for worse in (0, 1):
                crosstab.append({"scale": scale, "target": target, "n": n, "reject": rejected, "outer_worse": worse,
                                 "count": int(np.sum((reject.astype(int) == rejected) & (outer[valid].astype(int) == worse)))})
    return summary, tests, crosstab


def summarize_programs(records, failed):
    margins = np.array([r["margin"] for r in records])
    smallest = int(np.argmin(margins))
    below = [{k: r[k] for k in ("study", "source", "scale", "target", "n", "replicate", "fit", "margin")}
             for r in records if r["margin"] < MARGIN_THRESHOLD]
    dropped = [r["dropped"] for r in records]
    by_study = {}
    for r in records:
        by_study.setdefault(r["study"], []).append(r["margin"])
    return {
        "level_offset": estimation.LEVEL_OFFSET,
        "zero_residual": estimation.ZERO_RESIDUAL,
        "linear_programs": len(records),
        "linear_programs_by_study": {k: len(v) for k, v in by_study.items()},
        "smallest_uniqueness_margin": float(margins.min()),
        "smallest_margin_by_study": {k: float(min(v)) for k, v in by_study.items()},
        "smallest_margin_program": {k: records[smallest][k] for k in ("study", "source", "scale", "target", "n",
                                                                       "replicate", "fit")},
        "margin_threshold": MARGIN_THRESHOLD,
        "programs_below_threshold": len(below),
        "programs_below_threshold_identity": below,
        "fits_with_dropped_columns": int(sum(d > 0 for d in dropped)),
        "largest_number_dropped": int(max(dropped)),
        "failed_tests": len(failed),
        "failed_test_identity": failed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--replicates", type=int, help="override the replication count for a smoke run")
    parser.add_argument("--targets", nargs="+", help="run only these targets, in the configuration's order")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.replicates is not None:
        cfg["replicates"] = args.replicates
    if args.targets is not None:
        unknown = set(args.targets) - set(cfg["targets"])
        if unknown:
            parser.error(f"unknown targets {sorted(unknown)}")
        cfg["targets"] = [t for t in cfg["targets"] if t in args.targets]
    if cfg["replicates"] < 2 or args.workers < 1:
        parser.error("use at least two replications and one worker")
    if cfg["level_offset"] != estimation.LEVEL_OFFSET:
        parser.error("level_offset in the configuration differs from acs.estimation.LEVEL_OFFSET")
    if (cfg["components"], cfg["retained"], cfg["quartiles"]) != (2, ["school"], [0.25, 0.5, 0.75]):
        parser.error("the output columns assume two scores, the retained covariate school and the three quartiles")
    samples = load_samples(cfg)
    source, alternate = cfg["source"], cfg["alternate_source"]
    for state in (source, alternate):
        if len(samples[state]["level"]) != cfg["n_source"]:
            parser.error(f"the {state} extract does not hold {cfg['n_source']} records")
    for target in cfg["targets"]:
        if len(samples[target]["level"]) < max(cfg["minimum_target_records"], max(cfg["labeled_sizes"]) + 1):
            parser.error(f"the {target} extract has too few records")
    programs = Programs()
    coefficients, source_fits = {}, []
    for state, scale, level in ((source, "level", cfg["level"]), (alternate, "level", cfg["level"]),
                                *((source, scale, 0.5) for scale in cfg["scales"])):
        s = samples[state]
        beta, _ = programs.fit(identity("source", None, f"dictionary/{level}", source=state, scale=scale,
                                        n=len(s[scale])), s[scale], s["phi"], level, s["ties"])
        coefficients[(state, scale, level)] = beta
        source_fits.append({"source": state, "scale": scale, "tau": level, "coefficients": [float(v) for v in beta]})
    thresholds = {row["STATEABRV"]: int(row["WAGTPCT"])
                  for row in csv.DictReader((HERE / "data" / "topcodes.csv").open(newline=""))}
    tasks = [("benchmark", (target,)) for target in cfg["targets"]]
    tasks += [("fullsample", (target, coefficients, thresholds[target])) for target in cfg["targets"]]
    tasks += [("replicate", (target, b, coefficients)) for target in cfg["targets"] for b in range(cfg["replicates"])]
    if args.workers == 1:
        initialize(cfg)
        results = [run_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=initialize, initargs=(cfg,)) as pool:
            results = list(pool.map(run_task, tasks, chunksize=1))
    collected = {}
    for result in results:
        for key, rows in result.items():
            collected.setdefault(key, []).extend(rows)
    records = programs.records + collected["programs"]
    rank = {"source": {v: i for i, v in enumerate((source, alternate))}, "target": {v: i for i, v in enumerate(cfg["targets"])},
            "method": {v: i for i, v in enumerate(METHODS)}, "scale": {v: i for i, v in enumerate(cfg["scales"])},
            "retained": {("+".join(v) if v else "none"): i for i, v in enumerate(cfg["sensitivity_retained"])}}
    collected["transfer"].sort(key=lambda r: (rank["source"][r["source"]], rank["target"][r["target"]], r["n"],
                                              r["replicate"], rank["method"][r["method"]]))
    collected["sensitivity"].sort(key=lambda r: (rank["target"][r["target"]], r["n"], r["replicate"], r["components"],
                                                 rank["retained"][r["retained"]]))
    collected["sharing"].sort(key=lambda r: (rank["scale"][r["scale"]], rank["target"][r["target"]], r["n"],
                                             r["replicate"], r["tau"]))
    collected["inputs_only"].sort(key=lambda r: (rank["target"][r["target"]], r["n"], r["replicate"]))
    benchmark = {row["target"]: row["L_ref"] for row in collected["benchmark"]}
    transfer_summary, transfer_paired, transfer_coefficients = summarize_transfer(collected["transfer"], benchmark, cfg)
    inputs_summary, inputs_paired = summarize_inputs_only(collected["inputs_only"], collected["transfer"], benchmark, cfg)
    sharing_summary, sharing_tests, sharing_crosstab = summarize_sharing(collected["sharing"], cfg)
    descriptives, topcodes = [], []
    for state in (source, alternate, *cfg["targets"]):
        descriptives.append({"sample": state, **describe(samples[state]["frame"], thresholds[state])})
    for state in (source, alternate):
        s = samples[state]
        prediction = s["phi"] @ coefficients[(state, "level", cfg["level"])]
        topcodes.append({"sample": state, "wage_threshold": thresholds[state] / 1000,
                         "share_wage_at_or_above": float(np.mean(s["frame"]["WAGP"].to_numpy() >= thresholds[state])),
                         "share_prediction_above": float(np.mean(prediction > thresholds[state] / 1000))})
    topcodes += collected["topcodes"]
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"configuration": cfg, "source_fits": source_fits, "dictionary_terms": list(DICTIONARY),
                "records": {state: len(samples[state]["level"]) for state in samples},
                "workers": args.workers, "python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__}
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.out / "solver_diagnostics.json").write_text(
        json.dumps(summarize_programs(records, collected["failed"]), indent=2) + "\n")
    for name, rows in (("descriptives", descriptives), ("topcode_shares", topcodes),
                       ("benchmark", collected["benchmark"]), ("fullsample_coefficients", collected["coefficients"]),
                       ("fullsample_fits", collected["fits"]), ("transfer_summary", transfer_summary),
                       ("transfer_paired", transfer_paired), ("transfer_coefficients", transfer_coefficients),
                       ("sharing_summary", sharing_summary), ("sharing_tests", sharing_tests),
                       ("sharing_crosstab", sharing_crosstab), ("inputs_only_summary", inputs_summary),
                       ("inputs_only_paired", inputs_paired)):
        write_rows(args.out / f"{name}.csv", rows)
    if collected["sensitivity"]:
        write_rows(args.out / "sensitivity_replications.csv", collected["sensitivity"])
        write_rows(args.out / "sensitivity_summary.csv", summarize_sensitivity(collected["sensitivity"],
                                                                                collected["transfer"], cfg))
    write_rows_gz(args.out / "transfer_replications.csv.gz", collected["transfer"])
    write_rows_gz(args.out / "sharing_replications.csv.gz", collected["sharing"])
    write_rows_gz(args.out / "inputs_only_replications.csv.gz", collected["inputs_only"])
    print(f"Completed {cfg['replicates']} replications of {len(cfg['targets'])} target(s): {len(records):,} linear "
          f"programs, smallest uniqueness margin {min(r['margin'] for r in records):.3g}, written to {args.out}")


if __name__ == "__main__":
    main()
