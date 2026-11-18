"""Checks of the ACS study: the committed extracts, the fitting conventions, and a saved results directory.

Usage, from the repository root:
    python -m acs.validate_acs [--results DIR] [--design-only]

The design checks need only acs/data/: the extracts against their manifest and counts, value ranges,
the schooling map and the dictionary, fits on 61 records against an independent primal linear
program, and intercept-only fits against the left empirical quantile. The results checks confirm
complete and unique design cells in every replication file, finite values, the links between files
that must agree (the main sensitivity specification and the transfer rows; the Texas and California
target-only rows), every summary recomputed from the replication rows, the source fits, descriptive
statistics, top-code shares and full-sample coverage recomputed from the data, the solver
diagnostics, and the in-sample coverage of each full-sample dictionary fit. Raises AssertionError
on the first failure.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from multiquantile.estimation import check_loss, composite_design, left_empirical_quantile
from multiquantile.validate_quartiles import independent_primal
from scripts.tolerance import read_table, within

from .estimation import LEVEL_OFFSET, coverage, fit, fit_composite, row_levels, score_map
from .variables import (COLUMNS, COVARIATES, DATA, DICTIONARY, STATES, covariates, describe, dictionary,
                        extract_path, income, log_income, read_extract, schooling)

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE / "reference"
DEFAULT_CONFIG = HERE / "config" / "acs.json"
METHODS = ("transfer", "target_only", "pooled", "source_only")
TRANSFER_TERMS = ("intercept", "source", "score1", "score2", "school")
RANGES = {"SPORDER": (1, 99), "PWGTP": (1, 10 ** 5), "AGEP": (17, 99), "SCHL": (1, 24), "WKHP": (1, 99),
          "WKWN": (1, 52), "SEX": (1, 2), "COW": (1, 7), "MAR": (1, 5), "PINCP": (101, 10 ** 8),
          "WAGP": (0, 10 ** 8)}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def ties_of(cfg, state, n):
    fips = STATES[state][1]
    return np.random.default_rng(np.random.SeedSequence([cfg["seed"], 3, fips, 0])).random(n)


def verify_design(cfg=None):
    """Check the extracts, the variables and the fitting conventions. Needs only acs/data/."""
    cfg = cfg or json.loads(DEFAULT_CONFIG.read_text())
    require(cfg["level_offset"] == LEVEL_OFFSET, "level_offset differs from acs.estimation.LEVEL_OFFSET")
    manifest = json.loads((DATA / "manifest.json").read_text())
    counts = pd.read_csv(DATA / "sample_counts.csv")
    kept = dict(zip(counts.loc[counts.step == "kept", "state"], counts.loc[counts.step == "kept", "records"]))
    require(set(kept) == set(STATES) and set(manifest["extracts"]) == {extract_path(s).name for s in STATES},
            "sample_counts.csv and manifest.json must cover the ten states")
    for state in STATES:
        content = gzip.decompress(extract_path(state).read_bytes())
        entry = manifest["extracts"][extract_path(state).name]
        require(hashlib.sha256(content).hexdigest() == entry["sha256_uncompressed"], f"{state}: extract hash")
        frame = read_extract(state)
        require(len(frame) == kept[state] == entry["records"], f"{state}: record count")
        require(frame["SERIALNO"].str.len().gt(0).all(), f"{state}: empty SERIALNO")
        require(not frame.duplicated(["SERIALNO", "SPORDER"]).any(), f"{state}: duplicated person")
        order = frame.sort_values(["SERIALNO", "SPORDER"], kind="mergesort").index
        require(bool((order == frame.index).all()), f"{state}: extract is not sorted by SERIALNO and SPORDER")
        for column, (low, high) in RANGES.items():
            require(frame[column].between(low, high).all(), f"{state}: {column} outside [{low}, {high}]")
        require(frame["ADJINC"].nunique() == 1 and frame["ADJINC"].iloc[0] == manifest["adjinc"][state],
                f"{state}: ADJINC is not one value per file")
        phi = dictionary(frame)
        require(phi.shape[1] == len(DICTIONARY) == 16 and np.linalg.matrix_rank(phi) == 16,
                f"{state}: dictionary must have 16 columns of full rank")
        require(np.isfinite(phi).all() and np.isfinite(log_income(frame)).all(), f"{state}: non-finite variable")
    years = schooling(np.arange(1, 25))
    require(np.isfinite(years).all() and np.all(np.diff(years) >= 0), "schooling map must be defined and nondecreasing")
    primary = read_extract(cfg["primary_target"]).iloc[:61]
    y = income(primary)
    x = covariates(primary)
    ties = ties_of(cfg, cfg["primary_target"], len(read_extract(cfg["primary_target"])))[:61]
    design = np.column_stack((np.ones(61), x[:, 0], x[:, 1]))
    for tau in (0.25, 0.5, 0.9):
        diagnostics = []
        beta, dropped = fit(y, design, tau, ties, diagnostics)
        loss, optimum = check_loss(y - design @ beta, tau), independent_primal(y, design, tau)
        require(dropped == 0 and abs(loss - optimum) <= 1e-5 * optimum and diagnostics[0][2] > 0,
                f"fit at {tau} misses the independent primal optimum")
        perturbed = independent_primal(y, design, row_levels(tau, ties))
        np.testing.assert_allclose(check_loss(y - design @ beta, row_levels(tau, ties)), perturbed, rtol=1e-8)
    taus = np.array(cfg["quartiles"])
    diagnostics = []
    intercepts, slopes = fit_composite(y, design[:, 1:], taus, ties, diagnostics)
    stacked = composite_design(design[:, 1:], len(taus))
    response, nominal = np.tile(y, len(taus)), np.repeat(taus, 61)
    loss = check_loss(response - stacked @ np.r_[intercepts, slopes], nominal)
    optimum = independent_primal(response, stacked, nominal)
    require(abs(loss - optimum) <= 1e-5 * optimum and diagnostics[0][2] > 0,
            "composite fit misses the independent primal optimum")
    for target in cfg["targets"]:
        frame = read_extract(target)
        values = income(frame)[:200]
        u = ties_of(cfg, target, len(frame))[:200]
        for tau in (0.25, 0.5, 0.75, 0.9):
            beta, _ = fit(values, np.ones((200, 1)), tau, u)
            require(bool(within(beta[0], left_empirical_quantile(values, tau))),
                    f"{target}: intercept-only fit at {tau} is not the left empirical quantile")
    print("Validated the ten extracts, the variables, the fits against an independent primal LP, and the left "
          "empirical quantiles of intercept-only fits.")


def compare(label, recomputed, saved, keys):
    """Require the saved summary to match the recomputed one, row for row and column for column."""
    require(len(recomputed) == len(saved), f"{label}: {len(saved)} rows, expected {len(recomputed)}")
    merged = saved.merge(recomputed, on=keys, how="outer", suffixes=("", "_recomputed"), indicator=True)
    require(bool((merged["_merge"] == "both").all()), f"{label}: design cells differ")
    for column in recomputed.columns:
        if column in keys:
            continue
        a, b = merged[column].to_numpy(float), merged[column + "_recomputed"].to_numpy(float)
        require(bool(np.array_equal(np.isnan(a), np.isnan(b))), f"{label}: {column} missing pattern")
        np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], err_msg=f"{label}: {column}")


def mcse(values):
    return values.std(ddof=1) / np.sqrt(len(values))


def expected_programs(cfg):
    """Number of linear programs a run must report."""
    sizes, quartiles = len(cfg["labeled_sizes"]), len(cfg["quartiles"])
    specifications = len(cfg["sensitivity_components"]) * len(cfg["sensitivity_retained"])
    per_target = cfg["folds"] + 2 + len(cfg["scales"]) * (quartiles + 1)
    per_replicate = 3 * sizes + len(cfg["scales"]) * sizes * (quartiles + 1)
    primary = cfg["primary_target"] in cfg["targets"]
    extra = (2 * sizes + specifications * sizes) * cfg["replicates"] if primary else 0
    return 2 + len(cfg["scales"]) + len(cfg["targets"]) * (per_target + cfg["replicates"] * per_replicate) + extra


def verify_results(folder):
    """Validate one results directory written by run_acs. Raises AssertionError on failure."""
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text())
    cfg = metadata["configuration"]
    targets, replicates, sizes = cfg["targets"], cfg["replicates"], cfg["labeled_sizes"]
    primary = cfg["primary_target"] in targets
    tau, tolerance = cfg["level"], cfg["coverage_tolerance"]

    transfer = read_table(folder / "transfer_replications.csv.gz")
    expected = {(cfg["source"], t, n, b, m) for t in targets for n in sizes for b in range(replicates) for m in METHODS}
    if primary:
        expected |= {(cfg["alternate_source"], cfg["primary_target"], n, b, m) for n in sizes for b in range(replicates)
                     for m in METHODS}
    keys = list(zip(transfer.source, transfer.target, transfer.n, transfer.replicate, transfer.method))
    require(len(keys) == len(expected) and set(keys) == expected, "transfer_replications: missing or duplicated cells")
    require(np.isfinite(transfer[["loss", "coverage"]]).all().all() and transfer.loss.ge(0).all()
            and transfer.coverage.between(0, 1).all(), "transfer_replications: losses and coverage")
    is_transfer = transfer.method == "transfer"
    coefficient_columns = [f"coef_{term}" for term in TRANSFER_TERMS]
    require(np.isfinite(transfer.loc[is_transfer, coefficient_columns]).all().all()
            and transfer.loc[~is_transfer, coefficient_columns].isna().all().all(), "transfer coefficients")
    require(np.isfinite(transfer.loc[transfer.method != "source_only", "coef_norm"]).all()
            and transfer.loc[transfer.method == "source_only", "coef_norm"].isna().all(), "coefficient norms")
    for state in targets:
        n_eval = len(read_extract(state)) - max(sizes)
        require(transfer.loc[transfer.target == state, "n_eval"].eq(n_eval).all(), f"{state}: held-out size")
    if primary:
        own = transfer[(transfer.method == "target_only") & (transfer.target == cfg["primary_target"])]
        by_source = {s: g.drop(columns="source").set_index(["n", "replicate"]).sort_index() for s, g in own.groupby("source")}
        pd.testing.assert_frame_equal(by_source[cfg["source"]], by_source[cfg["alternate_source"]], check_exact=True)

    benchmark = read_table(folder / "benchmark.csv")
    require(list(benchmark.target) == targets and benchmark.L_ref.gt(0).all()
            and benchmark.coverage_ref.between(0, 1).all(), "benchmark.csv")
    reference_loss = dict(zip(benchmark.target, benchmark.L_ref))

    groups = transfer.groupby(["source", "target", "n", "method"], sort=False)
    summary = groups.agg(R=("loss", "size"), mean_loss=("loss", "mean"), mcse_loss=("loss", mcse),
                         mean_coverage=("coverage", "mean"), mcse_coverage=("coverage", mcse)).reset_index()
    summary["excess_pct"] = 100 * (summary.mean_loss / summary.target.map(reference_loss) - 1)
    compare("transfer_summary", summary, read_table(folder / "transfer_summary.csv"), ["source", "target", "n", "method"])
    wide = transfer.pivot(index=["source", "target", "n", "replicate"], columns="method", values="loss").reset_index()
    paired = []
    for other in ("target_only", "pooled", "source_only"):
        difference = wide.assign(d=wide.transfer - wide[other]).groupby(["source", "target", "n"], sort=False).d
        part = difference.agg(R="size", mean="mean", mcse=mcse, share_negative=lambda s: float(np.mean(s < 0))).reset_index()
        paired.append(part.assign(contrast=f"transfer_minus_{other}"))
    compare("transfer_paired", pd.concat(paired), read_table(folder / "transfer_paired.csv"),
            ["source", "target", "n", "contrast"])
    inputs = read_table(folder / "inputs_only_replications.csv.gz")
    inputs_expected = {(t, n, b) for t in targets for n in sizes for b in range(replicates)}
    inputs_keys = list(zip(inputs.target, inputs.n, inputs.replicate))
    require(len(inputs_keys) == len(inputs_expected) and set(inputs_keys) == inputs_expected,
            "inputs_only_replications: missing or duplicated cells")
    require(np.isfinite(inputs[["loss", "coverage", "coef_norm", "margin"]]).all().all() and inputs.loss.ge(0).all()
            and inputs.coverage.between(0, 1).all() and inputs.margin.gt(0).all(), "inputs_only_replications: values")
    for state in targets:
        n_eval = len(read_extract(state)) - max(sizes)
        require(inputs.loc[inputs.target == state, "n_eval"].eq(n_eval).all(), f"{state}: inputs_only held-out size")
    inputs_summary = inputs.groupby(["target", "n"], sort=False).agg(
        R=("loss", "size"), mean_loss=("loss", "mean"), mcse_loss=("loss", mcse), mean_coverage=("coverage", "mean"),
        mcse_coverage=("coverage", mcse)).reset_index()
    inputs_summary["excess_pct"] = 100 * (inputs_summary.mean_loss / inputs_summary.target.map(reference_loss) - 1)
    compare("inputs_only_summary", inputs_summary, read_table(folder / "inputs_only_summary.csv"), ["target", "n"])
    joined = wide[wide.source == cfg["source"]].merge(inputs[["target", "n", "replicate", "loss"]], on=["target", "n", "replicate"])
    require(len(joined) == len(inputs), "inputs_only rows join the transfer rows")
    inputs_paired = []
    for contrast, d in (("transfer_minus_inputs_only", joined.transfer - joined.loss),
                        ("inputs_only_minus_target_only", joined.loss - joined.target_only)):
        part = joined.assign(d=d).groupby(["source", "target", "n"], sort=False).d.agg(
            R="size", mean="mean", mcse=mcse, share_negative=lambda s: float(np.mean(s < 0))).reset_index()
        inputs_paired.append(part.assign(contrast=contrast))
    compare("inputs_only_paired", pd.concat(inputs_paired), read_table(folder / "inputs_only_paired.csv"),
            ["source", "target", "n", "contrast"])
    coefficient_rows = []
    for (source, target, n), group in transfer[is_transfer].groupby(["source", "target", "n"], sort=False):
        for term in TRANSFER_TERMS:
            values = group[f"coef_{term}"]
            coefficient_rows.append({"source": source, "target": target, "n": n, "term": term, "R": len(values),
                                     "mean": values.mean(), "sd": values.std(ddof=1)})
    compare("transfer_coefficients", pd.DataFrame(coefficient_rows), read_table(folder / "transfer_coefficients.csv"),
            ["source", "target", "n", "term"])

    if primary:
        sensitivity = read_table(folder / "sensitivity_replications.csv")
        labels = ["+".join(v) if v else "none" for v in cfg["sensitivity_retained"]]
        expected = {(cfg["primary_target"], n, b, r, j) for n in sizes for b in range(replicates)
                    for r in cfg["sensitivity_components"] for j in labels}
        keys = list(zip(sensitivity.target, sensitivity.n, sensitivity.replicate, sensitivity.components, sensitivity.retained))
        require(len(keys) == len(expected) and set(keys) == expected, "sensitivity_replications: cells")
        require(np.isfinite(sensitivity[["loss", "coverage"]]).all().all() and sensitivity.coverage.between(0, 1).all(),
                "sensitivity_replications: values")
        main = sensitivity[(sensitivity.components == cfg["components"]) & (sensitivity.retained == "+".join(cfg["retained"]))]
        mine = transfer[is_transfer & (transfer.source == cfg["source"]) & (transfer.target == cfg["primary_target"])]
        joined = main.merge(mine, on=["n", "replicate"], suffixes=("", "_transfer"))
        require(len(joined) == len(main) and (joined.loss == joined.loss_transfer).all()
                and (joined.coverage == joined.coverage_transfer).all(),
                "the main sensitivity specification differs from the transfer rows")
        base = transfer[(transfer.method == "target_only") & (transfer.source == cfg["source"])
                        & (transfer.target == cfg["primary_target"])][["n", "replicate", "loss"]]
        merged = sensitivity.merge(base, on=["n", "replicate"], suffixes=("", "_target_only"))
        merged["difference"] = merged.loss - merged.loss_target_only
        recomputed = merged.groupby(["target", "n", "components", "retained"], sort=False).agg(
            R=("loss", "size"), mean_loss=("loss", "mean"), mcse_loss=("loss", mcse), mean_coverage=("coverage", "mean"),
            difference_from_target_only=("difference", "mean"), mcse_difference=("difference", mcse)).reset_index()
        compare("sensitivity_summary", recomputed, read_table(folder / "sensitivity_summary.csv"),
                ["target", "n", "components", "retained"])

    sharing = read_table(folder / "sharing_replications.csv.gz")
    expected = {(s, t, n, b, q) for s in cfg["scales"] for t in targets for n in sizes for b in range(replicates)
                for q in cfg["quartiles"]}
    keys = list(zip(sharing.scale, sharing.target, sharing.n, sharing.replicate, sharing.tau))
    require(len(keys) == len(expected) and set(keys) == expected, "sharing_replications: cells")
    values = ["loss_composite", "loss_separate", "slope_source_composite", "slope_school_composite",
              "slope_source_separate", "slope_school_separate"]
    require(np.isfinite(sharing[values]).all().all() and sharing[["loss_composite", "loss_separate"]].ge(0).all().all(),
            "sharing_replications: values")
    require(sharing.p_value.dropna().between(0, 1).all(), "p-values outside [0, 1]")
    split_keys = ["scale", "target", "n", "replicate"]
    for column in ("slope_source_composite", "slope_school_composite", "wald", "p_value"):
        spread = sharing.groupby(split_keys)[column].agg(lambda s: s.nunique(dropna=False))
        require(bool((spread == 1).all()), f"{column} must repeat across the levels of a replication")
    cells = ["scale", "target", "n", "tau"]
    sharing["difference"] = sharing.loss_composite - sharing.loss_separate
    recomputed = sharing.groupby(cells, sort=False).agg(
        R=("loss_composite", "size"), loss_composite=("loss_composite", "mean"), mcse_composite=("loss_composite", mcse),
        loss_separate=("loss_separate", "mean"), mcse_separate=("loss_separate", mcse),
        paired_difference=("difference", "mean"), paired_mcse=("difference", mcse)).reset_index()
    compare("sharing_summary", recomputed, read_table(folder / "sharing_summary.csv"), cells)
    low, high = cfg["quartiles"][0], cfg["quartiles"][-1]
    per_split = sharing.groupby(split_keys, sort=False).apply(lambda g: pd.Series({
        "p_value": g.p_value.iloc[0],
        "outer": g[g.tau.isin([low, high])].difference.sum() > 0,
        "every": g.loss_composite.sum() > g.loss_separate.sum()}), include_groups=False).reset_index()
    tests, crosstab = [], []
    for (scale, target, n), group in per_split.groupby(["scale", "target", "n"], sort=False):
        valid = group[group.p_value.notna()]
        reject = valid.p_value < cfg["alpha"]
        rate = reject.mean() if len(valid) else math.nan
        tests.append({"scale": scale, "target": target, "n": n, "R": len(group), "rejections": int(reject.sum()),
                      "rate": rate, "mcse": math.sqrt(rate * (1 - rate) / len(valid)) if len(valid) else math.nan,
                      "outer_worse": group.outer.astype(float).mean(), "all_worse": group.every.astype(float).mean()})
        for rejected in (0, 1):
            for worse in (0, 1):
                crosstab.append({"scale": scale, "target": target, "n": n, "reject": rejected, "outer_worse": worse,
                                 "count": int(((reject.astype(int) == rejected) & (valid.outer.astype(int) == worse)).sum())})
    compare("sharing_tests", pd.DataFrame(tests), read_table(folder / "sharing_tests.csv"), ["scale", "target", "n"])
    compare("sharing_crosstab", pd.DataFrame(crosstab), read_table(folder / "sharing_crosstab.csv"),
            ["scale", "target", "n", "reject", "outer_worse"])

    samples = {state: read_extract(state) for state in [cfg["source"], cfg["alternate_source"], *targets]}
    thresholds = {row.STATEABRV: int(row.WAGTPCT) for row in pd.read_csv(DATA / "topcodes.csv").itertuples()}
    descriptives = pd.DataFrame([{"sample": s, **describe(frame, thresholds[s])} for s, frame in samples.items()])
    compare("descriptives", descriptives, read_table(folder / "descriptives.csv"), ["sample"])
    require(metadata["records"] == {s: len(f) for s, f in samples.items()}, "metadata records per sample")
    source_fits = {(f["source"], f["scale"], f["tau"]): np.array(f["coefficients"]) for f in metadata["source_fits"]}
    for (state, scale, level), saved in source_fits.items():
        frame = samples[state]
        response = income(frame) if scale == "level" else log_income(frame)
        refit, _ = fit(response, dictionary(frame), level, ties_of(cfg, state, len(frame)))
        require(bool(np.all(within(refit, saved))), f"source fit {state} {scale} {level} does not reproduce")
    coefficients = read_table(folder / "fullsample_coefficients.csv")
    fits = read_table(folder / "fullsample_fits.csv")
    shares = read_table(folder / "topcode_shares.csv").set_index("sample")
    for state in (cfg["source"], cfg["alternate_source"]):
        prediction = dictionary(samples[state]) @ source_fits[(state, "level", tau)]
        require(shares.loc[state, "share_prediction_above"] == np.mean(prediction > thresholds[state] / 1000),
                f"{state}: top-code share of the source prediction")
    for target in targets:
        frame = samples[target]
        x, phi, y = covariates(frame), dictionary(frame), {"level": income(frame), "log": log_income(frame)}
        mine = coefficients[coefficients.target == target]
        center, loadings, p = score_map(x, cfg["components"])
        scores = (x - center) @ loadings / np.sqrt(p)
        designs = {("level", "dictionary"): phi,
                   ("level", "transfer"): np.column_stack((np.ones(len(x)), phi @ source_fits[(cfg["source"], "level", tau)],
                                                           scores, x[:, COVARIATES.index("school")]))}
        for scale in cfg["scales"]:
            median = phi @ source_fits[(cfg["source"], scale, 0.5)]
            designs[(scale, "separate")] = designs[(scale, "composite")] = np.column_stack(
                (np.ones(len(x)), median, x[:, COVARIATES.index("school")]))
        for row in fits[(fits.target == target) & (fits.fit != "slope_test")].itertuples():
            beta = mine[(mine.scale == row.scale) & (mine.fit == row.fit) & (mine.tau == row.tau)].estimate.to_numpy()
            predicted = designs[(row.scale, row.fit)] @ beta
            np.testing.assert_allclose(row.coverage, coverage(y[row.scale], predicted, tolerance),
                                       err_msg=f"{target} {row.scale} {row.fit} {row.tau}: coverage")
            if row.fit == "transfer":
                require(shares.loc[target, "share_prediction_above"] == np.mean(predicted > thresholds[target] / 1000),
                        f"{target}: top-code share of the transfer prediction")
            if row.fit == "dictionary":
                require(0.89 <= row.coverage <= 0.91, f"{target}: full-sample dictionary coverage {row.coverage}")
        require(len(mine[mine.fit == "dictionary"]) == len(DICTIONARY), f"{target}: dictionary coefficients")
        tests = fits[(fits.target == target) & (fits.fit == "slope_test")]
        require(sorted(tests.scale) == sorted(cfg["scales"]) and tests.p_value.dropna().between(0, 1).all(),
                f"{target}: full-sample slope tests")
    for state, row in shares.iterrows():
        require(row.share_wage_at_or_above == np.mean(samples[state]["WAGP"].to_numpy() >= thresholds[state])
                and row.wage_threshold == thresholds[state] / 1000, f"{state}: top-code shares")

    diagnostics = json.loads((folder / "solver_diagnostics.json").read_text())
    require(diagnostics["smallest_uniqueness_margin"] > 0, "a fit is not the unique minimizer")
    require(diagnostics["linear_programs"] == expected_programs(cfg), "solver diagnostics do not cover every program")
    require(diagnostics["programs_below_threshold"] == len(diagnostics["programs_below_threshold_identity"]),
            "solver diagnostics: identities of small margins")
    require(diagnostics["failed_tests"] == int(sharing.groupby(split_keys).p_value.first().isna().sum())
            + int(fits.loc[fits.fit == "slope_test", "p_value"].isna().sum()), "solver diagnostics: failed tests")
    print(f"Validated {len(transfer):,} transfer rows, {len(inputs):,} rows without the source prediction, {len(sharing):,} sharing rows, "
          f"every summary, the source fits, "
          f"descriptive statistics and full-sample fits in {folder}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--design-only", action="store_true", help="check the extracts and conventions only")
    args = parser.parse_args()
    verify_design()
    if not args.design_only:
        verify_results(args.results)


if __name__ == "__main__":
    main()
