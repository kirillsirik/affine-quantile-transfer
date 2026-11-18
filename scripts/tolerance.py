"""The comparison rule for reruns, fixed on 2026-11-06 before any study was rerun.

Floats agree when |a - b| <= max(RTOL * max(|a|, |b|), ATOL). Integers, labels and join keys must
match exactly, missing values must stay missing, and generated text must match exactly. ATOL is
the tolerance of the exact-seed reruns that validated the single-quantile grid when it was first run.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

RTOL = 1e-9
ATOL = 1e-10


def within(a, b):
    """Elementwise test of the float rule; NaN agrees only with NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    missing = np.isnan(a) | np.isnan(b)
    gap = np.abs(a - b)
    limit = np.maximum(RTOL * np.maximum(np.abs(a), np.abs(b)), ATOL)
    return np.where(missing, np.isnan(a) & np.isnan(b), gap <= limit)


def read_table(path, **options):
    """Read a CSV file with exact (round-trip) float parsing; pandas' default parser may be off by one ulp."""
    return pd.read_csv(path, float_precision="round_trip", **options)


def read_bytes(path):
    path = Path(path)
    return gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()


def compare_frames(new, old, key, subset=False):
    """Join two tables on key and compare every shared column.

    With subset=True the reference may hold extra rows (a smoke run covers fewer replications);
    every new row must still find its reference row. Returns (metrics, failures), where each
    metric is (column, kind, unequal, max_abs_gap, max_rel_gap, outside_tolerance).
    """
    failures = []
    if list(new.columns) != list(old.columns):
        failures.append(f"columns differ: {sorted(set(new.columns) ^ set(old.columns))}")
    for name, frame in (("new", new), ("reference", old)):
        if frame.duplicated(key).any():
            failures.append(f"duplicate keys in {name}")
    merged = new.merge(old, on=key, how="left" if subset else "outer", suffixes=("_new", "_old"), indicator=True)
    unmatched = int((merged["_merge"] != "both").sum())
    if unmatched:
        failures.append(f"{unmatched} keys unmatched")
    if not subset and len(new) != len(old):
        failures.append(f"rows {len(new)} vs {len(old)}")
    merged = merged[merged["_merge"] == "both"]
    metrics = []
    for column in [c for c in new.columns if c not in key and c in old.columns]:
        a, b = merged[column + "_new"], merged[column + "_old"]
        if pd.api.types.is_integer_dtype(a) and pd.api.types.is_integer_dtype(b):
            bad = int((a != b).sum())
            metrics.append((column, "int", bad, float((a - b).abs().max()) if len(a) else 0.0, float("nan"), bad))
            if bad:
                failures.append(f"{column}: {bad} integer mismatches")
            continue
        if not (pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b)):
            bad = int((a.astype(str) != b.astype(str)).sum())
            metrics.append((column, "label", bad, float("nan"), float("nan"), bad))
            if bad:
                failures.append(f"{column}: {bad} label mismatches")
            continue
        x, y = a.to_numpy(float), b.to_numpy(float)
        missing_x, missing_y = np.isnan(x), np.isnan(y)
        if (missing_x != missing_y).any():
            failures.append(f"{column}: missing pattern differs")
        keep = ~(missing_x | missing_y)
        gap = np.abs(x[keep] - y[keep])
        scale = np.maximum(np.abs(x[keep]), np.abs(y[keep]))
        relative = np.divide(gap, scale, out=np.zeros_like(gap), where=scale > 0)
        over = int((~within(x[keep], y[keep])).sum())
        metrics.append((column, "float", int((gap > 0).sum()), float(gap.max(initial=0)), float(relative.max(initial=0)), over))
        if over:
            failures.append(f"{column}: {over} values outside tolerance")
    return metrics, failures


def compare_tables(label, new_path, old_path, key, subset=False, reference_filter=None):
    """Compare two CSV files (the reference may be gzipped). Returns a result dict."""
    new = read_table(new_path)
    old = read_table(old_path)
    if reference_filter is not None:
        old = reference_filter(old)
    metrics, failures = compare_frames(new, old, key, subset)
    bytes_equal = None if subset else read_bytes(new_path) == read_bytes(old_path)
    return {"label": label, "rows": len(new), "bytes_equal": bytes_equal, "metrics": metrics, "failures": failures}


def json_differences(new, old, ignore=("python", "workers"), path=""):
    """Differences between two JSON documents under the float rule, skipping the ignored keys."""
    if isinstance(new, dict) and isinstance(old, dict):
        out = [] if new.keys() == old.keys() else [f"{path}: keys differ {sorted(set(new) ^ set(old))}"]
        for key in new.keys() & old.keys():
            if key not in ignore:
                out += json_differences(new[key], old[key], ignore, f"{path}/{key}")
        return out
    if isinstance(new, list) and isinstance(old, list) and len(new) == len(old):
        return [d for i, (a, b) in enumerate(zip(new, old)) for d in json_differences(a, b, ignore, f"{path}/{i}")]
    if isinstance(new, float) and isinstance(old, float):
        return [] if bool(within(new, old)) else [f"{path}: {new} vs {old}"]
    return [] if new == old else [f"{path}: {new} vs {old}"]
