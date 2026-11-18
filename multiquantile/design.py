"""Data-generating design of the quartile study and the slope-test run.

Covariates: V and B independent Uniform(-1, 1) and H = |V| - 1/2, so a raw row is (1, H, B).
The source median is 0.4 + 1.5 H + 0.2 B and the target location is 0.94 + 1.65 H + 0.82 B.
The target regressors are (1, Ghat, B), where Ghat is the frozen source fit without its intercept.

Every random draw comes from rng_at(seed, stream, replicate), so it depends only on the master
seed, a fixed stream number and the replication index, never on the worker count or order.
Streams: 1 source sample, 2 evaluation design, 3 quartile-study target designs, 4 and 5 the
quartile study's normal and t3 innovations.
"""
import numpy as np
from scipy.stats import norm, t

from .estimation import quantile_lp

SOURCE_MEDIAN = [0.4, 1.5, 0.2]
TARGET_LOCATION = [0.94, 1.65, 0.82]
LAWS = {"normal": norm, "t3": t(3)}
INNOVATION_STREAMS = {"normal": 4, "t3": 5}


def rng_at(seed, stream, replicate=0):
    """Generator for one (seed, stream, replicate) triple."""
    return np.random.default_rng(np.random.SeedSequence([seed, stream, replicate]))


def raw_design(rng, n):
    """n raw rows (1, H, B)."""
    v, b = rng.uniform(-1, 1, (2, n))
    return np.column_stack((np.ones(n), np.abs(v) - 0.5, b))


def target_features(raw, source):
    """Target regressors (1, Ghat, B) with Ghat = cH H + cB B from the source coefficients."""
    centered_anchor = raw[:, 1:] @ source[1:]
    return np.column_stack((raw[:, 0], centered_anchor, raw[:, 2]))


def fixed_inputs(cfg):
    """Fit the source median once and draw the evaluation design.

    Uses cfg["seed"], cfg["n_source"] and cfg["n_evaluation"]. Returns the source median
    coefficients on (1, H, B) and the raw evaluation rows; both stay fixed in every replication.
    """
    source_rng = rng_at(cfg["seed"], 1)
    source_raw = raw_design(source_rng, cfg["n_source"])
    source_y = source_raw @ SOURCE_MEDIAN + source_rng.standard_normal(cfg["n_source"])
    source = quantile_lp(source_y, source_raw, 0.5)
    if abs(source[1]) < 1e-6:
        raise ArithmeticError("The frozen source fit has a degenerate anchor")
    evaluation = raw_design(rng_at(cfg["seed"], 2), cfg["n_evaluation"])
    return source, evaluation


def target_quantile_coefficients(source, q, scale_slope):
    """True coefficients on (1, Ghat, B) of the target quantile location + (1 + scale_slope B) q."""
    return np.array([0.94 + q, 1.65 / source[1], 0.82 - 1.65 * source[2] / source[1] + scale_slope * q])
