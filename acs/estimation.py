"""Quantile-regression fits for the ACS study.

Conventions
- Every fit goes through multiquantile.estimation.quantile_lp (dual linear program, HiGHS dual simplex,
  accepted only if the check loss equals the dual optimum within OBJECTIVE_RTOL * (1 + loss)).
- Ties. With discrete covariates and incomes reported at round values, the minimizer of the check loss
  is often not unique, and the vertex a simplex solver returns then depends on the platform. Each record
  therefore carries a fixed number u in [0, 1), and a fit at nominal level tau solves the program with
  the row levels tau - LEVEL_OFFSET * (1 + u_i). For an intercept-only fit this selects the left
  empirical quantile, the rule the simulation studies use for composite intercepts. Losses, coverage
  and the slope test use the nominal level.
- Columns. A design can lose rank in a small sample (an indicator that is zero for every labeled
  record). The fit keeps the leftmost linearly independent columns and sets the other coefficients to 0.
"""
import numpy as np
from scipy.optimize import linprog

from multiquantile.estimation import check_loss, composite_design, quantile_lp

LEVEL_OFFSET = 1e-5
ZERO_RESIDUAL = 1e-7


def row_levels(tau, ties):
    """Row levels tau - LEVEL_OFFSET * (1 + u_i) for tie numbers u_i in [0, 1)."""
    return tau - LEVEL_OFFSET * (1.0 + np.asarray(ties, dtype=float))


def independent_columns(design):
    """Boolean mask of the leftmost linearly independent columns of design."""
    keep = np.zeros(design.shape[1], dtype=bool)
    rank = 0
    for j in range(design.shape[1]):
        keep[j] = True
        new_rank = np.linalg.matrix_rank(design[:, keep])
        if new_rank > rank:
            rank = new_rank
        else:
            keep[j] = False
    return keep


def uniqueness_margin(y, design, beta, levels):
    """Largest t for which duals u_i in [level_i - 1 + t, level_i - t] of the zero-residual rows balance
    the other rows. The fit is the unique minimizer exactly when t > 0 and the zero-residual rows span
    the columns; returns -1.0 if they do not span."""
    residual = y - design @ beta
    zero = np.abs(residual) <= ZERO_RESIDUAL * (1.0 + np.abs(y))
    above, below = (residual > 0) & ~zero, (residual < 0) & ~zero
    target = -(levels[above] @ design[above] + (levels[below] - 1.0) @ design[below])
    rows = design[zero]
    k, p = rows.shape
    if k < p or np.linalg.matrix_rank(rows) < p:
        return -1.0
    cost = np.r_[np.zeros(k), -1.0]
    equality = np.column_stack((rows.T, np.zeros(p)))
    upper = np.vstack((np.column_stack((np.eye(k), np.ones(k))), np.column_stack((-np.eye(k), np.ones(k)))))
    limit = np.r_[levels[zero], 1.0 - levels[zero]]
    result = linprog(cost, A_ub=upper, b_ub=limit, A_eq=equality, b_eq=target,
                     bounds=[(None, None)] * k + [(None, 0.5)], method="highs")
    if result.status != 0:
        raise ArithmeticError("uniqueness certificate failed: " + result.message)
    return float(result.x[-1])


def fit(y, design, tau, ties, diagnostics=None):
    """Quantile regression of y on the columns of design at nominal level tau.

    Returns the coefficient vector (zeros for dropped columns) and the number of dropped columns.
    diagnostics, if given, is a list that receives (rows, columns, uniqueness margin).
    """
    y = np.asarray(y, dtype=float)
    design = np.asarray(design, dtype=float)
    levels = row_levels(tau, ties)
    keep = np.ones(design.shape[1], dtype=bool)
    if np.linalg.matrix_rank(design) < design.shape[1]:
        keep = independent_columns(design)
    used = design[:, keep]
    beta = np.zeros(design.shape[1])
    beta[keep] = quantile_lp(y, used, levels)
    if diagnostics is not None:
        diagnostics.append((len(y), int(keep.sum()), uniqueness_margin(y, used, beta[keep], levels)))
    return beta, int((~keep).sum())


def fit_composite(y, slope_design, taus, ties, diagnostics=None):
    """Composite fit: one intercept per level and slopes common to all levels.

    slope_design has no constant column. Returns (intercepts, slopes). The stacked program uses the
    row levels of every block, so its minimizer is unique and no intercept rule is needed afterwards.
    """
    y = np.asarray(y, dtype=float)
    slope_design = np.asarray(slope_design, dtype=float)
    taus = np.asarray(taus, dtype=float)
    n, m = len(y), len(taus)
    design = composite_design(slope_design, m)
    response = np.tile(y, m)
    levels = np.repeat(taus, n) - LEVEL_OFFSET * (1.0 + np.tile(np.asarray(ties, dtype=float), m))
    beta = quantile_lp(response, design, levels)
    if diagnostics is not None:
        diagnostics.append((len(response), design.shape[1], uniqueness_margin(response, design, beta, levels)))
    return beta[:m], beta[m:]


def mean_check_loss(y, prediction, tau):
    """Mean check loss at the nominal level."""
    return float(check_loss(np.asarray(y, dtype=float) - prediction, tau) / len(y))


def coverage(y, prediction, tolerance):
    """Share of responses at or below the prediction; a response within tolerance above counts as at."""
    return float(np.mean(np.asarray(y, dtype=float) <= prediction + tolerance))


def score_map(covariates, components):
    """Center, loadings and scale of principal-component scores from unlabeled covariates.

    Returns (mean, loadings, p): the scores of x are (x - mean) @ loadings / sqrt(p). The covariance uses
    divisor n. Each loading vector is signed so that its entry of largest absolute value is positive.
    """
    covariates = np.asarray(covariates, dtype=float)
    center = covariates.mean(axis=0)
    centered = covariates - center
    values, vectors = np.linalg.eigh(centered.T @ centered / len(covariates))
    loadings = vectors[:, ::-1][:, :components].copy()
    for j in range(components):
        if loadings[np.argmax(np.abs(loadings[:, j])), j] < 0:
            loadings[:, j] *= -1.0
    return center, loadings, covariates.shape[1]
