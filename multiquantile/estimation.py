"""Quantile-regression fits of the quartile study, the slope-test run and the ACS study.

Conventions
- Check loss: rho_tau(u) = max(tau * u, (tau - 1) * u), summed over observations.
- A fit solves the dual linear program  max y'v  subject to  D'v = 0  and  tau - 1 <= v_i <= tau
  with the HiGHS dual simplex. The coefficients are the negated equality multipliers. A fit is
  accepted only if its check loss equals the dual optimum within OBJECTIVE_RTOL * (1 + loss).
- Composite quantile regression stacks one copy of the data per level, with one intercept
  indicator per level block and slopes common to all levels; the dual box limits change by block.
- With the common slopes fixed, the composite loss separates by level, and at level tau every point
  between the order statistics of ranks ceil(n tau) and floor(n tau) + 1 of the residuals
  y - X b minimizes it. When n tau is an integer that is a whole interval, and the point the LP
  returns depends on the platform. fit_composite therefore keeps the LP slopes and sets each
  intercept to the left end, the left empirical quantile of the residuals.
"""
import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.stats import chi2, norm

OBJECTIVE_RTOL = 1e-7


def check_loss(residual, levels):
    """Summed check loss of residuals at a scalar level or at one level per residual."""
    levels = np.broadcast_to(levels, residual.shape)
    return np.maximum(levels * residual, (levels - 1) * residual).sum()


def solve_dual(y, design, levels):
    """Solve the quantile-regression dual LP.

    Returns the coefficients (negated equality multipliers), the dual optimum, which equals the
    minimal check loss, and the dual solution v.
    """
    levels = np.broadcast_to(levels, y.shape)
    result = linprog(-y, A_eq=design.T, b_eq=np.zeros(design.shape[1]),
                     bounds=np.column_stack((levels - 1, levels)), method="highs-ds")
    if not result.success:
        raise RuntimeError(result.message)
    return -result.eqlin.marginals, -result.fun, result.x


def quantile_lp(y, design, levels):
    """Quantile regression of y on the columns of design.

    levels is one level, or one level per row of a stacked composite design. Raises
    ArithmeticError if the check loss of the returned coefficients misses the dual optimum.
    """
    beta, optimum, _ = solve_dual(y, design, levels)
    loss = check_loss(y - design @ beta, levels)
    if abs(loss - optimum) > OBJECTIVE_RTOL * (1 + loss):
        raise ArithmeticError("Quantile primal and dual objectives disagree")
    return beta


def composite_design(slope_design, n_levels):
    """Stack n_levels copies of the slope regressors next to one intercept indicator per copy.

    slope_design must not contain a constant column. Row block j belongs to level j.
    """
    n = len(slope_design)
    return np.column_stack((np.repeat(np.eye(n_levels), n, axis=0), np.tile(slope_design, (n_levels, 1))))


def left_quantile_rank(n, level):
    """One-based rank ceil(n * level) of the left empirical quantile of n values.

    The level is read as the shortest decimal that represents it, so 0.28 is 28/100 and the
    product is exact. In floating point 25 * 0.28 is 7.000000000000001, which would give rank 8.
    """
    level = float(level)
    if not 0 < level < 1:
        raise ValueError(f"quantile level must lie strictly between 0 and 1, got {level}")
    if n < 1:
        raise ValueError("the left empirical quantile needs at least one value")
    return math.ceil(n * Fraction(repr(level)))


def left_empirical_quantile(values, level):
    """Smallest v among the values with #{values <= v} >= level * n: the order statistic of rank ceil(n * level).

    When n * level is an integer k, every point between the k-th and (k+1)-th order statistics
    minimizes the check loss and this is the left end; otherwise it is the unique minimizer.
    """
    ordered = np.sort(np.asarray(values, dtype=float).ravel())
    return ordered[left_quantile_rank(len(ordered), level) - 1]


@dataclass(frozen=True)
class CompositeFit:
    """Composite quantile regression with left-quantile intercepts.

    intercepts: one per level, in the order of the levels; slopes: common to all levels;
    loss: check loss of the selected fit; lp_loss: check loss at the point the LP returned;
    lp_optimum: dual optimum; lp_intercepts: the intercepts the LP returned;
    dual_residual: largest |D'v| of the dual solution.
    """
    intercepts: np.ndarray
    slopes: np.ndarray
    loss: float
    lp_loss: float
    lp_optimum: float
    lp_intercepts: np.ndarray
    dual_residual: float

    def coefficients(self, j):
        """Coefficients (intercept, slopes) at level j."""
        return np.r_[self.intercepts[j], self.slopes]

    @property
    def relative_gap(self):
        """|loss - dual optimum| / (1 + loss) at the selected intercepts."""
        return abs(self.loss - self.lp_optimum) / (1 + self.loss)

    @property
    def lp_relative_gap(self):
        return abs(self.lp_loss - self.lp_optimum) / (1 + self.lp_loss)

    def moved(self, rtol=1e-9):
        """Number of levels whose selected intercept differs from the one the LP returned."""
        return int(np.sum(np.abs(self.intercepts - self.lp_intercepts) > rtol * (1 + np.abs(self.intercepts))))


def fit_composite(y, slope_design, levels):
    """Composite QR of y with one free intercept per level and slopes common to all levels.

    y has n entries, slope_design is n by p without a constant column, and levels has m entries.
    The LP determines the slopes b. Each intercept is then the left empirical quantile of
    y - slope_design @ b at its level, so for increasing levels the intercepts are nondecreasing
    and they move with any shift of y. Both the LP point and the selected point must reach the
    dual optimum within OBJECTIVE_RTOL * (1 + loss), otherwise ArithmeticError is raised.
    """
    y = np.asarray(y, dtype=float)
    slope_design = np.asarray(slope_design, dtype=float)
    levels = np.asarray(levels, dtype=float)
    n, m = len(y), len(levels)
    design = composite_design(slope_design, m)
    response, per_row = np.tile(y, m), np.repeat(levels, n)
    beta, optimum, dual = solve_dual(response, design, per_row)
    lp_loss = check_loss(response - design @ beta, per_row)
    if abs(lp_loss - optimum) > OBJECTIVE_RTOL * (1 + lp_loss):
        raise ArithmeticError("Quantile primal and dual objectives disagree")
    slopes = beta[m:]
    residual = y - slope_design @ slopes
    intercepts = np.array([left_empirical_quantile(residual, tau) for tau in levels])
    loss = float(sum(check_loss(residual - b, tau) for b, tau in zip(intercepts, levels)))
    if abs(loss - optimum) > OBJECTIVE_RTOL * (1 + loss):
        raise ArithmeticError("The left-quantile intercepts do not attain the composite LP optimum")
    return CompositeFit(intercepts=intercepts, slopes=slopes, loss=loss, lp_loss=float(lp_loss),
                        lp_optimum=float(optimum), lp_intercepts=beta[:m].copy(),
                        dual_residual=float(np.max(np.abs(design.T @ dual))))


def summarize_fits(fits):
    """Solver diagnostics over a list of CompositeFit objects, for solver_diagnostics.json."""
    return {
        "composite_intercepts": "left empirical quantile of y - X b at each level, one-based rank ceil(n tau)",
        "objective_rtol": OBJECTIVE_RTOL,
        "composite_fits": len(fits),
        "level_fits": int(sum(len(f.intercepts) for f in fits)),
        "max_relative_gap_lp": max(f.lp_relative_gap for f in fits),
        "max_relative_gap_selected": max(f.relative_gap for f in fits),
        "max_dual_equality_residual": max(f.dual_residual for f in fits),
        "level_fits_where_lp_returned_another_intercept": int(sum(f.moved() for f in fits)),
    }


def slope_wald(y, design, levels, separate):
    """Wald test that all slopes are equal across levels, from separate fits.

    Affine conditional quantiles, a kernel estimate of each level's bread with bandwidth
    1.06 * IQR / 1.349 * n**(-1/5) of that level's residuals, and the nested-score meat
    (min(tau_j, tau_k) - tau_j tau_k) times the design second moment. Returns the statistic and
    its chi-squared p-value with (levels - 1) * (slopes) degrees of freedom.
    """
    n, p = design.shape
    m = len(levels)
    design_moment = design.T @ design / n
    inverse_breads = []
    for coef in separate:
        residual = y - design @ coef
        q25, q75 = np.quantile(residual, [0.25, 0.75])
        h = 1.06 * (q75 - q25) / 1.349 * n ** (-0.2)
        if h <= 0:
            raise ArithmeticError("Zero residual bandwidth")
        weights = norm.pdf(residual / h) / h
        bread = (design.T * weights) @ design / n
        inverse_breads.append(np.linalg.solve(bread, np.eye(p)))
    bridge = np.minimum.outer(levels, levels) - np.outer(levels, levels)
    covariance = np.block([
        [bridge[j, k] * inverse_breads[j] @ design_moment @ inverse_breads[k].T
         for k in range(m)] for j in range(m)
    ])
    adjacent = np.diff(np.eye(m), axis=0)
    contrast = np.kron(adjacent, np.eye(p)[1:, :])
    diff = contrast @ separate.ravel()
    contrast_cov = contrast @ covariance @ contrast.T
    if np.linalg.matrix_rank(contrast_cov) != len(diff):
        raise ArithmeticError("Singular Wald covariance")
    stat = n * diff @ np.linalg.solve(contrast_cov, diff)
    return stat, chi2.sf(stat, len(diff))
