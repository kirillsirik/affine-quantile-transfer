"""Weights for combining separate quartile slope estimates.

Under a location model the separate slope estimates at levels tau_j have joint asymptotic
covariance proportional to (min(tau_j, tau_k) - tau_j tau_k) / (f_j f_k), where f_j is the error
density at its tau_j quantile. A weight vector w on the simplex has variance multiplier w' C w.
"""
import itertools

import numpy as np
from scipy.stats import cauchy, norm, t

LEVELS = np.array([0.25, 0.50, 0.75])
BRIDGE = np.minimum.outer(LEVELS, LEVELS) - np.outer(LEVELS, LEVELS)
LAWS = {"normal": norm, "t3": t(3), "cauchy": cauchy}


def covariance(densities):
    """Common scalar part C of the joint covariance of the three separate slope estimates."""
    if np.any(densities <= 0) or not np.all(np.isfinite(densities)):
        raise ArithmeticError("Quantile densities must be positive and finite")
    return BRIDGE / np.outer(densities, densities)


def affine_optimum(matrix):
    """Minimize w' C w subject to sum(w) = 1, with signed weights."""
    direction = np.linalg.solve(matrix, np.ones(len(matrix)))
    return direction / direction.sum()


def simplex_optimum(matrix):
    """Minimize w' C w over the simplex by checking the interior optimum of every face.

    A strictly convex quadratic on the three-weight simplex has seven candidate faces.
    """
    best_value, best = np.inf, None
    for size in range(1, len(matrix) + 1):
        for subset in itertools.combinations(range(len(matrix)), size):
            idx = np.array(subset)
            interior = affine_optimum(matrix[np.ix_(idx, idx)])
            if interior.min() < -1e-12:
                continue
            weights = np.zeros(len(matrix))
            weights[idx] = np.maximum(interior, 0)
            weights /= weights.sum()
            value = weights @ matrix @ weights
            if value < best_value:
                best_value, best = value, weights
    return best


def analytic_rows():
    """Population variance multipliers and weights for normal, t3 and Cauchy errors."""
    result = []
    for family, law in LAWS.items():
        densities = law.pdf(law.ppf(LEVELS))
        matrix = covariance(densities)
        methods = dict(zip(("qr_25", "qr_50", "qr_75"), np.eye(3)))
        methods.update(equal_cqr=densities/densities.sum(), signed=affine_optimum(matrix),
                       nonnegative=simplex_optimum(matrix))
        for method, weights in methods.items():
            result.append({"noise": family, "method": method,
                           "weight_25": weights[0], "weight_50": weights[1], "weight_75": weights[2],
                           "variance_multiplier": weights @ matrix @ weights})
    return result


def residual_densities(y, design, separate):
    """Estimate f(F^{-1}(tau)) at zero in each separate fit's residual distribution.

    Gaussian kernel with bandwidth 1.06 * IQR / (1.349 * n**0.2) of that fit's residuals.
    """
    residual = y[:, None] - design @ separate.T
    lower, upper = np.quantile(residual, [0.25, 0.75], axis=0)
    bandwidth = (1.06 / 1.349) * (upper - lower) * len(y) ** (-0.2)
    if np.any(bandwidth <= 0):
        raise ArithmeticError("Nonpositive residual bandwidth")
    return norm.pdf(residual / bandwidth).mean(axis=0) / bandwidth
