"""Closed-form values for the slope-sharing calculations: constants of the three quartiles and the
limiting power of the 5% slope test."""
import numpy as np
from scipy import integrate, optimize, stats

from multiquantile.weighting import BRIDGE, LAWS, LEVELS, covariance, simplex_optimum

ALPHA = 0.05
ADJACENT = np.diff(np.eye(len(LEVELS)), axis=0)


def constants(law):
    """Quantiles, densities, variance multipliers, the density-weighted mean quantile and Gamma."""
    v = law.ppf(LEVELS)
    f = law.pdf(v)
    omega = covariance(f)
    weights = simplex_optimum(omega)
    step = ADJACENT @ v
    return {"v": v, "f": f, "c": np.diag(omega).copy(), "c_composite": BRIDGE.sum() / f.sum() ** 2,
            "c_optimal": weights @ omega @ weights, "weights": weights, "v_bar": (f * v).sum() / f.sum(),
            "gamma": step @ np.linalg.solve(ADJACENT @ omega @ ADJACENT.T, step)}


def power(nu, noncentrality):
    """P{chi2_nu(noncentrality) > chi2_{nu, 1 - ALPHA}}; ALPHA at noncentrality 0."""
    if noncentrality <= 0:
        return ALPHA
    return float(stats.ncx2.sf(stats.chi2.ppf(1 - ALPHA, nu), nu, noncentrality))


def boundary_noncentrality(k, d, level=0):
    """Noncentrality at which sharing stops improving prediction at one level (default: an outer quartile)."""
    return d * k["gamma"] * (k["c"][level] - k["c_composite"]) / (k["v"][level] - k["v_bar"]) ** 2


def boundary_noncentrality_equal_weights(k, d):
    """The same for the prediction error averaged over the three levels with equal weights."""
    return d * k["gamma"] * (k["c"] - k["c_composite"]).sum() / ((k["v"] - k["v_bar"]) ** 2).sum()


def quartile_approximation(k, n, d, scale_slope):
    """Approximate differences (x 1000, composite minus separate) and rejection percentage."""
    departure = n * scale_slope ** 2 / 3
    differences = [(1000 / n) * ((k["c_composite"] - k["c"][j]) * d + (k["v"][j] - k["v_bar"]) ** 2 * departure)
                   for j in range(len(LEVELS))]
    return differences, 100 * power((len(LEVELS) - 1) * d, k["gamma"] * departure)


def crossover(k, n, d):
    """Scale slope at which the outer-quartile approximation changes sign."""
    return float(np.sqrt(3 * (k["c"][0] - k["c_composite"]) * d / (n * k["v"][2] ** 2)))


def population_shift(law, tau, omega):
    """Single-quantile design: the tau-quantile a of omega * G + error, with G uniform on [-1, 1] and the
    error centered at its tau-quantile and scaled to unit interquartile range. The population check-loss
    fit of the transfer class misses the target quantile by a - omega * G, with mean square
    omega**2 / 3 + a**2."""
    quantile, iqr = law.ppf(tau), law.ppf(0.75) - law.ppf(0.25)

    def excess(a):
        value, _ = integrate.quad(lambda g: law.cdf(quantile + iqr * (a - omega * g)) / 2, -1, 1)
        return value - tau

    return float(optimize.brentq(excess, -2, 2, xtol=1e-13))
