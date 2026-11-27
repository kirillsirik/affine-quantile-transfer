"""Left empirical quantile intercepts for composite quantile regression.

Tolerances. A selected fit must reach the LP optimum within OBJECTIVE_RTOL * (1 + loss), the
acceptance rule every fit already uses. Slopes from the dual simplex and from the independent
interior-point primal must agree within the same 1e-7, since both solvers stop at feasibility and
optimality tolerances of that order. Shifts of y change the LP data only by rounding, so slopes
and intercepts must move with the shift up to the comparison rule max(1e-9 * scale, 1e-10), many
orders above double-precision error and far below any reported digit.
"""
import unittest

import numpy as np

from multiquantile.estimation import (OBJECTIVE_RTOL, check_loss, fit_composite, left_empirical_quantile,
                                      left_quantile_rank, quantile_lp)
from multiquantile.validate_quartiles import independent_primal_solution
from scripts.tolerance import within

QUARTILES = np.array([0.25, 0.5, 0.75])


def composite_loss(y, slope_design, intercepts, slopes, levels):
    residual = y - slope_design @ slopes
    return sum(check_loss(residual - b, tau) for b, tau in zip(intercepts, levels))


def sample(n, seed=11, p=2):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n, p))
    return x @ np.linspace(1.0, -0.5, p) + 0.3 + rng.standard_t(3, n), x


class LeftEmpiricalQuantileTest(unittest.TestCase):
    def test_interval_of_minimizers_returns_its_left_end(self):
        values = np.array([0.0, 2.0, 4.0, 8.0])
        self.assertEqual(left_empirical_quantile(values, 0.5), 2.0)
        losses = {b: check_loss(values - b, 0.5) for b in (1.9, 2.0, 3.0, 4.0, 4.1)}
        self.assertEqual(losses[2.0], losses[3.0])
        self.assertEqual(losses[2.0], losses[4.0])
        self.assertGreater(losses[1.9], losses[2.0])
        self.assertGreater(losses[4.1], losses[4.0])

    def test_composite_fit_returns_two_on_the_same_interval(self):
        y = np.array([0.0, 2.0, 4.0, 8.0])
        fit = fit_composite(y, np.empty((4, 0)), [0.5])
        self.assertEqual(fit.intercepts[0], 2.0)
        self.assertLessEqual(fit.relative_gap, OBJECTIVE_RTOL)
        self.assertTrue(2.0 - 1e-9 <= fit.lp_intercepts[0] <= 4.0 + 1e-9)

    def test_integer_and_noninteger_n_tau(self):
        values = np.array([5.0, -1.0, 3.0, 9.0, 0.5, 7.0, 2.0, 4.0])
        ordered = np.sort(values)
        for level, rank in ((0.25, 2), (0.3, 3), (0.5, 4), (0.51, 5), (0.9, 8), (0.125, 1)):
            self.assertEqual(left_quantile_rank(len(values), level), rank)
            self.assertEqual(left_empirical_quantile(values, level), ordered[rank - 1])
        grid = np.linspace(-2, 10, 12001)
        for level in (0.25, 0.3, 0.5, 0.9):
            losses = np.array([check_loss(values - b, level) for b in grid])
            first_minimizer = grid[np.flatnonzero(losses <= losses.min() + 1e-12)[0]]
            self.assertAlmostEqual(first_minimizer, left_empirical_quantile(values, level), places=9)

    def test_rank_uses_exact_decimal_levels(self):
        self.assertEqual(repr(25 * 0.28), "7.000000000000001")
        self.assertEqual(left_quantile_rank(25, 0.28), 7)
        self.assertEqual(left_quantile_rank(300, 0.75), 225)
        values = np.random.default_rng(3).standard_normal(300)
        for level in QUARTILES:
            self.assertEqual(left_empirical_quantile(values, level), np.quantile(values, level, method="inverted_cdf"))

    def test_repeated_and_all_equal_residuals(self):
        repeated = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 5.0])
        self.assertEqual(left_empirical_quantile(repeated, 0.5), 1.0)
        self.assertEqual(left_empirical_quantile(repeated, 0.6), 2.0)
        self.assertEqual(left_empirical_quantile(repeated, 0.9), 5.0)
        constant = np.full(6, 3.0)
        for level in (0.01, 0.25, 0.5, 0.99):
            self.assertEqual(left_empirical_quantile(constant, level), 3.0)

    def test_invalid_input(self):
        for level in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                left_empirical_quantile([1.0, 2.0], level)
        with self.assertRaises(ValueError):
            left_empirical_quantile([], 0.5)


class CompositeFitTest(unittest.TestCase):
    def test_left_ends_reach_the_optimum_of_an_independent_primal(self):
        for n in (40, 41):  # n tau is an integer at every quartile for n = 40, at none for n = 41
            y, x = sample(n)
            fit = fit_composite(y, x, QUARTILES)
            residual = y - x @ fit.slopes
            for j, tau in enumerate(QUARTILES):
                self.assertEqual(fit.intercepts[j], left_empirical_quantile(residual, tau))
            stacked = np.column_stack((np.repeat(np.eye(3), n, axis=0), np.tile(x, (3, 1))))
            optimum, coefficients = independent_primal_solution(np.tile(y, 3), stacked, np.repeat(QUARTILES, n))
            self.assertLessEqual(abs(fit.loss - optimum), OBJECTIVE_RTOL * (1 + fit.loss))
            self.assertLessEqual(fit.relative_gap, OBJECTIVE_RTOL)
            np.testing.assert_allclose(fit.slopes, coefficients[3:], rtol=0, atol=1e-7)

    def test_each_intercept_is_the_left_end_of_its_minimizing_interval(self):
        # With 40 tau = k, the minimizers at fixed slopes are [r_(k), r_(k+1)]; at a vertex of the
        # LP two residuals can coincide and shrink that interval to a point.
        y, x = sample(40, seed=5)
        fit = fit_composite(y, x, QUARTILES)
        residual = np.sort(y - x @ fit.slopes)
        step, genuine = 1e-6, 0
        for j, tau in enumerate(QUARTILES):
            k = int(40 * tau)
            low, high = residual[k - 1], residual[k]
            self.assertEqual(fit.intercepts[j], low)
            self.assertTrue(low - 1e-9 <= fit.lp_intercepts[j] <= high + 1e-9)
            base = check_loss(residual - low, tau)
            self.assertGreater(check_loss(residual - (low - step), tau), base + step / 2)
            if high - low > 2 * step:
                genuine += 1
                self.assertAlmostEqual(check_loss(residual - (low + step), tau), base, places=9)
                self.assertAlmostEqual(check_loss(residual - high, tau), base, places=9)
        self.assertGreater(genuine, 0)

    def test_ordered_intercepts(self):
        levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        for n, seed in ((20, 1), (37, 2), (300, 3)):
            y, x = sample(n, seed)
            fit = fit_composite(y, x, levels)
            self.assertTrue(np.all(np.diff(fit.intercepts) >= 0))

    def test_all_equal_responses(self):
        x = np.random.default_rng(4).uniform(-1, 1, (30, 1))
        fit = fit_composite(np.full(30, 3.0), x, QUARTILES)
        np.testing.assert_allclose(fit.slopes, 0.0, atol=1e-12)
        np.testing.assert_allclose(fit.intercepts, 3.0, rtol=0, atol=1e-12)
        self.assertLessEqual(fit.loss, 1e-10)

    def test_translation_equivariance(self):
        y, x = sample(40, seed=8)
        base = fit_composite(y, x, QUARTILES)
        for shift in (2.5, -1e3):
            moved = fit_composite(y + shift, x, QUARTILES)
            self.assertTrue(np.all(within(moved.slopes, base.slopes)))
            self.assertTrue(np.all(within(moved.intercepts, base.intercepts + shift)))
            self.assertLessEqual(abs(moved.loss - base.loss), OBJECTIVE_RTOL * (1 + base.loss))
        gamma = np.array([0.7, -1.3])
        tilted = fit_composite(y + x @ gamma, x, QUARTILES)
        self.assertTrue(np.all(within(tilted.slopes, base.slopes + gamma)))
        self.assertTrue(np.all(within(tilted.intercepts, base.intercepts)))

    def test_slopes_match_the_plain_lp(self):
        y, x = sample(60, seed=9)
        stacked = np.column_stack((np.repeat(np.eye(3), 60, axis=0), np.tile(x, (3, 1))))
        plain = quantile_lp(np.tile(y, 3), stacked, np.repeat(QUARTILES, 60))
        np.testing.assert_array_equal(fit_composite(y, x, QUARTILES).slopes, plain[3:])
        loss = composite_loss(y, x, plain[:3], plain[3:], QUARTILES)
        self.assertLessEqual(abs(loss - fit_composite(y, x, QUARTILES).loss), OBJECTIVE_RTOL * (1 + loss))


if __name__ == "__main__":
    unittest.main()
