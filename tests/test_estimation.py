"""Quantile-regression fits against hand calculations and an independent primal LP."""
import unittest

import numpy as np

from multiquantile.estimation import OBJECTIVE_RTOL, check_loss, composite_design, quantile_lp
from multiquantile.validate_quartiles import independent_primal


class CheckLossTest(unittest.TestCase):
    def test_values(self):
        residual = np.array([-2.0, 0.0, 3.0])
        self.assertAlmostEqual(check_loss(residual, 0.25), 0.75 * 2 + 0.25 * 3)
        self.assertAlmostEqual(check_loss(residual, np.array([0.5, 0.5, 0.9])), 1.0 + 0.9 * 3)


class QuantileLpTest(unittest.TestCase):
    def test_intercept_only_median(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        beta = quantile_lp(y, np.ones((4, 1)), 0.5)
        self.assertTrue(2 - 1e-9 <= beta[0] <= 3 + 1e-9)
        self.assertAlmostEqual(check_loss(y - beta[0], 0.5), 2.0)

    def test_single_and_composite_fits_reach_the_primal_optimum(self):
        rng = np.random.default_rng(7)
        x = np.column_stack((np.ones(40), rng.uniform(-1, 1, (40, 2))))
        y = x @ [0.5, 1.0, -2.0] + rng.standard_t(3, 40)
        for tau in (0.1, 0.5, 0.75):
            beta = quantile_lp(y, x, tau)
            loss = check_loss(y - x @ beta, tau)
            self.assertLessEqual(abs(loss - independent_primal(y, x, tau)), OBJECTIVE_RTOL * (1 + loss))
        levels = np.array([0.25, 0.5, 0.75])
        stacked, response, per_row = composite_design(x[:, 1:], 3), np.tile(y, 3), np.repeat(levels, 40)
        beta = quantile_lp(response, stacked, per_row)
        loss = check_loss(response - stacked @ beta, per_row)
        self.assertLessEqual(abs(loss - independent_primal(response, stacked, per_row)), OBJECTIVE_RTOL * (1 + loss))

    def test_composite_design_blocks(self):
        slopes = np.arange(6.0).reshape(3, 2)
        stacked = composite_design(slopes, 2)
        self.assertEqual(stacked.shape, (6, 4))
        np.testing.assert_array_equal(stacked[:, :2], np.repeat(np.eye(2), 3, axis=0))
        np.testing.assert_array_equal(stacked[3:, 2:], slopes)


if __name__ == "__main__":
    unittest.main()
