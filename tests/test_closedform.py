"""Closed-form values that can be written by hand."""
import math
import unittest

from scipy.stats import cauchy, norm

from closedform.local import ALPHA, boundary_noncentrality, constants, power


class ClosedForm(unittest.TestCase):
    def test_variance_multipliers(self):
        self.assertAlmostEqual(constants(norm)["c"][1], math.pi / 2, places=12)
        k = constants(cauchy)
        self.assertAlmostEqual(k["c"][1], math.pi ** 2 / 4, places=12)
        self.assertAlmostEqual(k["c"][0], 3 * math.pi ** 2 / 4, places=12)
        self.assertAlmostEqual(k["c"][2], 3 * math.pi ** 2 / 4, places=12)

    def test_cauchy_gamma_and_boundary(self):
        k = constants(cauchy)
        self.assertAlmostEqual(k["gamma"], 4 / math.pi ** 2, places=12)
        self.assertAlmostEqual(boundary_noncentrality(k, 1), 1.75, places=12)
        self.assertAlmostEqual(boundary_noncentrality(k, 3), 3 * 1.75, places=12)

    def test_power_at_zero_noncentrality(self):
        for nu in (2, 4, 10, 20):
            self.assertEqual(power(nu, 0), ALPHA)
            self.assertAlmostEqual(power(nu, 1e-12), 0.05, places=9)


if __name__ == "__main__":
    unittest.main()
