"""The ACS variables and fitting conventions against hand calculations, and the first replication
against the saved reference rows."""
import json
import math
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from acs.estimation import (coverage, fit, fit_composite, independent_columns, mean_check_loss, row_levels, score_map,
                            uniqueness_margin)
from acs.variables import COVARIATES, STATES, covariates, dictionary, income, log_income, read_extract, schooling
from multiquantile import estimation as mq
from multiquantile.estimation import left_empirical_quantile, slope_wald
from scripts.tolerance import compare_frames, read_table

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "acs" / "reference"


def records(**columns):
    base = {"SERIALNO": "x", "SPORDER": 1, "PWGTP": 1, "PINCP": 50000, "WAGP": 50000, "ADJINC": 1015250}
    n = len(next(iter(columns.values())))
    return pd.DataFrame({**{k: [v] * n for k, v in base.items()}, **columns})


class VariablesTest(unittest.TestCase):
    def test_schooling_map(self):
        codes = [1, 3, 4, 10, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        years = [0, 0, 1, 7, 11, 11, 12, 12, 12.5, 13, 14, 16, 18, 19, 20]
        np.testing.assert_array_equal(schooling(np.array(codes)), years)
        with self.assertRaises(ValueError):
            schooling(np.array([25]))

    def test_three_records_by_hand(self):
        frame = records(AGEP=[45, 25, 62], SCHL=[21, 16, 24], WKHP=[50, 20, 99], WKWN=[52, 40, 1], SEX=[2, 1, 2],
                        COW=[6, 4, 1], MAR=[1, 3, 1])
        expected_x = [[0.5, 1.0, 1.0, 0.0, 1, 1, 0, 1],
                      [-1.5, -1 / 3, -2.0, -1.2, 0, 0, 1, 0],
                      [2.2, 7 / 3, 5.9, -5.1, 1, 0, 0, 1]]
        expected_phi = [[1, 0.5, 1.5, 0, 1.0, 4 / 3, 0, 1.0, 1.0, 0.0, 1, 1, 0, 1, 1.0, 1.0],
                        [1, -1.5, 0, 0, -1 / 3, 0, 0, -2.0, 0, -1.2, 0, 0, 1, 0, 0, 2 / 3],
                        [1, 2.2, 3.2, 1.2, 7 / 3, 8 / 3, 4 / 3, 5.9, 5.9, -5.1, 1, 0, 0, 1, 7 / 3, 7 / 3 * 5.9]]
        np.testing.assert_allclose(covariates(frame), expected_x, rtol=0, atol=1e-14)
        np.testing.assert_allclose(dictionary(frame), expected_phi, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(income(frame), [50.7625] * 3)
        np.testing.assert_allclose(log_income(frame), [math.log(50762.5)] * 3, rtol=1e-15)


class EstimationTest(unittest.TestCase):
    def test_intercept_only_fit_is_the_left_empirical_quantile_on_tied_data(self):
        ties = np.random.default_rng(1).random(8)
        beta, _ = fit(np.array([0.0, 2.0, 4.0, 8.0]), np.ones((4, 1)), 0.5, ties[:4])
        self.assertEqual(beta[0], 2.0)
        tied = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 5.0])
        for tau in (0.25, 0.5, 0.6, 0.75, 0.9):
            beta, _ = fit(tied, np.ones((8, 1)), tau, ties)
            self.assertEqual(beta[0], left_empirical_quantile(tied, tau), tau)

    def test_composite_fit_matches_the_simulation_studies_on_continuous_data(self):
        rng = np.random.default_rng(5)
        x = rng.uniform(-1, 1, (300, 2))
        y = 0.4 + x @ [1.0, -0.5] + rng.standard_t(3, 300)
        intercepts, slopes = fit_composite(y, x, [0.25, 0.5, 0.75], rng.random(300))
        reference = mq.fit_composite(y, x, np.array([0.25, 0.5, 0.75]))
        np.testing.assert_allclose(slopes, reference.slopes, rtol=0, atol=1e-12)
        np.testing.assert_allclose(intercepts, reference.intercepts, rtol=0, atol=1e-12)

    def test_column_rule(self):
        rng = np.random.default_rng(2)
        x1, x2 = rng.uniform(-1, 1, (2, 40))
        design = np.column_stack((np.ones(40), x1, np.zeros(40), 2 * x1 + 1, x2))
        np.testing.assert_array_equal(independent_columns(design), [True, True, False, False, True])
        y = 1 + x1 - x2 + rng.standard_normal(40)
        ties = rng.random(40)
        beta, dropped = fit(y, design, 0.5, ties)
        reduced, _ = fit(y, design[:, [0, 1, 4]], 0.5, ties)
        self.assertEqual(dropped, 2)
        np.testing.assert_array_equal(beta[[2, 3]], 0.0)
        np.testing.assert_array_equal(beta[[0, 1, 4]], reduced)

    def test_uniqueness_margin(self):
        y, design = np.array([1.0, 2.0, 3.0, 4.0]), np.ones((4, 1))
        exact = np.full(4, 0.5)
        self.assertAlmostEqual(uniqueness_margin(y, design, np.array([2.0]), exact), 0.0, delta=1e-9)
        self.assertEqual(uniqueness_margin(y, design, np.array([2.5]), exact), -1.0)
        ties = np.random.default_rng(3).random(4)
        diagnostics = []
        beta, _ = fit(y, design, 0.5, ties, diagnostics)
        self.assertEqual(beta[0], 2.0)
        self.assertGreater(diagnostics[0][2], 0)
        self.assertGreater(uniqueness_margin(y, design, beta, row_levels(0.5, ties)), 0)

    def test_coverage_tolerance(self):
        y = np.array([1.0, 1.0 + 5e-9, 1.0 + 2e-8, 0.5])
        self.assertEqual(coverage(y, np.ones(4), 1e-8), 0.75)
        self.assertEqual(mean_check_loss(np.array([2.0, 0.0]), np.ones(2), 0.9), (0.9 + 0.1) / 2)


class FirstReplication(unittest.TestCase):
    def test_primary_target_at_n_100_reproduces_both_studies(self):
        cfg = json.loads((REFERENCE / "metadata.json").read_text())["configuration"]
        source_fits = {(f["source"], f["scale"], f["tau"]): np.array(f["coefficients"])
                       for f in json.loads((REFERENCE / "metadata.json").read_text())["source_fits"]}
        target, n, tau = cfg["primary_target"], 100, cfg["level"]

        def generator(stream, state, index):
            return np.random.default_rng(np.random.SeedSequence([cfg["seed"], stream, STATES[state][1], index]))

        frame, source = read_extract(target), read_extract(cfg["source"])
        x, phi, y = covariates(frame), dictionary(frame), income(frame)
        ties = generator(3, target, 0).random(len(frame))
        source_ties = generator(3, cfg["source"], 0).random(len(source))
        perm = generator(2, target, 0).permutation(len(frame))
        labeled, held = perm[:n], perm[max(cfg["labeled_sizes"]):]
        center, loadings, p = score_map(x[held], 4)
        scores = (x - center) @ loadings / np.sqrt(p)
        prediction = phi @ source_fits[(cfg["source"], "level", tau)]
        design = np.column_stack((np.ones(len(x)), prediction, scores[:, :2], x[:, COVARIATES.index("school")]))
        transfer, _ = fit(y[labeled], design[labeled], tau, ties[labeled])
        target_only, _ = fit(y[labeled], phi[labeled], tau, ties[labeled])
        pooled, _ = fit(np.r_[income(source), y[labeled]], np.vstack((dictionary(source), phi[labeled])), tau,
                        np.r_[source_ties, ties[labeled]])
        rows = []
        for method, predicted, beta in (("transfer", design[held] @ transfer, transfer),
                                        ("target_only", phi[held] @ target_only, target_only),
                                        ("pooled", phi[held] @ pooled, pooled), ("source_only", prediction[held], None)):
            rows.append({"source": cfg["source"], "target": target, "n": n, "replicate": 0, "method": method,
                         "loss": mean_check_loss(y[held], predicted, tau), "coverage": coverage(y[held], predicted, 1e-8),
                         "coef_norm": np.linalg.norm(beta) if beta is not None else np.nan,
                         **{f"coef_{t}": (transfer[i] if method == "transfer" else np.nan)
                            for i, t in enumerate(("intercept", "source", "score1", "score2", "school"))},
                         "dropped_columns": 0, "n_eval": len(held)})
        reference = read_table(REFERENCE / "transfer_replications.csv.gz")
        reference = reference[(reference.source == cfg["source"]) & (reference.target == target) & (reference.n == n)
                              & (reference.replicate == 0)].reset_index(drop=True)
        new = pd.DataFrame(rows)[reference.columns]
        _, failures = compare_frames(new, reference, ["source", "target", "n", "replicate", "method"])
        self.assertEqual(failures, [])
        rows = []
        for scale, response in (("level", y), ("log", log_income(frame))):
            median = phi @ source_fits[(cfg["source"], scale, 0.5)]
            slopes_design = np.column_stack((median, x[:, COVARIATES.index("school")]))
            full = np.column_stack((np.ones(len(x)), slopes_design))
            separate = np.vstack([fit(response[labeled], full[labeled], q, ties[labeled])[0] for q in cfg["quartiles"]])
            intercepts, slopes = fit_composite(response[labeled], slopes_design[labeled], cfg["quartiles"], ties[labeled])
            wald, p_value = slope_wald(response[labeled], full[labeled], np.array(cfg["quartiles"]), separate)
            for j, q in enumerate(cfg["quartiles"]):
                rows.append({"scale": scale, "target": target, "n": n, "replicate": 0, "tau": q,
                             "loss_composite": mean_check_loss(response[held], intercepts[j] + slopes_design[held] @ slopes, q),
                             "loss_separate": mean_check_loss(response[held], full[held] @ separate[j], q),
                             "slope_source_composite": slopes[0], "slope_school_composite": slopes[1],
                             "slope_source_separate": separate[j, 1], "slope_school_separate": separate[j, 2],
                             "wald": wald, "p_value": p_value})
        reference = read_table(REFERENCE / "sharing_replications.csv.gz")
        reference = reference[(reference.target == target) & (reference.n == n)
                              & (reference.replicate == 0)].reset_index(drop=True)
        _, failures = compare_frames(pd.DataFrame(rows)[reference.columns], reference,
                                     ["scale", "target", "n", "replicate", "tau"])
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
