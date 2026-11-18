"""The comparison rule fixed before the reruns, and the table comparison built on it."""
import unittest

import numpy as np
import pandas as pd

from scripts.tolerance import ATOL, RTOL, compare_frames, json_differences, within


class WithinTest(unittest.TestCase):
    def test_rule_constants(self):
        self.assertEqual((RTOL, ATOL), (1e-9, 1e-10))

    def test_relative_and_absolute_parts(self):
        self.assertTrue(within(1.0, 1.0 + 5e-10))
        self.assertFalse(within(1.0, 1.0 + 2e-9))
        self.assertTrue(within(0.0, 5e-11))
        self.assertFalse(within(0.0, 2e-10))
        self.assertTrue(within(1e6, 1e6 + 1e-4))

    def test_missing_values_must_stay_missing(self):
        self.assertTrue(within(np.nan, np.nan))
        self.assertFalse(within(np.nan, 1.0))
        self.assertFalse(within(1.0, np.nan))


class CompareFramesTest(unittest.TestCase):
    def setUp(self):
        self.old = pd.DataFrame({"k": [1, 2, 3], "label": ["a", "b", "c"], "n": [1, 2, 3], "x": [0.1, 0.2, 0.3]})

    def test_identical_tables_pass(self):
        _, failures = compare_frames(self.old.copy(), self.old, ["k"])
        self.assertEqual(failures, [])

    def test_each_kind_of_mismatch_fails(self):
        new = self.old.copy()
        new.loc[0, "label"] = "z"
        new.loc[1, "n"] = 7
        new.loc[2, "x"] = 0.3 + 1e-6
        _, failures = compare_frames(new, self.old, ["k"])
        self.assertEqual(len(failures), 3)

    def test_missing_rows_and_duplicate_keys_fail(self):
        _, failures = compare_frames(self.old.iloc[:2], self.old, ["k"])
        self.assertTrue(any("unmatched" in f for f in failures))
        _, failures = compare_frames(pd.concat([self.old, self.old.iloc[:1]]), self.old, ["k"])
        self.assertTrue(any("duplicate" in f for f in failures))

    def test_subset_mode_allows_extra_reference_rows_only(self):
        _, failures = compare_frames(self.old.iloc[:2], self.old, ["k"], subset=True)
        self.assertEqual(failures, [])
        extra = pd.concat([self.old, pd.DataFrame({"k": [9], "label": ["q"], "n": [9], "x": [0.9]})])
        _, failures = compare_frames(extra, self.old, ["k"], subset=True)
        self.assertTrue(any("unmatched" in f for f in failures))

    def test_json_rule_ignores_python_and_workers(self):
        self.assertEqual(json_differences({"python": "3.12.13", "workers": 2, "x": 1.0},
                                          {"python": "3.12.14", "workers": 4, "x": 1.0 + 1e-12}), [])
        self.assertTrue(json_differences({"x": 1.0}, {"x": 1.1}))


if __name__ == "__main__":
    unittest.main()
