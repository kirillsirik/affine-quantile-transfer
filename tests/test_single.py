"""The single-quantile trial against the grid's function, and replication 0 against the reference rows."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from scripts.tolerance import compare_frames, read_table

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "singlequantile" / "reference"
GRID = ROOT / "singlequantile" / "grid"
KEY = ["variant", "noise", "replicate", "departure"]
# A fresh interpreter, as in the grid's runs: HiGHS keeps one global thread pool per process, and the
# grid's fits request a single thread.
CODE = """
import json, sys
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import run_simulations as rs
from singlequantile.trial import trial
out = {}
for noise in rs.NOISES:
    out[noise] = {"grid": rs.fit_trial(0, noise, 0.9, 1000, 100, 5000, [0.0, 0.5, 1.0], "known", 1000),
                  "bounded": trial(rs, 0, noise, 10.0, True), "unbounded": trial(rs, 0, noise, None, False)}
json.dump(out, sys.stdout)
"""


class SingleQuantileTrial(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run([sys.executable, "-c", CODE, str(GRID), str(ROOT)],
                                cwd=GRID, capture_output=True, text=True, check=True)
        cls.trials = json.loads(result.stdout)

    def test_trial_with_the_bound_and_the_ball_reproduces_fit_trial(self):
        for noise, runs in self.trials.items():
            grid = {(r["delta"], r["method"]): r for r in runs["grid"]}
            for row in runs["bounded"]:
                transfer = grid[(row["departure"], "transfer")]
                self.assertEqual(row["mse_transfer"], transfer["mse"])
                self.assertEqual(row["mse_target_only"], grid[(row["departure"], "target_only")]["mse"])
                self.assertEqual(row["mse_pooled"], grid[(row["departure"], "pooled")]["mse"])
                self.assertEqual(row["transfer_norm"], transfer["coef_norm"])
                self.assertEqual(row["transfer_fallback"], transfer["affine_fallback"])
                for name in ("source", "target", "pooled"):
                    self.assertEqual(row[f"{name}_bound_active"], transfer[f"{name}_box_active"])

    def test_replication_zero_of_each_variant_reproduces_the_reference_rows(self):
        rows = pd.DataFrame([{"variant": variant, **row} for runs in self.trials.values()
                             for variant in ("bounded", "unbounded") for row in runs[variant]])
        reference = read_table(REFERENCE / "replications.csv.gz")
        reference = reference[reference.replicate == 0].reset_index(drop=True)
        self.assertEqual(len(rows), len(reference))
        _, failures = compare_frames(rows[reference.columns], reference, KEY)
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
