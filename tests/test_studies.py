"""Each study's first replication reproduces its saved reference rows under the comparison rule."""
import io
import json
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

from multiquantile.design import fixed_inputs, target_features
from multiquantile.run_quartiles import run_replicate
from scripts.tolerance import compare_frames, read_table

ROOT = Path(__file__).resolve().parents[1]
MQ = ROOT / "multiquantile"


def assert_rows_match(test, new, reference, key):
    _, failures = compare_frames(new[reference.columns], reference.reset_index(drop=True), key)
    test.assertEqual(failures, [])


class FirstReplication(unittest.TestCase):
    def test_quartile_study(self):
        cfg = json.loads((MQ / "config" / "quartiles.json").read_text())
        source, evaluation = fixed_inputs(cfg)
        features = target_features(evaluation, source)
        rows, _ = run_replicate(0, cfg=cfg, source=source, eval_gram=features.T @ features / len(features))
        rows = pd.DataFrame(rows)
        reference = read_table(MQ / "reference" / "quartiles" / "replications.csv")
        reference = reference[reference.replicate == 0]
        self.assertEqual(len(rows), len(reference))
        assert_rows_match(self, rows, reference, ["replicate", "noise", "scale_slope", "tau"])

    def test_grid(self):
        # A fresh interpreter, as in the grid's runs: HiGHS keeps one global thread pool per
        # process, and the grid's fits request a single thread.
        code = ("import sys, pandas as pd; from run_simulations import fit_trial; "
                "pd.DataFrame(fit_trial(0, 'normal', 0.5, 1000, 100, 5000, [0.0, 0.5, 1.0], 'known', 1000))"
                ".to_csv(sys.stdout, index=False)")
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "singlequantile" / "grid",
                                capture_output=True, text=True, check=True)
        rows = read_table(io.StringIO(result.stdout))
        reference = read_table(ROOT / "singlequantile" / "grid" / "reference" / "raw_results.csv.gz")
        reference = reference[(reference.experiment == "known") & (reference.rep == 0) & (reference.noise == "normal")
                              & (reference.tau == 0.5) & (reference.nT == 100)]
        self.assertEqual(len(rows), 15)
        assert_rows_match(self, rows, reference, ["experiment", "rep", "noise", "tau", "delta", "nT", "N", "method"])


if __name__ == "__main__":
    unittest.main()
