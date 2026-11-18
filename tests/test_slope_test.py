"""The slope-test run: replication 0 against its reference rows and the quartile study, and its configuration."""
import json
import unittest
from pathlib import Path

import pandas as pd

from multiquantile.run_slope_test import run_replicate
from scripts.tolerance import compare_frames, read_table

MQ = Path(__file__).resolve().parents[1] / "multiquantile"


class SlopeTest(unittest.TestCase):
    def test_configuration_shares_the_quartile_values(self):
        slope = json.loads((MQ / "config" / "slope_test.json").read_text())
        quartiles = json.loads((MQ / "config" / "quartiles.json").read_text())
        for key, value in slope.items():
            if key != "replicates":
                self.assertEqual(value, quartiles[key], key)

    def test_replication_zero(self):
        cfg = json.loads((MQ / "config" / "slope_test.json").read_text())
        rows = pd.DataFrame(run_replicate(0, cfg=cfg))
        reference = read_table(MQ / "reference" / "slope-test" / "statistics.csv")
        reference = reference[reference.replicate == 0].reset_index(drop=True)
        self.assertEqual(len(rows), len(reference))
        _, failures = compare_frames(rows[reference.columns], reference, ["replicate", "noise", "scale_slope"])
        self.assertEqual(failures, [])
        quartile = read_table(MQ / "reference" / "quartiles" / "replications.csv")
        quartile = quartile[(quartile.replicate == 0) & (quartile.tau == 0.5)]
        joined = rows.merge(quartile, on=["replicate", "noise", "scale_slope"], suffixes=("", "_quartile"))
        self.assertEqual(len(joined), len(rows))
        self.assertTrue(((joined.wald - joined.wald_quartile).abs() <= 1e-8 * joined.wald_quartile.abs()).all())


if __name__ == "__main__":
    unittest.main()
