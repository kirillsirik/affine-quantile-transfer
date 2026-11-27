"""The numbers of paper/numbers.csv agree with the saved results, and a changed number or file fails."""
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import check_numbers

ROOT = Path(__file__).resolve().parents[1]


def last_digit_up(text):
    """'0.090' -> '0.091', '12,275' -> '12,276': the printed number moved by one unit in its last place."""
    digits = [i for i, c in enumerate(text) if c.isdigit()]
    i = digits[-1]
    if text[i] != "9":
        return text[:i] + str(int(text[i]) + 1) + text[i + 1:]
    return text[:i] + "8" + text[i + 1:]


class NumbersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = check_numbers.read_numbers()

    def test_every_number_agrees_with_the_saved_results(self):
        sources = check_numbers.Sources()
        failures = [(row["where"], row["quantity"]) for row in self.rows if not check_numbers.check(row, sources)[0]]
        self.assertEqual(failures, [])

    def test_every_table_and_figure_is_listed(self):
        places = {row["where"] for row in self.rows}
        for exhibit in ("Table 1", "Table 2", "Table 3", "Table 4", "Table 5", "Table 6", "Table 7",
                        "Figure 1", "Figure 2", "Figure 3"):
            self.assertIn(exhibit, places)

    def test_a_number_off_by_one_in_its_last_digit_fails(self):
        sources = check_numbers.Sources()
        numbers = [row for row in self.rows if row["decimals"] != "" and row["printed"][-1].isdigit()]
        for row in numbers[::25]:
            changed = {**row, "printed": last_digit_up(row["printed"])}
            self.assertFalse(check_numbers.check(changed, sources)[0], (row["where"], row["quantity"]))

    def test_a_false_statement_fails(self):
        row = next(row for row in self.rows if row["decimals"] == "")
        self.assertFalse(check_numbers.check({**row, "value": f"not ({row['value']})"}, check_numbers.Sources())[0])

    def test_a_run_directory_is_read_instead_of_the_saved_results(self):
        run = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, run)
        shutil.copytree(ROOT / "closedform" / "reference", run / "closed-form")
        rows = [row for row in self.rows if row["source"].startswith("closedform/reference/")]
        self.assertTrue(all(check_numbers.check(row, check_numbers.Sources(run))[0] for row in rows))
        constants = run / "closed-form" / "constants.csv"
        constants.write_text(constants.read_text().replace("1.856767469110269", "1.9", 1))
        changed = check_numbers.Sources(run)
        self.assertFalse(all(check_numbers.check(row, changed)[0] for row in rows if row["where"] == "Table 1"))


if __name__ == "__main__":
    unittest.main()
