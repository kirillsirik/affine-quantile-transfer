"""Recompute every number of the paper from the saved outputs and compare it with the printed one.

Usage, from the repository root:
    python -m scripts.check_numbers                  # the saved outputs in the repository
    python -m scripts.check_numbers --run-dir DIR    # the outputs that make reproduce wrote to DIR

paper/numbers.csv has one row per number printed in a table or in the text of the paper. A row
names the output file the number comes from (`source`), the rows of that file it uses (`rows`, a
pandas query) and an expression (`value`) that turns them into the printed number. Columns of the
selected rows are available by name, a JSON file as `data`; one(x) is the single value of x, and
cell(file, column, **conditions) is the single value of a column in another file. The result,
rounded to `decimals` places, must equal the printed number. A row without `decimals` is a
statement in words, such as "in more than half of its samples", and its expression must be true.

With --run-dir, files that make reproduce writes are read from DIR; the ACS extracts in acs/data/,
which make reproduce does not rebuild, are read from the repository. Exits nonzero if any number
differs.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "paper" / "numbers.csv"
COLUMNS = ["where", "quantity", "printed", "decimals", "source", "rows", "value"]
# Saved outputs and the folders of a make reproduce run that hold the same files.
RUN_FOLDERS = {
    "singlequantile/grid/reference/": "grid/",
    "singlequantile/reference/": "single/",
    "multiquantile/reference/quartiles/": "quartiles/",
    "multiquantile/reference/slope-test/": "slope-test/",
    "closedform/reference/": "closed-form/",
    "acs/reference/": "acs/",
}


def one(values):
    values = np.asarray(values)
    if values.size != 1:
        raise ValueError(f"expected one value, found {values.size}")
    return values.item()


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def printed_value(text):
    """The number in a printed string such as '−0.052', '12,275', '31.8%' or 'five'."""
    if text in WORDS:
        return float(WORDS[text])
    return float(text.replace("\N{MINUS SIGN}", "-").replace(",", "").rstrip("%"))


class Sources:
    def __init__(self, run_dir=None):
        self.run_dir = run_dir
        self.cache = {}

    def path(self, source):
        if self.run_dir is not None:
            for prefix, folder in RUN_FOLDERS.items():
                if source.startswith(prefix):
                    path = self.run_dir / folder / source[len(prefix):]
                    # make reproduce writes the raw rows of the grid uncompressed
                    if not path.exists() and path.suffix == ".gz":
                        path = path.with_suffix("")
                    return path
        return ROOT / source

    def read(self, source):
        if source not in self.cache:
            path = self.path(source)
            if path.suffix == ".json":
                self.cache[source] = json.loads(path.read_text())
            else:
                self.cache[source] = pd.read_csv(path, keep_default_na=False, na_values=[""])
        return self.cache[source]


def evaluate(row, sources):
    content = sources.read(row["source"])

    def cell(source, column, **conditions):
        frame = sources.read(source)
        for name, value in conditions.items():
            frame = frame[frame[name] == value]
        return one(frame[column])

    names = {"__builtins__": {}, "one": one, "cell": cell, "np": np, "stats": stats, "math": math}
    names.update({f.__name__: f for f in (abs, all, any, float, int, len, list, max, min, range, round, set, sorted,
                                          sum, zip)})
    if isinstance(content, dict):
        names["data"] = content
    else:
        frame = content.query(row["rows"], engine="python") if row["rows"] else content
        if frame.empty:
            raise ValueError("no rows selected")
        names.update({column: frame[column] for column in frame.columns})
        names["frame"] = frame
    return eval(row["value"], names)


def check(row, sources):
    """(ok, computed) for one row of paper/numbers.csv."""
    result = evaluate(row, sources)
    if row["decimals"] == "":
        return bool(result), bool(result)
    decimals = int(row["decimals"])
    computed = float(result)
    printed = printed_value(row["printed"])
    return abs(computed - printed) <= 0.5 * 10 ** -decimals + 1e-9 * max(1.0, abs(printed)), computed


def read_numbers():
    with NUMBERS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != COLUMNS:
            raise SystemExit(f"{NUMBERS.relative_to(ROOT)}: expected the columns {', '.join(COLUMNS)}")
        return list(reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    sources = Sources(args.run_dir.resolve() if args.run_dir else None)
    rows = read_numbers()
    failures = []
    for line, row in enumerate(rows, start=2):
        try:
            ok, computed = check(row, sources)
        except Exception as error:  # a missing file or column is a failure of that row
            ok, computed = False, f"error: {error}"
        if not ok:
            failures.append(f"  line {line}: {row['where']}, {row['quantity']}: printed {row['printed']}, "
                            f"computed {computed}")
    numbers = sum(row["decimals"] != "" for row in rows)
    where = f"the run in {args.run_dir}" if args.run_dir else "the saved outputs"
    if failures:
        print(f"{len(failures)} of {len(rows)} rows of paper/numbers.csv differ from {where}:")
        print("\n".join(failures))
        sys.exit(1)
    print(f"paper/numbers.csv: all {numbers} numbers and {len(rows) - numbers} statements agree with {where}.")


if __name__ == "__main__":
    main()
