"""Small helpers shared by the multiquantile runners and validators."""
import csv

import numpy as np


def write_csv(path, rows):
    """Write a list of dicts with identical keys, in key order."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path):
    """Read a CSV file into a list of dicts of strings."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def standard_error(values):
    """Monte Carlo standard error of a mean: sample standard deviation over sqrt(R)."""
    return np.std(values, ddof=1) / np.sqrt(len(values))
