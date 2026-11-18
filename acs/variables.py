"""Variables of the ACS study, as pure functions of the columns of a committed extract.

Outcome: income = PINCP * (ADJINC / 1e6) / 1000, total person income in thousands of 2024 dollars.
Covariates x (8 entries): age, school, hours, weeks, female, selfemp, gov, married.
Dictionary Phi(x) (16 columns): the covariates, an intercept, hinge terms in age, schooling and
hours, and two interactions. acs/README.md lists the definitions.
"""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Postal code: (state name, FIPS code). The extract of a state is <postal code in lower case>.csv.gz.
STATES = {
    "CA": ("California", 6), "TX": ("Texas", 48), "WY": ("Wyoming", 56), "VT": ("Vermont", 50),
    "AK": ("Alaska", 2), "ND": ("North Dakota", 38), "SD": ("South Dakota", 46), "DE": ("Delaware", 10),
    "RI": ("Rhode Island", 44), "MT": ("Montana", 30),
}
COLUMNS = ("SERIALNO", "SPORDER", "PWGTP", "AGEP", "SCHL", "WKHP", "WKWN", "SEX", "COW", "MAR", "PINCP",
           "WAGP", "ADJINC")
COVARIATES = ("age", "school", "hours", "weeks", "female", "selfemp", "gov", "married")
DICTIONARY = ("intercept", "age", "age_over_30", "age_over_50", "school", "school_over_12", "school_over_16",
              "hours", "hours_over_40", "weeks", "female", "selfemp", "gov", "married", "female_x_school",
              "school_x_hours")
# Years of schooling S for SCHL = 1, ..., 24 (index 0 is unused).
SCHOOLING_YEARS = np.array([np.nan, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 12, 12.5, 13, 14, 16,
                            18, 19, 20])


def extract_path(state, folder=DATA):
    return Path(folder) / f"{state.lower()}.csv.gz"


def read_extract(state, folder=DATA):
    """The committed extract of one state: SERIALNO as text, every other column as int64."""
    with gzip.open(extract_path(state, folder), "rt", newline="") as handle:
        frame = pd.read_csv(handle, dtype={"SERIALNO": str})
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"unexpected columns in the {state} extract")
    return frame


def schooling(schl):
    """Years of schooling from SCHL (1-24)."""
    schl = np.asarray(schl)
    if np.any((schl < 1) | (schl > 24)):
        raise ValueError("SCHL outside 1-24")
    return SCHOOLING_YEARS[schl.astype(int)]


def income(frame):
    """Total person income in thousands of 2024 dollars, PINCP * (ADJINC / 1e6) / 1000.0 in this order."""
    return frame["PINCP"].to_numpy(dtype=float) * (frame["ADJINC"].to_numpy(dtype=float) / 1e6) / 1000.0


def log_income(frame):
    """Logarithm of income in dollars."""
    return np.log(1000.0 * income(frame))


def covariates(frame):
    """The n by 8 covariate matrix, columns in the order of COVARIATES."""
    years = schooling(frame["SCHL"])
    cow = frame["COW"].to_numpy()
    return np.column_stack((
        (frame["AGEP"].to_numpy(dtype=float) - 40) / 10,
        (years - 13) / 3,
        (frame["WKHP"].to_numpy(dtype=float) - 40) / 10,
        (frame["WKWN"].to_numpy(dtype=float) - 52) / 10,
        (frame["SEX"].to_numpy() == 2).astype(float),
        np.isin(cow, (6, 7)).astype(float),
        np.isin(cow, (3, 4, 5)).astype(float),
        (frame["MAR"].to_numpy() == 1).astype(float),
    ))


def dictionary(frame):
    """The n by 16 dictionary Phi(x), columns in the order of DICTIONARY."""
    age = frame["AGEP"].to_numpy(dtype=float)
    years = schooling(frame["SCHL"])
    hours = frame["WKHP"].to_numpy(dtype=float)
    x = covariates(frame)
    return np.column_stack((
        np.ones(len(frame)), x[:, 0], np.maximum(age - 30, 0) / 10, np.maximum(age - 50, 0) / 10,
        x[:, 1], np.maximum(years - 12, 0) / 3, np.maximum(years - 16, 0) / 3,
        x[:, 2], np.maximum(hours - 40, 0) / 10, x[:, 3], x[:, 4], x[:, 5], x[:, 6], x[:, 7],
        x[:, 4] * x[:, 1], x[:, 1] * x[:, 2],
    ))


def hill(values, k):
    """Hill estimate of the tail index from the k largest values: 1 / mean(log(Y_(i) / Y_(k+1)))."""
    ordered = np.sort(np.asarray(values, dtype=float))[::-1]
    return float(1.0 / np.mean(np.log(ordered[:k] / ordered[k])))


def describe(frame, wage_threshold):
    """Descriptive statistics of one sample; income in thousands of 2024 dollars.

    wage_threshold is the state's WAGTPCT in dollars. sd uses divisor N - 1; kurtosis is the fourth
    central moment over the squared variance, both with divisor N. Quantiles are left inverses
    (numpy method inverted_cdf). shared_record is the share of records whose eight covariates and
    PINCP equal those of another record of the sample.
    """
    y = income(frame)
    n = len(y)
    levels = (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)
    q = np.quantile(y, levels, method="inverted_cdf")
    centered = y - y.mean()
    x = covariates(frame)
    keys = pd.DataFrame(x, columns=COVARIATES).assign(PINCP=frame["PINCP"].to_numpy())
    row = {"records": n, "mean": float(y.mean()), "sd": float(y.std(ddof=1))}
    row.update({f"q{round(100 * level):02d}": float(value) for level, value in zip(levels, q)})
    row.update({
        "maximum": float(y.max()),
        "kurtosis": float(np.mean(centered ** 4) / np.mean(centered ** 2) ** 2),
        "ratio_q99_median": float(q[5] / q[2]),
        "hill_k_2pct": int(np.floor(0.02 * n)), "hill_alpha_2pct": hill(y, int(np.floor(0.02 * n))),
        "hill_k_5pct": int(np.floor(0.05 * n)), "hill_alpha_5pct": hill(y, int(np.floor(0.05 * n))),
        "share_pincp_multiple_1000": float(np.mean(frame["PINCP"].to_numpy() % 1000 == 0)),
        "share_pincp_multiple_5000": float(np.mean(frame["PINCP"].to_numpy() % 5000 == 0)),
        "share_wage_topcoded": float(np.mean(frame["WAGP"].to_numpy() >= wage_threshold)),
        "shared_record": float(np.mean(keys.duplicated(keep=False).to_numpy())),
    })
    row.update({f"mean_{name}": float(value) for name, value in zip(COVARIATES, x.mean(axis=0))})
    return row
