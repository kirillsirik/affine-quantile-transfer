"""Build the committed ACS extracts from the Census Bureau's public-use files.

Usage, from the repository root:
    python -m acs.build_data --raw build/acs-raw --out DIR [--check]

Downloads the 2024 ACS 1-Year PUMS person files of ten states and the 2024 top-code table from
www2.census.gov into the --raw directory (kept out of git), unless they are already there, and
tests each zip. For each state it reads the 13 columns of acs/README.md, applies the six filters in
order, counts the records after each step, sorts by SERIALNO (text) and SPORDER (integer), keeps a
20,000-record subsample of California and of Texas and every record of the eight targets, and
writes to DIR: <state>.csv.gz for the ten states (gzip without a timestamp), sample_counts.csv,
population.csv (descriptive statistics of every filtered record of California and Texas),
topcodes.csv and manifest.json.

With --check, DIR is a scratch directory: every download must match the SHA-256 recorded in
acs/data/manifest.json, and the rebuilt extracts, counts and top codes must equal the committed
files byte for byte (uncompressed), population.csv under the comparison rule of scripts/tolerance.py.
The study itself reads only acs/data/ and needs no network.
"""
import argparse
import gzip
import hashlib
import io
import json
import shutil
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import numpy as np
import pandas as pd

from multiquantile.common import write_csv

from .variables import COLUMNS, DATA, STATES, describe, extract_path

DIRECTORY = "https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/"
TOPCODES = ("https://www2.census.gov/programs-surveys/acs/tech_docs/pums/top_coded_values/"
            "2024_pums_top_and_bottom_coded_values.csv")
TOPCODE_COLUMNS = ("STATE", "STATEABRV", "WAGTPCT", "T_WAG")
USER_AGENT = "Mozilla/5.0 (compatible; acs-extract-builder/1.0)"
SEED = 20261114
SUBSAMPLE = {"CA": 20000, "TX": 20000}
MINIMUM = {"CA": 100000, "TX": 20000}
MINIMUM_TARGET = 2000
FILTERS = (
    ("age_over_16", lambda f: f["AGEP"] > 16),
    ("income_over_100", lambda f: f["PINCP"] > 100),
    ("hours_positive", lambda f: f["WKHP"] > 0),
    ("weight_positive", lambda f: f["PWGTP"] >= 1),
    ("class_of_worker_1_to_7", lambda f: f["COW"].isin(range(1, 8))),
    ("complete", lambda f: f[["SCHL", "WKWN", "SEX", "MAR", "WAGP"]].notna().all(axis=1)),
)
RETRIEVED = "retrieved.json"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def downloads():
    """(file name, URL) of every download, the ten state files first."""
    files = [(f"csv_p{state.lower()}.zip", DIRECTORY + f"csv_p{state.lower()}.zip") for state in STATES]
    return files + [(TOPCODES.rsplit("/", 1)[1], TOPCODES)]


def fetch(raw):
    """Download every missing file into raw; return {file: retrieval record}."""
    raw.mkdir(parents=True, exist_ok=True)
    record_path = raw / RETRIEVED
    records = json.loads(record_path.read_text()) if record_path.exists() else {}
    for name, url in downloads():
        target = raw / name
        if not target.exists():
            print(f"downloading {url}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            partial = raw / (name + ".part")
            with urllib.request.urlopen(request, timeout=600) as response, partial.open("wb") as handle:
                shutil.copyfileobj(response, handle)
                modified = response.headers.get("Last-Modified")
            partial.rename(target)
            records[name] = {"retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "last_modified_utc": (parsedate_to_datetime(modified).astimezone(timezone.utc)
                                                   .strftime("%Y-%m-%dT%H:%M:%SZ") if modified else None)}
            record_path.write_text(json.dumps(records, indent=2) + "\n")
        elif name not in records:
            stamp = datetime.fromtimestamp(target.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            records[name] = {"retrieved_utc": stamp, "last_modified_utc": None,
                             "note": "file was already present; time is its modification time"}
            record_path.write_text(json.dumps(records, indent=2) + "\n")
    return records


def csv_member(path):
    """Name of the single CSV member of a tested zip."""
    with zipfile.ZipFile(path) as archive:
        broken = archive.testzip()
        if broken is not None:
            raise SystemExit(f"{path.name}: corrupt member {broken}")
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
    if len(members) != 1:
        raise SystemExit(f"{path.name}: expected one CSV member, found {members}")
    return members[0]


def read_state(path, member):
    """The 13 columns of one person file; SERIALNO as text, the rest numeric (blank -> missing)."""
    with zipfile.ZipFile(path) as archive, archive.open(member) as handle:
        frame = pd.read_csv(handle, usecols=list(COLUMNS), dtype=str)
    frame = frame[list(COLUMNS)]
    for column in COLUMNS[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def filter_state(state, frame):
    """Apply the six filters in order. Returns the filtered frame and [(step, records)]."""
    counts = [("raw", len(frame))]
    for step, rule in FILTERS:
        frame = frame[rule(frame)]
        counts.append((step, len(frame)))
    minimum = MINIMUM.get(state, MINIMUM_TARGET)
    if len(frame) < minimum:
        raise SystemExit(f"{state}: {len(frame)} filtered records, fewer than {minimum}")
    frame = frame.astype({column: "int64" for column in COLUMNS[1:]})
    frame = frame.sort_values(["SERIALNO", "SPORDER"], kind="mergesort").reset_index(drop=True)
    return frame, counts


def subsample(state, frame):
    """Positions numpy.sort(rng.permutation(N)[:20000]) for the two source states, every record otherwise."""
    if state not in SUBSAMPLE:
        return frame
    fips = STATES[state][1]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1, fips, 0]))
    positions = np.sort(rng.permutation(len(frame))[:SUBSAMPLE[state]])
    return frame.iloc[positions].reset_index(drop=True)


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n").encode()


def write_gzip(path, data):
    with path.open("wb") as handle, gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as packed:
        packed.write(data)


def build(raw, out):
    records = fetch(raw)
    out.mkdir(parents=True, exist_ok=True)
    topcode_table = pd.read_csv(raw / downloads()[-1][0], dtype=str)
    missing = [c for c in TOPCODE_COLUMNS if c not in topcode_table.columns]
    if missing:
        raise SystemExit(f"top-code file lacks the columns {missing}")
    topcodes = topcode_table.loc[topcode_table["STATEABRV"].isin(STATES), list(TOPCODE_COLUMNS)]
    if sorted(topcodes["STATEABRV"]) != sorted(STATES):
        raise SystemExit("top-code file does not list every state once")
    threshold = {row.STATEABRV: int(row.WAGTPCT) for row in topcodes.itertuples()}
    manifest = {"directory": DIRECTORY, "downloads": [], "adjinc": {}, "extracts": {}}
    counts, population = [], []
    for (name, url), state in zip(downloads(), list(STATES) + [None]):
        path = raw / name
        data = path.read_bytes()
        member = csv_member(path) if state else None
        manifest["downloads"].append({"file": name, "url": url, "bytes": len(data), "sha256": sha256(data),
                                      "retrieved_utc": records[name]["retrieved_utc"],
                                      "last_modified_utc": records[name].get("last_modified_utc"),
                                      "csv_member": member})
        if state is None:
            continue
        print(f"reading {name} ({member})", flush=True)
        frame = read_state(path, member)
        adjinc = frame["ADJINC"].dropna().unique()
        if len(adjinc) != 1 or frame["ADJINC"].isna().any():
            raise SystemExit(f"{state}: ADJINC is not one value per file: {adjinc}")
        manifest["adjinc"][state] = int(adjinc[0])
        filtered, steps = filter_state(state, frame)
        kept = subsample(state, filtered)
        counts += [{"state": state, "step": step, "records": n} for step, n in steps]
        counts.append({"state": state, "step": "kept", "records": len(kept)})
        if state in SUBSAMPLE:
            population.append({"sample": state, **describe(filtered, threshold[state])})
        content = csv_bytes(kept)
        write_gzip(extract_path(state, out), content)
        manifest["extracts"][extract_path(state, out).name] = {"records": len(kept), "sha256_uncompressed": sha256(content)}
    write_csv(out / "sample_counts.csv", counts)
    write_csv(out / "population.csv", population)
    write_csv(out / "topcodes.csv", topcodes.to_dict("records"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(manifest['extracts'])} extracts, sample_counts.csv, population.csv, topcodes.csv and "
          f"manifest.json to {out}")
    return manifest


def check(raw, out):
    """Verify the downloads against the committed manifest and rebuild the extracts into out."""
    from scripts.tolerance import compare_frames, read_table
    committed = json.loads((DATA / "manifest.json").read_text())
    fetch(raw)
    problems = []
    for entry in committed["downloads"]:
        digest = sha256((raw / entry["file"]).read_bytes())
        if digest != entry["sha256"]:
            problems.append(f"{entry['file']}: SHA-256 {digest} differs from the committed {entry['sha256']}")
    if problems:
        raise SystemExit("Downloads differ from acs/data/manifest.json:\n" + "\n".join(problems))
    rebuilt = build(raw, out)
    for state in STATES:
        new = gzip.decompress(extract_path(state, out).read_bytes())
        old = gzip.decompress(extract_path(state).read_bytes())
        if new != old:
            problems.append(f"{extract_path(state).name}: rebuilt content differs from the committed extract")
    for name in ("sample_counts.csv", "topcodes.csv"):
        if (out / name).read_bytes() != (DATA / name).read_bytes():
            problems.append(f"{name}: rebuilt file differs from the committed one")
    _, failures = compare_frames(read_table(out / "population.csv"), read_table(DATA / "population.csv"), ["sample"])
    problems += [f"population.csv: {failure}" for failure in failures]
    if rebuilt["adjinc"] != committed["adjinc"] or rebuilt["extracts"] != committed["extracts"]:
        problems.append("manifest.json: ADJINC values or extract hashes differ")
    for problem in problems:
        print("FAIL", problem)
    print(f"Checked {len(committed['downloads'])} downloads and {len(STATES)} extracts against acs/data/: "
          + ("no differences" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=Path("build") / "acs-raw", help="download directory")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--check", action="store_true", help="rebuild into a scratch directory and compare")
    args = parser.parse_args()
    if args.check:
        if args.out.resolve() == DATA.resolve():
            parser.error("--check writes to a scratch directory, not to acs/data")
        return check(args.raw.resolve(), args.out.resolve())
    build(args.raw.resolve(), args.out.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
