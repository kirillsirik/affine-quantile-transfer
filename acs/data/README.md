# ACS extracts: data notice

This repository uses U.S. Census Bureau data. It is not endorsed or certified by the Census Bureau, and the conclusions drawn from the data are the author's responsibility.

These are public-use records with disclosure protection applied by the Census Bureau. Use them for statistical purposes only.

## Source

U.S. Census Bureau, 2024 American Community Survey 1-Year Public Use Microdata Sample (PUMS), person records, CSV files, from the directory https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/:

| State | Role | File |
| --- | --- | --- |
| California | source | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pca.zip |
| Texas | alternate source | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_ptx.zip |
| Wyoming | primary target | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pwy.zip |
| Vermont | target | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pvt.zip |
| Alaska | target | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pak.zip |
| North Dakota | target | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pnd.zip |
| South Dakota | target | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_psd.zip |
| Delaware | target | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pde.zip |
| Rhode Island | target | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pri.zip |
| Montana | target | https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pmt.zip |

The state top-code values come from https://www2.census.gov/programs-surveys/acs/tech_docs/pums/top_coded_values/2024_pums_top_and_bottom_coded_values.csv. The data dictionary, which defines every variable, is https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2024.csv; it is not copied here.

The files were retrieved on November 14, 2026 (UTC). The server gave their last modification as December 4, 2025, which dates the release.

Citation: U.S. Census Bureau (2025). 2024 American Community Survey 1-Year Public Use Microdata Sample [data files]. Retrieved November 14, 2026 from https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/

`manifest.json` records, for every downloaded file, its URL, size in bytes, SHA-256, time of retrieval and the name of its CSV member, together with the value of ADJINC in each state file and the SHA-256 of the uncompressed content of each extract. `make acs-data` checks every download against it and rebuilds the extracts, which must equal these files byte for byte.

## What the extracts are

The files here are a derived extract, not the Census Bureau's product. Anyone who needs the data should download the original files above.

`acs/build_data.py` made them as follows.

1. From each state's person file it read 13 columns: SERIALNO, SPORDER, PWGTP, AGEP, SCHL, WKHP, WKWN, SEX, COW, MAR, PINCP, WAGP and ADJINC. Blank cells (not in universe) became missing values.
2. It kept the records that pass six filters, in this order: AGEP > 16; PINCP > 100; WKHP > 0; PWGTP >= 1; COW from 1 to 7; and SCHL, WKWN, SEX, MAR and WAGP not missing. `sample_counts.csv` gives the number of records after each step.
3. It sorted the records of each state by SERIALNO (as text), then SPORDER (as an integer).
4. For California and Texas it kept 20,000 records, at the positions `numpy.sort(rng.permutation(N)[:20000])` of the sorted file, with `rng = numpy.random.default_rng(numpy.random.SeedSequence([20261114, 1, FIPS, 0]))` and N the number of filtered records. For the eight other states it kept every filtered record.
5. It wrote each state's records, all 13 columns as integers except SERIALNO, to `<state>.csv.gz` (gzip without a timestamp).

| File | State | Records after the six filters | Records kept | ADJINC |
| --- | --- | ---: | ---: | ---: |
| `ca.csv.gz` | California | 202,998 | 20,000 | 1015250 |
| `tx.csv.gz` | Texas | 147,511 | 20,000 | 1015250 |
| `wy.csv.gz` | Wyoming | 3,198 | 3,198 | 1015250 |
| `vt.csv.gz` | Vermont | 3,835 | 3,835 | 1015250 |
| `ak.csv.gz` | Alaska | 3,496 | 3,496 | 1015250 |
| `nd.csv.gz` | North Dakota | 4,505 | 4,505 | 1015250 |
| `sd.csv.gz` | South Dakota | 5,130 | 5,130 | 1015250 |
| `de.csv.gz` | Delaware | 4,896 | 4,896 | 1015250 |
| `ri.csv.gz` | Rhode Island | 5,604 | 5,604 | 1015250 |
| `mt.csv.gz` | Montana | 5,772 | 5,772 | 1015250 |

Source: `acs/data/manifest.json`, `acs/data/sample_counts.csv`.

## Columns

The data dictionary gives the full definitions and codes.

| Column | Meaning |
| --- | --- |
| SERIALNO | Housing unit or group quarters identifier (text) |
| SPORDER | Person number within the housing unit |
| PWGTP | Person weight. Included but not used: the study's fits are unweighted. |
| AGEP | Age in years, top-coded |
| SCHL | Educational attainment, codes 1-24 |
| WKHP | Usual hours worked per week in the past 12 months; 99 means 99 or more |
| WKWN | Weeks worked in the past 12 months, 1-52 |
| SEX | 1 male, 2 female |
| COW | Class of worker: 1 and 2 private, 3 to 5 government, 6 and 7 self-employed |
| MAR | Marital status: 1 married, 2 to 5 other |
| PINCP | Total person income in the past 12 months, in dollars |
| WAGP | Wage or salary income in the past 12 months, in dollars, top-coded by state |
| ADJINC | Adjustment factor to 2024 dollars, with six implied decimals; one value per file |

Wages at or above a state's threshold WAGTPCT are recorded as that state's top-code value T_WAG; every such record in the extracts has this value. The extracts keep top-coded values as they are.

## Other files

| File | Content |
| --- | --- |
| `sample_counts.csv` | Records per state after reading (`raw`), after each filter, and kept |
| `population.csv` | The descriptive statistics of `acs/README.md` for every filtered record of California and of Texas, before subsampling, to compare with the 20,000-record subsamples |
| `topcodes.csv` | The rows of the top-code table for the ten states: STATE, STATEABRV, WAGTPCT (the wage threshold) and T_WAG (the value assigned at or above it) |
| `manifest.json` | The download record described above |

## Rebuilding

`make acs-data` downloads the files into `build/acs-raw/` (outside git), checks each download against `manifest.json`, rebuilds the extracts into `build/acs-data/`, and requires their uncompressed content to equal the files here byte for byte. It is the only command that needs the network.
