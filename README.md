# Affine Transfer for Conditional Quantiles with Heavy-Tailed Responses

[![checks](https://github.com/kirillsirik/affine-quantile-transfer/actions/workflows/checks.yml/badge.svg)](https://github.com/kirillsirik/affine-quantile-transfer/actions/workflows/checks.yml)
[![full reproduction](https://github.com/kirillsirik/affine-quantile-transfer/actions/workflows/full-reproduction.yml/badge.svg)](https://github.com/kirillsirik/affine-quantile-transfer/actions/workflows/full-reproduction.yml)

Kirill Sirik\
Department of Operations Research and Financial Engineering, Princeton University\
Senior thesis, November 2026. Advisor: Professor Jianqing Fan

- [Full paper](paper/Sirik_Affine_Quantile_Transfer_Full.pdf): Sections 1–6, the references and Appendices A–C (33 pages)
- [Paper without appendix](paper/Sirik_Affine_Quantile_Transfer_Paper.pdf): Sections 1–6 and the works they cite (24 pages)
- [Appendix](paper/Sirik_Affine_Quantile_Transfer_Appendix.pdf): the proofs and the design calculations (10 pages)

This repository holds the code, data and saved results behind every number in the paper. One study uses public-use microdata of the U.S. Census Bureau; every other data set is simulated.

## Reproducing the paper

You need Python 3.12, GNU make and Bash. From the repository root:

```sh
make setup        # create .venv with the pinned packages; set PYTHON=/path/to/python3.12 if needed
make check        # tests, the saved results, and every number of the paper (about 15 seconds)
make reproduce    # rerun every study and check every number again (about 20 minutes)
make figures      # redraw the three figures into build/figures/
```

[`paper/numbers.csv`](paper/numbers.csv) lists every number that the paper prints from a computation or from the data: the 682 numbers of its tables, text and figures, and 80 statements in words, such as "in more than half of its samples". Each row names the saved file the number comes from, the rows it uses, and the expression that produces it. `make check` recomputes every row from the saved results in this repository. `make reproduce` reruns every study into `build/reproduce/`, compares each output file with the saved one, and then recomputes every row from the new run. The design values that the paper states (sample sizes, replication counts, seeds and quantile levels) are the settings in the `config/` folders and the run records.

`make setup` installs the exact versions in `requirements.txt` (binary wheels, each checked against its hash): NumPy 2.3.5, SciPy 1.17.0, pandas 2.2.3 and Matplotlib 3.10.8. `make smoke` runs every study at a few replications, and `make help` lists all commands.

## Where each table and figure comes from

| Exhibit | Saved results | Code |
| --- | --- | --- |
| Table 1 | `closedform/reference/constants.csv` | `closedform/` |
| Figure 1 | `closedform/reference/boundary_curves.csv`, `boundary_power.csv` | `closedform/`; drawn by `figures/make_boundary_power.py` |
| Table 2 | `closedform/reference/boundary_power.csv`, `least_squares.csv` | `closedform/` |
| Table 3 | `singlequantile/reference/summary.csv`, `paired.csv` | `singlequantile/run_single.py` |
| Table 4 | `singlequantile/reference/summary.csv` | `singlequantile/run_single.py` |
| Figure 2 | `singlequantile/grid/reference/summary.csv`; the dashed line from `closedform/reference/single_quantile_population.csv` | `singlequantile/grid/run_simulations.py`; drawn by `figures/make_sample_size.py` |
| Table 5 | `singlequantile/grid/reference/raw_results.csv.gz` | `singlequantile/grid/run_simulations.py` |
| Table 6 | `multiquantile/reference/quartiles/replications.csv`, `multiquantile/reference/slope-test/test_rates.csv`; limits from `closedform/reference/constants.csv`, `quartile_approximation.csv` | `multiquantile/`, `closedform/` |
| Table 7 | `acs/reference/transfer_summary.csv`, `transfer_paired.csv`, `inputs_only_summary.csv` | `acs/run_acs.py` |
| Figure 3 | `acs/reference/transfer_summary.csv`, `inputs_only_summary.csv`, `benchmark.csv` | `acs/run_acs.py`; drawn by `figures/make_acs_loss.py` |

The numbers quoted in the text come from the same files and from a few others (for example `singlequantile/reference/activity.csv`, `closedform/reference/example_bound.csv` and `acs/reference/inputs_only_paired.csv`); `paper/numbers.csv` gives the file and the expression for each one. The figures in the PDF were typeset with the paper from the same data.

## Studies

| Study | In the paper | Folder | Data | Size | Run time |
| --- | --- | --- | --- | --- | ---: |
| Closed-form values | Section 4 (Tables 1 and 2, Figure 1) and Remark 3.4 | `closedform/` | Formulas | None | 1 s |
| Single-quantile study | Section 5.1 (Tables 3 and 4) | `singlequantile/` | Simulated | 2,000 replications for each of 3 error laws, with 3 departures each | 8 min |
| Single-quantile grid | Section 5.1 (Figure 2 and Table 5) | `singlequantile/grid/` | Simulated | 28 scenarios of 200 replications | 90 s |
| Quartile study and slope-test run | Section 5.2 (Table 6) | `multiquantile/` | Simulated | For each of 6 designs, 5,000 replications for the slope test, of which the first 600 also enter the prediction comparison | 40 s |
| ACS study | Section 5.3 (Table 7, Figure 3) | `acs/` | 2024 ACS public-use microdata of 10 states | 200 random splits of each of 8 target states | 10 min |

Run times are those of `make reproduce` on an Apple M2 Max with `WORKERS=4`. Each folder's README describes its design, its files and how to run it alone. `docs/methods.md` gives the statistical conventions (the linear programs, ties, Monte Carlo standard errors and which inputs are oracle quantities), and `docs/data.md` lists every data file and saved result.

## Data

The ACS study uses the U.S. Census Bureau's 2024 American Community Survey 1-Year Public Use Microdata Sample, person records of California, Texas and the eight least populous states. The paper uses California and the eight targets; the Texas records serve a sensitivity check with an alternate source that the repository keeps. The Census Bureau publishes these files for public use, with disclosure protection applied; they may be used for statistical purposes only, and this repository is not endorsed or certified by the Census Bureau. Cite them as: U.S. Census Bureau (2025). 2024 American Community Survey 1-Year Public Use Microdata Sample [data files]. Retrieved November 14, 2026 from https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/.

The repository holds extracts, not the raw files: `acs/data/` has the 76,436 records that the study uses, in 13 columns. `make acs-data` downloads the Census files, checks each against the SHA-256 recorded in `acs/data/manifest.json`, and rebuilds the extracts, which must equal the committed ones byte for byte; it is the only command that needs the network. `acs/data/README.md` is the data notice: the source, the citation, the columns, the filters and how the records may be used.

## Validation

Reruns are compared with the saved results under one rule: floats agree when |a − b| <= max(1e-9 × max(|a|, |b|), 1e-10), and integers, labels and generated text must match exactly. `make reproduce` has passed on the Apple M2 Max (macOS) that produced the saved results and on a native x86-64 Linux runner of GitHub Actions, where the full-reproduction workflow can be started by hand. The worker count does not change any result. Windows was not tested, and the code checks the numbers, not the proofs.

## Layout

| Path | Contents |
| --- | --- |
| `paper/` | The three PDFs and `numbers.csv` |
| `closedform/`, `singlequantile/`, `multiquantile/`, `acs/` | One folder per study: code, configuration, saved results in `reference/`, and a README |
| `figures/` | The scripts that draw the three figures, with reference copies |
| `scripts/` | The study runner, the comparison of reruns, the checks of the saved results and of `paper/numbers.csv` |
| `tests/` | Unit tests |
| `docs/` | `methods.md` and `data.md` |

## Acknowledgment

I acknowledge the use of Claude for assisting with simulation and replication code. All empirical results were verified by the author.

## Citing and license

`CITATION.cff` gives the citation of the thesis, which GitHub shows as "Cite this repository". The code, the documentation and the saved results are released under the MIT License in `LICENSE`. The three PDFs in `paper/` are not covered by it: © 2026 Kirill Sirik, all rights reserved. The ACS extracts are U.S. Census Bureau public-use data, described in `acs/data/README.md`.
