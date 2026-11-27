"""Regenerate every figure into an explicit directory and compare it with the reference copy.

Usage, from the repository root:
    python -m scripts.make_figures --out DIR

Each figure of the paper (PDF and PNG) is drawn from the saved outputs into DIR and compared with
figures/reference/ (PDF bytes and PNG pixels). Differences are reported, not treated as failures,
because font rasterization can vary across platforms. A failing script stops the run with a
nonzero exit status.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.image as image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (("figures.make_boundary_power", "boundary_power"), ("figures.make_sample_size", "sample_size"),
           ("figures.make_acs_loss", "acs_loss"))


def run(argv, cwd):
    subprocess.run(argv, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for module, stem in FIGURES:
        run([sys.executable, "-m", module, "--output", str(out / f"{stem}.pdf")], ROOT)
        reference = ROOT / "figures" / "reference"
        pdf = (out / f"{stem}.pdf").read_bytes() == (reference / f"{stem}.pdf").read_bytes()
        new, old = image.imread(out / f"{stem}.png"), image.imread(reference / f"{stem}.png")
        pixels = new.shape == old.shape and bool(np.array_equal(new, old))
        print(f"{stem + '.pdf':<44}{'identical to' if pdf else 'DIFFERS from'} figures/reference/{stem}.pdf; "
              f"png pixels {'identical' if pixels else 'DIFFER'}")
    print(f"Figures written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
