"""Mean squared prediction error against the target sample size, under t3 errors (Figure 2).

Usage, from the repository root:
    python -m figures.make_sample_size --output FILE [--grid DIR] [--closed-form DIR]

Reads summary.csv of the single-quantile grid (default singlequantile/grid/reference/) and
single_quantile_population.csv of the closed-form values (default closedform/reference/). Each
panel shows, for one departure omega, the mean squared error of the 0.90-quantile prediction of the
transfer and target-only fits at n = 50, 100, 200 and 400 target observations (known scores,
1,000 source observations, 200 replications per point), with bars of two Monte Carlo standard
errors either side; both axes are logarithmic. The dashed line in the omega = 1 panel is the
population error of the ideal-input transfer fit. Writes FILE (PDF) and a PNG at 200 dpi next to
it. The reference copies are figures/reference/sample_size.pdf and sample_size.png.
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
TEXT, SECONDARY = "#26292e", "#6b7075"
METHODS = (("transfer", "Transfer fit", "#2a78d6", "-", "o"), ("target_only", "Target-only fit", "#eb6834", "--", "s"))
PANELS = ((0.0, "No omitted effect ($\\omega = 0$)"), (1.0, "Omitted effect ($\\omega = 1$)"))
SIZES = (50, 100, 200, 400)
YTICKS = (0.05, 0.1, 0.2, 0.5, 1)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid", type=Path, default=PROJECT / "singlequantile" / "grid" / "reference")
    parser.add_argument("--closed-form", type=Path, default=PROJECT / "closedform" / "reference")
    args = parser.parse_args()
    summary = pd.read_csv(args.grid / "summary.csv")
    summary = summary[(summary.experiment == "known") & (summary.noise == "t3") & (summary.tau == 0.9)]
    population = pd.read_csv(args.closed_form / "single_quantile_population.csv")
    floor = population[(population.law == "t3") & (population.omega == 1.0)].squared_error.item()
    with plt.rc_context({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
        for ax, (omega, title) in zip(axes, PANELS):
            for method, label, color, style, marker in METHODS:
                rows = summary[(summary.delta == omega) & (summary.method == method)].set_index("nT").loc[list(SIZES)]
                ax.errorbar(SIZES, rows.mse_mean, yerr=2 * rows.mse_mcse, color=color, linestyle=style, linewidth=1.6,
                            marker=marker, markersize=5, elinewidth=1.0, capsize=0, label=label)
            if omega == 1.0:
                ax.axhline(floor, color=SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
                ax.annotate(f"population error {floor:.3f}", xy=(SIZES[-1], floor), xytext=(0, -5),
                            textcoords="offset points", ha="right", va="top", fontsize=8, color=SECONDARY)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set(xticks=SIZES, xlim=(42, 470), yticks=YTICKS, ylim=(0.04, 1.5))
            ax.set_xticklabels([str(n) for n in SIZES])
            ax.set_yticklabels([f"{y:g}" for y in YTICKS])
            ax.minorticks_off()
            ax.grid(axis="y", alpha=0.15)
            ax.set_title(title, color=TEXT)
            ax.set_xlabel("Target sample size n")
        axes[0].set_ylabel("Mean squared prediction error")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, handlelength=2.6)
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.27, top=0.9, wspace=0.12)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(args.output.with_suffix(".png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
