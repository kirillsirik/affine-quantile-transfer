"""Limiting power of the 5% slope test against the size of a scale departure, relative to the boundary.

Usage, from the repository root:
    python -m figures.make_boundary_power --output FILE [--results DIR]

Reads boundary_curves.csv and boundary_power.csv of a closed-form results directory (default
closedform/reference/). Each panel shows, for d = 1, 2, 5 and 10 inputs, the limiting power of the
test at relative departure r, where r = 1 is the departure at which sharing the slopes stops
improving prediction at an outer quartile. Writes FILE (PDF) and a PNG at 200 dpi next to it. The
reference copies are figures/reference/boundary_power.pdf and boundary_power.png.
"""
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
TEXT, SECONDARY = "#26292e", "#6b7075"
CURVES = ((1, "#6CAED3"), (2, "#4788AC"), (5, "#1D6284"), (10, "#003B5B"))
PANELS = (("normal", "Normal errors"), ("t3", "$t_3$ errors"))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=PROJECT / "closedform" / "reference")
    args = parser.parse_args()
    curves = pd.read_csv(args.results / "boundary_curves.csv")
    boundary = pd.read_csv(args.results / "boundary_power.csv")
    boundary = boundary[boundary.criterion == "outer_quartile"].set_index(["law", "d"]).power
    with plt.rc_context({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
        for ax, (law, title) in zip(axes, PANELS):
            ax.axvline(1.0, color=SECONDARY, linewidth=0.7)
            for d, color in CURVES:
                ax.plot(curves.relative_size, curves[f"{law}_d{d}"], color=color, linewidth=1.6, label=f"d = {d}")
                ax.plot([1.0], [boundary[law, d]], marker="o", markersize=4.5, color=color, linestyle="none")
                if d == 10:
                    ax.annotate("d = 10", xy=(1.0, boundary[law, d]), xytext=(-6, 4), textcoords="offset points",
                                ha="right", va="bottom", fontsize=8, color=TEXT)
                if d == 1:
                    ax.annotate("d = 1", xy=(1.0, boundary[law, d]), xytext=(6, -5), textcoords="offset points",
                                ha="left", va="top", fontsize=8, color=TEXT)
            ax.text(0.3, 0.17, "sharing\nhelps", ha="center", va="bottom", color=SECONDARY, fontsize=8, linespacing=1.1)
            ax.text(2.05, 0.03, "sharing\nhurts", ha="center", va="bottom", color=SECONDARY, fontsize=8, linespacing=1.1)
            ax.set(xlim=(0, 3), ylim=(0, 1), xticks=[0, 1, 2, 3], yticks=[0, 0.25, 0.5, 0.75, 1])
            ax.set_title(title, color=TEXT)
            ax.set_xlabel("Scale departure relative to the boundary")
            ax.grid(axis="y", alpha=0.15)
        axes[0].set_ylabel("Limiting power of the 5% test")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.27, top=0.9, wspace=0.12)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(args.output.with_suffix(".png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
