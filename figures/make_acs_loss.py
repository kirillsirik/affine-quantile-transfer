"""Held-out check loss of the four methods of the ACS study and of the reduced fit, relative to the full-sample benchmark.

Usage, from the repository root:
    python -m figures.make_acs_loss --output FILE [--results DIR]

Reads transfer_summary.csv, inputs_only_summary.csv and benchmark.csv of an ACS results directory (default acs/reference/)
and plots excess_pct, the held-out check loss at the 0.90 level above the ten-fold cross-validated
benchmark L_ref, in percent, against the labeled sample size, one panel per target state, source
California. Bars span two Monte Carlo standard errors either side. The first panel also labels each
line at its right end, with short names. Writes FILE (PDF) and a PNG at 200 dpi next to it. The reference copies are figures/reference/acs_loss.pdf and acs_loss.png.
"""
import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from acs.variables import STATES

PROJECT = Path(__file__).resolve().parents[1]
TEXT = "#26292e"
METHODS = (("transfer", "Transfer fit", "#2a78d6", "-", "o"),
           ("reduced", "Reduced fit", "#8a5cb8", (0, (4, 1.5, 1, 1.5, 1, 1.5)), "v"),
           ("target_only", "Target-only fit", "#eb6834", "--", "s"),
           ("pooled", "Pooled fit", "#1baf7a", "-.", "^"),
           ("source_only", "Source prediction", "#eda100", ":", "D"))
SHORT_NAMES = ("Transfer", "Reduced", "Target-only", "Pooled", "Source")


def place_end_labels(fig, ax, curves, names):
    """Label each curve at its right end, in the text color, without covering a line, marker or label.

    Labels are right-aligned just left of the last point and placed in ascending order of the curves'
    last values; each moves vertically, one point at a time, to the nearest position where its box
    touches no curve and no earlier label.
    """
    renderer = fig.canvas.get_renderer()
    dense = []
    for x, y in curves:
        grid = np.exp(np.linspace(np.log(x[0]), np.log(x[-1]), 400))
        dense.append(ax.transData.transform(np.column_stack((grid, np.interp(np.log(grid), np.log(x), y)))))
    dense = np.vstack(dense)
    frame = ax.get_window_extent(renderer)
    placed = []
    for index in np.argsort([y[-1] for _, y in curves], kind="stable"):
        x, y = curves[index]
        text = ax.annotate(names[index], xy=(x[-1], y[-1]), xytext=(-6, 0), textcoords="offset points", ha="right",
                           va="center", fontsize=7, color=TEXT)
        for step in sorted(range(-45, 46), key=abs):
            text.set_position((-6, step))
            box = text.get_window_extent(renderer)
            hits = ((dense[:, 0] >= box.x0 - 3) & (dense[:, 0] <= box.x1 + 3)
                    & (dense[:, 1] >= box.y0 - 1.5) & (dense[:, 1] <= box.y1 + 1.5))
            if (not hits.any() and not any(box.overlaps(other) for other in placed)
                    and frame.y0 <= box.y0 and box.y1 <= frame.y1):
                break
        else:
            raise RuntimeError(f"no free place for the label {names[index]!r}")
        placed.append(text.get_window_extent(renderer))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, default=PROJECT / "acs" / "reference")
    args = parser.parse_args()
    cfg = json.loads((args.results / "metadata.json").read_text())["configuration"]
    summary = pd.read_csv(args.results / "transfer_summary.csv")
    summary = summary[summary.source == cfg["source"]]
    # the reduced fit: the target fit on the transfer inputs without the source prediction
    reduced = pd.read_csv(args.results / "inputs_only_summary.csv").assign(method="reduced", source=cfg["source"])
    summary = pd.concat([summary, reduced[[c for c in summary.columns if c in reduced.columns]]], ignore_index=True)
    benchmark = pd.read_csv(args.results / "benchmark.csv").set_index("target").L_ref
    summary = summary.assign(bar=200 * summary.mcse_loss / summary.target.map(benchmark))
    targets = [cfg["primary_target"]] + [t for t in cfg["targets"] if t != cfg["primary_target"]]
    sizes = cfg["labeled_sizes"]
    low = 5 * np.floor((summary.excess_pct - summary.bar).min() / 5 - 1e-9) - 5
    high = 10 * np.ceil((summary.excess_pct + summary.bar).max() / 10)
    with plt.rc_context({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 4, figsize=(6.4, 4.3), sharex=True, sharey=True)
        for ax, target in zip(axes.flat, targets):
            mine = summary[summary.target == target]
            curves = []
            for method, label, color, style, marker in METHODS:
                rows = mine[mine.method == method].set_index("n").loc[sizes]
                ax.errorbar(sizes, rows.excess_pct, yerr=rows.bar, color=color, linestyle=style, linewidth=1.6,
                            marker=marker, markersize=5, elinewidth=1.0, capsize=0, label=label)
                curves.append((np.array(sizes, dtype=float), rows.excess_pct.to_numpy()))
            ax.set_xscale("log")
            ax.set(xticks=sizes, xlim=(85, 940), ylim=(low, high))
            ax.set_xticklabels([str(n) for n in sizes])
            ax.minorticks_off()
            ax.grid(axis="y", alpha=0.15)
            ax.set_title(STATES[target][0], color=TEXT)
            if target == targets[0]:
                first, first_curves = ax, curves
        fig.supxlabel("Labeled sample size n", fontsize=9, y=0.135)
        fig.supylabel("Held-out check loss above the full-sample benchmark (%)", fontsize=9, x=0.015)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, handlelength=2.6, columnspacing=1.2)
        fig.subplots_adjust(left=0.1, right=0.99, bottom=0.24, top=0.94, wspace=0.12, hspace=0.32)
        place_end_labels(fig, first, first_curves, SHORT_NAMES)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, metadata={"CreationDate": None, "ModDate": None})
        fig.savefig(args.output.with_suffix(".png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
