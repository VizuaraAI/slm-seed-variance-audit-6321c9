"""Regenerate the paper's two figures from your own results.

The kit ships illustrative versions of both figures. This module replaces them
with the real thing once you have runs on disk, so the figure in the paper and
the numbers in the table always come from the same JSON files. Keeping that
link mechanical is how you avoid the classic late-stage panic where the plot
and the table disagree and nobody remembers which is current.

Figure 2a: mean accuracy per model scale with across-seed error bars.
Figure 2b: every harvested claim as a lollipop at its SES value, with the
           seed-noise floor drawn at SES = 1.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # no display on a remote box
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend call)
import matplotlib.transforms as transforms  # noqa: E402
import numpy as np  # noqa: E402

from . import modeling  # noqa: E402

PARAMS_BY_NAME: Dict[str, float] = {m["name"]: m["params_m"] for m in modeling.MODEL_LADDER}


def plot_seed_spread(seed_stats: Sequence[Dict], out_path: str) -> str:
    """Figure 2a: accuracy with across-seed error bars, ordered by model size."""
    tasks = sorted({row["task"] for row in seed_stats})
    models = sorted(
        {row["model"] for row in seed_stats},
        key=lambda name: PARAMS_BY_NAME.get(name, float("inf")),
    )

    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=200)
    width = 0.8 / max(1, len(tasks))
    positions = np.arange(len(models), dtype=float)

    for task_index, task in enumerate(tasks):
        means, errors = [], []
        for model in models:
            # Average the per-template cells, and take the largest sigma across
            # templates as the error bar: the conservative choice again.
            cells = [
                row
                for row in seed_stats
                if row["model"] == model and row["task"] == task
            ]
            means.append(float(np.mean([c["mean"] for c in cells])) if cells else np.nan)
            errors.append(float(np.max([c["std"] for c in cells])) if cells else 0.0)
        offset = (task_index - (len(tasks) - 1) / 2) * width
        ax.bar(
            positions + offset,
            means,
            width=width * 0.9,
            yerr=errors,
            capsize=3,
            label=task,
            edgecolor="white",
            linewidth=0.6,
        )

    labels = []
    for model in models:
        params = PARAMS_BY_NAME.get(model)
        short = model.split("/")[-1]
        labels.append(f"{short}\n{params:.0f}M" if params else short)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Seed spread by model scale", fontsize=10)
    ax.yaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=len(tasks))

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_claim_audit(
    audit_rows: Sequence[Dict], out_path: str, floor: float = 1.0, strong: float = 2.0
) -> str:
    """Figure 2b: published gaps in units of seed sigma."""
    rows = [r for r in audit_rows if r.get("ses") is not None]
    if not rows:
        raise ValueError("no scored claims to plot; fill in experiments/claims.csv")
    rows = sorted(rows, key=lambda r: float(r["ses"]))

    ses = [float(r["ses"]) for r in rows]
    labels = [f"{r['claim_id']} ({r['benchmark']})" for r in rows]
    y = np.arange(len(rows), dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 0.34 * len(rows) + 1.6), dpi=200)
    ax.axvspan(0, floor, color="0.92", zorder=0)
    for index, value in enumerate(ses):
        below = value < floor
        colour = "0.6" if below else ("#c2541a" if value >= strong else "0.35")
        ax.hlines(y[index], 0, value, color=colour, linewidth=1.2, zorder=2)
        ax.plot(value, y[index], "o", color=colour, markersize=5, zorder=3)

    ax.axvline(floor, linestyle="--", color="0.3", linewidth=1.0)
    ax.axvline(strong, linestyle=":", color="0.3", linewidth=1.0)
    # Blended transform: x in data units (pinned to the threshold lines),
    # y in axes units (pinned just inside the top of the plot area).
    label_pos = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(floor, 0.97, " seed noise floor", fontsize=7, color="0.3",
            transform=label_pos, va="top", ha="left")
    ax.text(strong, 0.97, f" SES = {strong:g}", fontsize=7, color="0.3",
            transform=label_pos, va="top", ha="left")
    ax.set_ylim(-0.7, len(rows) - 0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Reported gap in units of seed standard deviation (SES)")
    ax.set_xlim(left=0)
    ax.set_title("Published gaps against the measured seed-noise floor", fontsize=10)
    ax.xaxis.grid(True, linewidth=0.4, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _read_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the paper figures")
    parser.add_argument(
        "--results",
        default="results",
        help="directory written by python -m src.analysis",
    )
    parser.add_argument(
        "--figures",
        default="figures",
        help="where to write the PNGs",
    )
    args = parser.parse_args()

    seed_stats_path = os.path.join(args.results, "seed_stats.json")
    if os.path.exists(seed_stats_path):
        path = plot_seed_spread(
            _read_json(seed_stats_path),
            os.path.join(args.figures, "fig2a_seed_spread.png"),
        )
        print(f"[plots] wrote {path}")
    else:
        print(f"[plots] no {seed_stats_path}; run the analysis first")

    audit_path = os.path.join(args.results, "claim_audit.json")
    if os.path.exists(audit_path):
        path = plot_claim_audit(
            _read_json(audit_path),
            os.path.join(args.figures, "fig2b_claim_audit.png"),
        )
        print(f"[plots] wrote {path}")
    else:
        print(f"[plots] no {audit_path}; fill in the claim database first")


if __name__ == "__main__":
    main()
