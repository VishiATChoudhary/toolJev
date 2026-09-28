"""Render README charts from bench/results.

    uv run python -m bench.charts
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
OUT = Path(__file__).resolve().parents[1] / "docs"

# Reference palette, categorical slots 1-3 (validated all-pairs for CVD in light mode).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def abstention() -> None:
    s = json.loads((RESULTS / "summary.json").read_text())["abstention"]
    # Hosted Jev ran out of API credits after an 80-query MCPToolBench++ pilot, so it is
    # reported in the text rather than drawn beside full-benchmark bars.
    series = [("dense embedding, max cosine (MiniLM)", "dense-minilm", ORANGE),
              ("toolJev per-tool Nouls, local nanojev", "tooljev-encoder", BLUE)]
    benches = list(s)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    width = 0.26
    for i, (label, key, color) in enumerate(series):
        xs = [j + (i - 0.5) * (width + 0.02) for j in range(len(benches))]
        vals = [s[b].get(key, float("nan")) for b in benches]
        bars = ax.bar(xs, vals, width, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        for b, v in zip(bars, vals):
            if v == v:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, color=INK2)
    ax.axhline(0.5, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(-0.45, 0.505, "chance", fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(range(len(benches)), [b.replace(" (", "\n(") for b in benches], color=INK)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUROC, answerable vs not", color=INK)
    ax.set_title("Which signal knows when no tool fits?", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "abstention.png", dpi=160)


def gating() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    _style(ax)
    llm = None
    for backend, label, color in [("nanojev", "local nanojev", BLUE)]:
        f = RESULTS / f"gating_triage400_{backend}.json"
        if not f.exists():
            continue
        g = json.loads(f.read_text())
        llm = g["llm_accuracy"]
        rows = [r for r in g["rows"] if r["jev_coverage"] > 0]
        xs = [r["jev_coverage"] * 100 for r in rows]
        ys = [r["jev_accuracy_on_kept"] * 100 for r in rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.annotate(f"{label}: {ys[0]:.0f}% with no gate", (xs[0], ys[0]), textcoords="offset points",
                    xytext=(8, -4), ha="left", va="top", fontsize=8, color=INK2)
    if llm is not None:
        ax.axhline(llm * 100, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
        ax.text(100, llm * 100 + 0.5, f"Claude Haiku routing every ticket itself: {llm * 100:.1f}%",
                fontsize=8, color=INK2, ha="left", va="bottom")
    ax.set_xlim(102, 0)  # read left to right as "stricter confidence gate"
    ax.set_xlabel("share of 400 tickets Jev handles (confidence gate tightens to the right)", color=INK)
    ax.set_ylabel("accuracy on the tickets Jev handles (%)", color=INK)
    ax.set_title("Jev's confidence tracks its accuracy (local nanojev, 400 banking77 tickets)",
                 loc="left", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "gating.png", dpi=160)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    abstention()
    gating()
    print("wrote", OUT / "abstention.png", OUT / "gating.png")
