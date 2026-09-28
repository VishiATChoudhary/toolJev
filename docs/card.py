"""Draw docs/results_card.png, a 16:9 summary for sharing.   uv run python docs/card.py

Every number is from bench/RESULTS.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).parent
BG, INK, INK2, LINE, TILE = "#fcfcfb", "#0b0b0b", "#52514e", "#dcdad4", "#ffffff"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#b07800"

TILES = [
    ("83%", "right tool ranked first, retrieval + hosted Jev",
     "MCPToolBench++. Retrieval alone: 73%.\nLiveMCPBench 40% → 54%, When2Call 92% → 99%", BLUE),
    ("5.3x cheaper", "an agent routing 400 support tickets",
     "98.8% right for $0.063 with Jev in code.\nClaude Haiku routing each one itself: 99.3%, $0.33", ORANGE),
    ("99.7%", "right on the 97% of tickets Jev was ≥ 0.9 sure of",
     "400 banking77 tickets, no LLM per ticket.\nIts confidence is safe to act on in code", AQUA),
    ("77% fewer", "input tokens than Claude Code's own tool search",
     "612 MCP tools: right tool 89% vs 81% (lenient),\n28% cheaper, but slower", YELLOW),
]

fig = plt.figure(figsize=(16, 9), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")

ax.text(7, 81, "toolJev", fontsize=30, weight="bold", color=INK, va="center")
ax.text(7, 74, "Code Mode for MCP, where the sub-model is a calibrated decision model (Jev), not an LLM.",
        fontsize=15, color=INK2, va="center")
ax.text(7, 69.5, "Benchmarked with hosted Jev on MCPToolBench++, LiveMCPBench, When2Call and live Claude agents:",
        fontsize=15, color=INK2, va="center")

w, h, gap = 71, 26, 4
for i, (big, label, note, color) in enumerate(TILES):
    x = 7 + (i % 2) * (w + gap)
    y = 36 - (i // 2) * (h + gap)
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.8",
                                fc=TILE, ec=LINE, lw=1.2))
    ax.add_patch(FancyBboxPatch((x, y + 2), 0.9, h - 4, boxstyle="round,pad=0,rounding_size=0.45",
                                fc=color, ec="none"))
    ax.text(x + 4, y + h - 7, big, fontsize=34, weight="bold", color=INK, va="center")
    ax.text(x + 4, y + h - 14, label, fontsize=13.5, color=INK, va="center")
    ax.text(x + 4, y + 5.5, note, fontsize=11.5, color=INK2, va="center", linespacing=1.5)

ax.text(7, 2.5, "github.com/VishiATChoudhary/toolJev  ·  methods, sample sizes and every caveat in bench/RESULTS.md",
        fontsize=12, color=INK2, va="center")
fig.savefig(OUT / "results_card.png", dpi=100)
print("wrote", OUT / "results_card.png")
