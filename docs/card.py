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
    ("61% fewer", "input tokens than Claude Code's own tool search",
     "same task success (0.81) over 612 MCP tools,\n23% lower cost, but slower", BLUE),
    ("18% → 62%", "right tool first: Jev alone vs retrieval first",
     "MCPToolBench++, 1,509 queries. So retrieval\npicks the tools and Jev judges them", ORANGE),
    ("98.3%", "accuracy on the 29% of tickets Jev was surest of",
     "400 banking77 tickets. Claude Haiku routing\nall of them itself: 99.3%", AQUA),
    ("3.6-5.2x", "cheaper bulk triage, done in code with jev.map",
     "400 tickets in one execute, but 12-25 points\nless accurate with the local model", YELLOW),
]

fig = plt.figure(figsize=(16, 9), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")

ax.text(7, 81, "toolJev", fontsize=30, weight="bold", color=INK, va="center")
ax.text(7, 74, "Code Mode for MCP, where the sub-model is a calibrated decision model (Jev), not an LLM.",
        fontsize=15, color=INK2, va="center")
ax.text(7, 69.5, "I benchmarked it on MCPToolBench++, LiveMCPBench, When2Call and live Claude agents:",
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
