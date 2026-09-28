"""Draw docs/architecture.{svg,png}.   uv run python docs/diagram.py"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent
BG, INK, INK2, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c2"
BLUE, BLUE_BG = "#2a78d6", "#e8f0fb"  # Jev
AQUA, AQUA_BG = "#1baf7a", "#e5f6ef"  # sandbox
ORANGE, ORANGE_BG = "#eb6834", "#fdece5"  # retrieval
PANEL = "#f3f2ee"

fig = plt.figure(figsize=(16, 9), facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def box(x, y, w, h, fc, ec, lw=1.4, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw))


def text(x, y, s, size=12, color=INK, weight="normal", ha="left", va="center", family="sans-serif"):
    ax.text(x, y, s, fontsize=size, color=color, weight=weight, ha=ha, va=va, family=family)


def arrow(x1, y1, x2, y2, color=INK2, lw=1.6, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16,
                                 color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"))


# title
text(6, 84, "toolJev", 26, weight="bold")
text(26.5, 84.2, "a Code Mode MCP gateway where the sub-model is Jev, not an LLM", 15, INK2)

# agent
box(6, 34, 22, 30, PANEL, LINE)
text(17, 58, "Your agent", 15, weight="bold", ha="center")
text(17, 53.5, "Claude Code, or any", 11, INK2, ha="center")
text(17, 50.5, "MCP client", 11, INK2, ha="center")
text(17, 44, "sees 2 tools,", 12, ha="center")
text(17, 40.5, "not 612", 12, ha="center", weight="bold")

# gateway frame
box(36, 10, 84, 66, "#ffffff", LINE, lw=1.2)
text(39, 72.5, "toolJev gateway", 13, INK2, weight="bold")

# search lane
text(39, 66, "search(query)", 13, weight="bold", family="monospace")
box(39, 50, 22, 12, ORANGE_BG, ORANGE)
text(50, 58.3, "Retrieval", 12, weight="bold", ha="center")
text(50, 54.8, "BM25 + MiniLM, fused", 10, INK2, ha="center")
text(50, 52.3, "top 15 of 612 · 5 ms", 10, INK2, ha="center")
box(66, 50, 22, 12, BLUE_BG, BLUE)
text(77, 58.3, "Jev: does it fit?", 12, weight="bold", ha="center")
text(77, 54.8, "one Noul per candidate", 10, INK2, ha="center")
text(77, 52.3, "calibrated probability", 10, INK2, ha="center")
box(93, 50, 23, 12, PANEL, LINE)
text(104.5, 58.3, "Shortlist", 12, weight="bold", ha="center")
text(104.5, 54.8, "Python signatures,", 10, INK2, ha="center")
text(104.5, 52.3, "or \"nothing fits\"", 10, INK2, ha="center")
arrow(61.3, 56, 65.7, 56)
arrow(88.3, 56, 92.7, 56)

# execute lane
text(39, 44, "execute(code, session=)", 13, weight="bold", family="monospace")
box(39, 14, 77, 26, AQUA_BG, AQUA)
text(42, 36.5, "Monty sandbox  ·  no files, network or env  ·  state kept per session, like an RLM REPL",
     10.5, INK2)
box(42, 17.5, 33, 15.5, "#ffffff", AQUA, lw=1.0)
text(44, 29.8, "mcp.<server>.<tool>(...)", 11.5, weight="bold", family="monospace")
text(44, 25.8, "every upstream tool, as code", 10, INK2)
text(44, 22.8, "args checked against its schema", 10, INK2)
text(44, 19.8, "results stay in variables", 10, INK2)
box(80, 17.5, 33, 15.5, "#ffffff", BLUE, lw=1.0)
text(82, 29.8, "jev.choice / noul / map", 11.5, weight="bold", family="monospace", color=BLUE)
text(82, 25.8, "typed decisions, ~100 ms, no text", 10, INK2)
text(82, 22.8, "where an RLM would call an LLM", 10, INK2)
text(82, 19.8, "min_confidence flags unsure items", 10, INK2)

# agent <-> gateway
arrow(28.3, 56, 38.5, 56, rad=0.0)
arrow(28.3, 42, 38.5, 30, rad=0.0)
text(29, 62, "which tools?", 10, INK2)
text(29, 33.5, "code", 10, INK2)

# upstream servers
text(128, 72.5, "Upstream MCP servers", 13, INK2, weight="bold")
names = ["github", "google_maps", "paypal", "filesystem", "playwright", "tavily", "…", "79 servers"]
for i, n in enumerate(names):
    y = 66 - i * 6.6
    last = i == len(names) - 1
    box(128, y - 2.4, 26, 4.8, PANEL if not last else "#ffffff", LINE, lw=1.0, r=1.0)
    text(141, y, n, 10.5, INK if not last else INK2, ha="center",
         family="monospace" if not last else "sans-serif", weight="bold" if last else "normal")
arrow(116.3, 27, 127.5, 27, color=AQUA, style="<|-|>")
text(121.9, 30, "tool calls", 10, INK2, ha="center")

# backend strip
box(36, 2.5, 84, 5.2, BLUE_BG, BLUE, lw=1.0, r=1.0)
text(78, 5.1, "Jev backend:  hosted TypeSafe Jev  or  local nanojev  ·  one decide(state, questions) API",
     11, BLUE, ha="center")

fig.savefig(OUT / "architecture.svg")
fig.savefig(OUT / "architecture.png", dpi=110)
print("wrote", OUT / "architecture.svg", OUT / "architecture.png")
