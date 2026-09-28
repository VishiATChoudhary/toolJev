"""Summarise agent benchmark runs: bench/results/agent_*.jsonl -> markdown tables.

    uv run python -m bench.agent.analyze

MCPToolBench++ runs are scored two ways. Strict: the agent called the benchmark's
gold tool. Lenient: it called the gold tool or a functional equivalent, under the
rules below, which were fixed before any scores were read and apply identically
to every condition. The large catalog adds real near-duplicates (LiveMCPBench
ships its own filesystem and stock-price servers), so strict scoring there
penalises correct behaviour.
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"

STOCK_PRICE = {"yahoo-finance_get_current_stock_price", "investor_get_ticker_data",
               "yfmcp_get_ticker_info", "Asset_Price_MCP_get_asset_price"}


def _action(tool: str) -> str:
    for prefix in ("playwright_", "puppeteer_"):
        if tool.startswith(prefix):
            return tool[len(prefix):]
    return tool


def equivalent(gold_server: str, gold_tool: str, server: str, tool: str) -> bool:
    if (server, tool) == (gold_server, gold_tool):
        return True
    if tool.endswith("_" + gold_tool) or tool == gold_tool:  # same tool, another server
        return True
    if gold_server in ("playwright", "puppeteer") and server in ("playwright", "puppeteer", "extras"):
        return _action(tool.split("_", 1)[-1] if server == "extras" else tool) == _action(gold_tool)
    if gold_tool == "get_stock_price_global_market" and tool in STOCK_PRICE:
        return True
    return False


def mcptb_table(path: Path) -> str:
    rows = [json.loads(l) for l in path.open()]
    by = defaultdict(list)
    for r in rows:
        r["lenient"] = any(equivalent(r["server"], r["tool"], c["server"], c["tool"])
                           for c in r["upstream_calls"])
        by[r["condition"]].append(r)
    out = ["| condition | n | right tool (strict) | right tool (lenient) | arg accuracy | median input tokens "
           "| total cost | mean LLM calls | median wall s | timeouts |", "|---" * 10 + "|"]
    for cond, v in by.items():
        aa = [r["arg_acc"] for r in v if r["arg_acc"] is not None]
        costs = [r["cost_usd"] for r in v if r["cost_usd"] is not None]  # a timed-out run reports none
        out.append(f"| {cond} | {len(v)} | {st.mean(r['tool_correct'] for r in v):.3f} | "
                   f"{st.mean(r['lenient'] for r in v):.3f} | {st.mean(aa) if aa else float('nan'):.3f} | "
                   f"{st.median(r['input_tokens'] for r in v):,.0f} | ${sum(costs):.2f} | "
                   f"{st.mean(r['llm_calls'] for r in v):.1f} | {st.median(r['wall_s'] for r in v):.0f} | "
                   f"{sum(r['timed_out'] for r in v)} |")
    return "\n".join(out)


def triage_table(path: Path) -> str:
    rows = [json.loads(l) for l in path.open()]
    by = defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    out = ["| condition | reps | accuracy (each rep) | mean accuracy | mean cost | mean LLM calls | mean wall s |",
           "|---" * 7 + "|"]
    for cond, v in by.items():
        accs = ", ".join(f"{r['accuracy']:.3f}" for r in v)
        out.append(f"| {cond} | {len(v)} | {accs} | {st.mean(r['accuracy'] for r in v):.3f} | "
                   f"${st.mean(r['cost_usd'] for r in v):.3f} | {st.mean(r['llm_calls'] for r in v):.1f} | "
                   f"{st.mean(r['wall_s'] for r in v):.0f} |")
    return "\n".join(out)


def main() -> None:
    for f in sorted(RESULTS.glob("agent_*.jsonl")):
        print(f"\n### {f.stem}\n")
        print(mcptb_table(f) if "mcptb" in f.name else triage_table(f))


if __name__ == "__main__":
    main()
