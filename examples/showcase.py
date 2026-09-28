"""The toolJev story in one terminal run: 612 tools behind two, then 400 tickets in one call.

    uv run python examples/showcase.py

Everything is real and local: the search is toolJev's own over a 612-tool catalog
(MCPToolBench++ + LiveMCPBench schemas), and the triage is one sandboxed execute
against an in-process support server, checked against banking77's labels.
No API key; the decision model is nanojev running locally.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
logging.disable(logging.WARNING)

from fastmcp import FastMCP  # noqa: E402

from tooljev.catalog import Catalog, ToolInfo  # noqa: E402
from tooljev.config import SandboxConfig, SearchConfig, ServerConfig  # noqa: E402
from tooljev.decider import make_decider  # noqa: E402
from tooljev.retrieve import Retriever  # noqa: E402
from tooljev.sandbox import Sandbox  # noqa: E402
from tooljev.search import Searcher  # noqa: E402

DATA = Path(__file__).parent / "data"
DIM, BOLD, GREEN, YELLOW, RED, CYAN, MAG, RESET = (
    "\033[2m", "\033[1m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[35m", "\033[0m")

# Demo queries. Search picks the right tool first about 62% of the time on
# MCPToolBench++ (bench/RESULTS.md); these show the mechanism, not the average.
QUERIES = [
    "driving directions from Boston to New York",
    "take a screenshot of the login button",
    "refund order 1234 on paypal",
    "write me a haiku about autumn",
]

QUEUES = {
    "activate_my_card": "activating a new card",
    "card_arrival": "when a card will arrive, card not delivered yet",
    "lost_or_stolen_card": "card lost or stolen",
    "exchange_rate": "exchange rates and currency conversion",
    "pin_blocked": "PIN blocked or locked",
    "top_up_failed": "top-up or adding money failed",
    "transfer_not_received_by_recipient": "transfer sent but recipient has not received it",
    "cash_withdrawal_charge": "fee charged for a cash or ATM withdrawal",
    "request_refund": "asking for a refund",
    "terminate_account": "closing or deleting the account",
}

CODE = '''tickets = await mcp.support.list_tickets()
answers = await jev.map([t["body"] for t in tickets],
                        {"queue": jev.Choice("Which queue handles this?", QUEUES)},
                        min_confidence=0.8)
unsure = []
for t, a in zip(tickets, answers):
    if a["confident"]:
        await mcp.support.route_ticket(ticket_id=t["id"], queue=a["queue"]["choice"])
    else:
        unsure.append(t["id"])
FINAL({"routed": len(tickets) - len(unsure), "unsure": unsure})'''


class StaticCatalog:
    """The Searcher's view of a catalog, loaded from a JSON snapshot instead of live servers."""

    def __init__(self, raw: dict):
        self.servers = {n: ServerConfig(n, None, description=v["description"]) for n, v in raw.items()}
        self._tools = {n: [ToolInfo(n, t["name"], t["description"], t["input_schema"] or {})
                           for t in v["tools"]] for n, v in raw.items()}

    async def refresh(self, force: bool = False) -> None:
        pass

    def tools(self, server: str | None = None) -> list[ToolInfo]:
        return self._tools.get(server, []) if server else [t for ts in self._tools.values() for t in ts]

    def server_names(self) -> list[str]:
        return list(self._tools)


def support_server(tickets: list[dict], routed: dict) -> FastMCP:
    s = FastMCP("support")

    @s.tool
    def list_tickets() -> list[dict]:
        """List open customer support tickets."""
        return [{"id": t["id"], "body": t["body"]} for t in tickets]

    @s.tool
    def route_ticket(ticket_id: int, queue: str) -> dict:
        """Route a ticket to a support queue."""
        routed[ticket_id] = queue
        return {"ok": True}

    return s


def say(text: str = "", pause: float = 0.0) -> None:
    print(text, flush=True)
    if pause:
        time.sleep(pause)


def highlight(code: str) -> str:
    code = re.sub(r"(\"[^\"]*\")", f"{GREEN}\\1{RESET}", code)
    code = re.sub(r"\b(await|for|in|if|else|zip|len)\b", f"{MAG}\\1{RESET}", code)
    code = re.sub(r"\b(mcp|jev)\.", f"{CYAN}\\1{RESET}.", code)
    return "\n".join(f"  {DIM}│{RESET} {line}" for line in code.splitlines())


async def spin(label: str) -> None:
    frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
    t0 = time.perf_counter()
    i = 0
    try:
        while True:
            print(f"\r  {CYAN}{frames[i % len(frames)]}{RESET} {label} {DIM}{time.perf_counter() - t0:4.1f} s{RESET}",
                  end="", flush=True)
            i += 1
            await asyncio.sleep(0.1)
    finally:
        print("\r\033[2K", end="", flush=True)


def bar(p: float, width: int = 14) -> str:
    n = round(p * width)
    return f"{GREEN if p >= 0.5 else DIM}{'█' * n}{DIM}{'░' * (width - n)}{RESET}"


async def main() -> None:
    catalog_raw = json.loads((DATA / "catalog.json").read_text())
    tickets = json.loads((DATA / "tickets.json").read_text())
    n_tools = sum(len(v["tools"]) for v in catalog_raw.values())

    # Load models before the show starts, so the timings below are steady-state.
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    say(f"{DIM}loading local models (nanojev encoder + MiniLM)…{RESET}")
    decider = make_decider("nanojev", kind="encoder", dtype="float16")
    await decider.warmup()
    catalog = StaticCatalog(catalog_raw)
    searcher = Searcher(catalog, decider, SearchConfig(max_tools=2), Retriever())
    await searcher.search("warm up")
    print("\033[2J\033[H", end="", flush=True)  # clear the loading line

    say(f"\n{BOLD}toolJev{RESET}  {DIM}Code Mode MCP gateway · the sub-model is Jev, not an LLM{RESET}")
    say(f"{DIM}{n_tools} tools across {len(catalog_raw)} MCP servers. Your agent sees two:{RESET} "
        f"{CYAN}search{RESET} and {CYAN}execute{RESET}\n", 1.2)

    for q in QUERIES:
        t0 = time.perf_counter()
        r = await searcher.search(q)
        ms = (time.perf_counter() - t0) * 1000
        say(f"{BOLD}search{RESET}(\"{q}\")  {DIM}{ms:.0f} ms{RESET}")
        if "warning" in r:
            say(f"  {YELLOW}⚠ nothing in {n_tools} tools fits{RESET}  "
                f"{DIM}(Jev: {r['in_catalog']:.2f}) · the agent answers itself{RESET}\n", 1.0)
            continue
        first, *rest = [t["path"] for t in r["tools"]]
        say(f"  {GREEN}→{RESET} {BOLD}{first}{RESET}  {DIM}also: {', '.join(rest)}{RESET}")
        say(f"  {DIM}fits {bar(r['in_catalog'], 10)} {r['in_catalog']:.2f}{RESET}\n", 0.8)

    say(f"{BOLD}execute{RESET}  {DIM}# route {len(tickets)} support tickets, in one call{RESET}")
    say(highlight(CODE), 1.5)

    routed: dict[int, str] = {}
    async with Catalog([ServerConfig("support", support_server(tickets, routed))]) as live, \
            Sandbox(live, decider, SandboxConfig()) as sandbox:
        code = f"QUEUES = {json.dumps(QUEUES)}\n{CODE}"
        print()
        spinner = asyncio.create_task(spin(f"sandbox running · jev.map over {len(tickets)} tickets"))
        t0 = time.perf_counter()
        r = await sandbox.execute(code)
        secs = time.perf_counter() - t0
        spinner.cancel()
        await asyncio.gather(spinner, return_exceptions=True)
    if not r["ok"]:
        say(f"{RED}{r['error']}{RESET}")
        sys.exit(1)

    gold = {t["id"]: t["intent"] for t in tickets}
    right = sum(routed[i] == gold[i] for i in routed)
    unsure = r["result"]["unsure"]
    calls = r["calls"]
    say(f"  {GREEN}✓{RESET} {BOLD}{len(routed)}{RESET} routed in code, where Jev was ≥ 0.8 confident"
        f"   {DIM}{secs:.1f} s, {sum(c['n'] for c in calls.values())} host calls{RESET}")
    say(f"  {YELLOW}↩{RESET} {BOLD}{len(unsure)}{RESET} unsure ones returned to the agent to read", 0.8)
    say(f"\n  {DIM}checked against banking77 labels:{RESET} the confident {len(routed)} were "
        f"{BOLD}{GREEN}{right / len(routed):.1%}{RESET} right")
    say(f"  {DIM}one execute call for the whole batch · no LLM tokens spent per ticket{RESET}\n", 2.5)


if __name__ == "__main__":
    asyncio.run(main())
