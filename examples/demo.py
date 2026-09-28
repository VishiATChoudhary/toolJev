"""Drive toolJev as an MCP client would, over stdio, against the example config.

    uv run python examples/demo.py [--config examples/config.toml]
"""

import argparse
import asyncio
import json
import sys
import time

from fastmcp import Client

TRIAGE = '''
tickets = await mcp.tickets.list_tickets()
answers = await jev.map([t["body"] for t in tickets], {
    "team": jev.Choice("Which team should handle this ticket", {
        "infra": "outages, errors, downtime, latency",
        "billing": "charges, invoices, refunds",
        "product": "feature requests, typos, design",
    }),
    "incident": jev.Noul("This describes a live production incident affecting customers"),
})
paged = []
for t, a in zip(tickets, answers):
    await mcp.tickets.assign_ticket(ticket_id=t["id"], team=a["team"]["choice"])
    if a["incident"]["noul"] > 0.5:
        await mcp.tickets.page_oncall(ticket_id=t["id"], summary=t["body"][:80])
        paged.append(t["id"])
FINAL({
    "routed": {t["id"]: [a["team"]["choice"], round(a["team"]["confidence"], 2)] for t, a in zip(tickets, answers)},
    "paged": paged,
})
'''


async def main(config: str) -> None:
    gateway = {"mcpServers": {"tooljev": {
        "command": sys.executable, "args": ["-m", "tooljev.server", "--config", config]}}}
    async with Client(gateway) as c:
        print("gateway tools:", [t.name for t in await c.list_tools()])
        for q in ["what's the weather in Lisbon this weekend",
                  "schedule a meeting with Ana tomorrow at 3pm",
                  "which support tickets are open",
                  "page the on-call engineer about the outage",
                  "is the air in Delhi safe to run in today",
                  "translate this paragraph into French",
                  "write me a haiku about autumn"]:
            t = time.perf_counter()
            r = (await c.call_tool("search", {"query": q})).structured_content
            ms = (time.perf_counter() - t) * 1000
            picks = [(x["path"], x.get("fit")) for x in r["tools"][:3]]
            flag = "  WARNING: nothing fits well" if "warning" in r else ""
            print(f"\nsearch {q!r}  ({ms:.0f} ms)\n  in_catalog={r.get('in_catalog')}{flag}\n  top tools (fit): {picks}")

        print("\nexecute: triage every open ticket in one call")
        r = (await c.call_tool("execute", {"code": TRIAGE})).structured_content
        print(json.dumps({k: r[k] for k in ("ok", "result", "calls", "ms") if k in r}, indent=2))
        if not r["ok"]:
            print(r["error"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="examples/config.toml")
    asyncio.run(main(p.parse_args().config))
