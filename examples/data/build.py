"""Rebuild the showcase data from the benchmark sources (needs bench/data, see bench/RESULTS.md).

    PYTHONPATH=. uv run python examples/data/build.py

catalog.json  612 MCP tools: MCPToolBench++ (MIT) + LiveMCPBench (Apache-2.0),
              names, descriptions and input schemas only.
tickets.json  400 banking77 test messages across 10 intents (CC-BY-4.0).
"""

import json
from pathlib import Path

from bench.agent.run_agent import mcptb_tasks, triage_task

HERE = Path(__file__).parent
ROOT = HERE.parents[1]

_, specs = mcptb_tasks(36)
catalog = {s["name"]: {"description": "", "tools": s["tools"]} for s in specs}
for entry in json.load(open(ROOT / "bench/data/livemcpbench/tools.json")):
    for srv, block in entry["tools"].items():
        if srv in catalog:  # e.g. both benchmarks ship a "filesystem" server; keep both
            srv = f"{srv}-live"
        catalog[srv] = {"description": entry.get("description") or entry["name"],
                        "tools": [{"name": t["name"], "description": t.get("description") or "",
                                   "input_schema": t.get("inputSchema") or {}} for t in block["tools"]]}
(HERE / "catalog.json").write_text(json.dumps(catalog))

task, tspecs = triage_task(per_intent=40)
tickets = tspecs[0]["tools"][0]["returns"]
(HERE / "tickets.json").write_text(json.dumps(
    [{"id": t["id"], "body": t["body"], "intent": task["gold"][t["id"]]} for t in tickets], indent=0))
print(sum(len(s["tools"]) for s in catalog.values()), "tools in", len(catalog), "servers;",
      len(tickets), "tickets")
