"""Agent-in-the-loop benchmarks: a real Claude agent (headless `claude -p`) solving
tasks either with every upstream tool exposed directly, or through toolJev.

    uv run python -m bench.agent.run_agent --task mcptb --n 36
    uv run python -m bench.agent.run_agent --task triage --reps 2

Upstream servers are mocks built from the benchmark's own tool schemas; every
call they receive is logged, and scoring reads those logs, not the agent's prose.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
RESULTS = ROOT / "bench" / "results"
MOCK = ROOT / "bench" / "agent" / "mockserver.py"

SYSTEM = ("You are an autonomous agent. Complete the user's task by calling the available tools. "
          "Never ask the user questions; make sensible assumptions and finish the task.")

TRIAGE_INTENTS = ["activate_my_card", "card_arrival", "lost_or_stolen_card", "exchange_rate", "pin_blocked",
                  "top_up_failed", "transfer_not_received_by_recipient", "cash_withdrawal_charge",
                  "request_refund", "terminate_account"]

HINT = ("\n\nTip: `jev.map(..., min_confidence=0.9)` classifies every ticket in one call and flags the ones "
        "it is sure about. Route those in code, then read the unconfident ones and route them yourself.")


# ---------------------------------------------------------------- tasks

def mcptb_tasks(n: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """n MCPToolBench++ queries stratified by category, plus the 10 server specs."""
    import ast
    import glob

    recs = []
    for f in sorted(glob.glob(str(ROOT / "bench/data/mcptoolbenchpp/*.json"))):
        recs += json.load(open(f))
    lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    specs: dict[str, dict] = {}
    for r in recs:
        tools = {t["name"]: t for t in lit(r["tools"])}
        for server, names in lit(r["mcp_tools_dict"]).items():
            spec = specs.setdefault(server, {"name": server, "tools": []})
            have = {t["name"] for t in spec["tools"]}
            spec["tools"] += [tools[n] for n in names if n in tools and n not in have]
    by_cat = defaultdict(list)
    for r in recs:
        labels = lit(r["function_call_label"])
        if len(labels) == 1:
            by_cat[r["category"]].append({"query": r["query"], "category": r["category"],
                                          "server": labels[0]["mcp_server"], "tool": labels[0]["name"],
                                          "input": labels[0].get("input") or {}})
    rng = random.Random(seed)
    per = max(1, n // len(by_cat))
    tasks = [t for cat in sorted(by_cat) for t in rng.sample(by_cat[cat], min(per, len(by_cat[cat])))]
    for i, t in enumerate(tasks):
        t["id"] = f"mcptb-{i}"
        t["prompt"] = t["query"] + "\n\nComplete this request using the available tools."
    return tasks, list(specs.values())


def distractor_spec() -> dict:
    """LiveMCPBench's 525 real tool schemas as one extra mock server, to grow the catalog.

    One process rather than 69 keeps the benchmark runnable; tool names carry the
    original server as a prefix and stay under the 64-char MCP tool-name limit.
    """
    tools, seen = [], set()
    for entry in json.load(open(ROOT / "bench/data/livemcpbench/tools.json")):
        for srv, block in entry["tools"].items():
            for t in block["tools"]:
                name = f"{srv}_{t['name']}".replace(" ", "_")[:48]
                if name in seen:
                    continue
                seen.add(name)
                tools.append({"name": name, "description": t.get("description") or t["name"],
                              "input_schema": t.get("inputSchema")})
    return {"name": "extras", "tools": tools}


def triage_task(per_intent: int = 8, seed: int = 0) -> tuple[dict, list[dict]]:
    """per_intent tickets for each of 10 banking77 intents (test split)."""
    from datasets import load_dataset

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    ds = load_dataset("legacy-datasets/banking77", split="test")
    names = ds.features["label"].names
    rng = random.Random(seed)
    tickets, gold = [], {}
    for intent in TRIAGE_INTENTS:
        rows = [r for r in ds if names[r["label"]] == intent]
        for r in rng.sample(rows, per_intent):
            tickets.append(r["text"])
    rng.shuffle(tickets)
    lookup = {r["text"]: names[r["label"]] for r in ds}
    listing = [{"id": 1000 + i, "body": text} for i, text in enumerate(tickets)]
    gold = {t["id"]: lookup[t["body"]] for t in listing}
    spec = {"name": "support", "tools": [
        {"name": "list_tickets", "description": "List all open customer support tickets for the bank.",
         "input_schema": {"type": "object", "properties": {}}, "returns": listing},
        {"name": "route_ticket", "description": "Route a support ticket to the queue that handles its issue.",
         "input_schema": {"type": "object", "properties": {
             "ticket_id": {"type": "integer"},
             "queue": {"type": "string", "enum": TRIAGE_INTENTS}}, "required": ["ticket_id", "queue"]}},
    ]}
    task = {"id": "triage", "gold": gold,
            "prompt": f"Route every open support ticket ({len(listing)} of them) to the correct queue "
                      f"using the tools. Each ticket must be routed exactly once."}
    return task, [spec]


# ---------------------------------------------------------------- one agent run

def write_configs(work: Path, specs: list[dict], condition: str, log: Path) -> Path:
    spec_files = {}
    for s in specs:
        f = work / f"spec_{s['name']}.json"
        f.write_text(json.dumps(s))
        spec_files[s["name"]] = f
    if condition in ("direct", "native"):
        servers = {name: {"command": PY, "args": [str(MOCK), str(f), str(log)]}
                   for name, f in spec_files.items()}
    else:
        if "hosted" in condition:
            # The key reaches the gateway by environment inheritance, never via this file.
            # Concurrency 4: the hosted API timed out under heavier parallel load in probes.
            toml = ['trace_dir = ""', "[decider]", 'backend = "hosted"', "[sandbox]", "max_concurrency = 4"]
        else:
            toml = ['trace_dir = ""', "[decider]", 'backend = "nanojev"', 'kind = "encoder"', 'dtype = "float16"']
        for name, f in spec_files.items():
            toml += [f'[servers."{name}"]', f'command = "{PY}"', f'args = ["{MOCK}", "{f}", "{log}"]']
        cfg = work / "tooljev.toml"
        cfg.write_text("\n".join(toml) + "\n")
        servers = {"tooljev": {"command": PY, "args": ["-m", "tooljev.server", "--config", str(cfg)],
                               "cwd": str(ROOT)}}
    mcp = work / "mcp.json"
    mcp.write_text(json.dumps({"mcpServers": servers}))
    return mcp


async def run_claude(prompt: str, specs: list[dict], condition: str, model: str, timeout: float) -> dict:
    work = Path(tempfile.mkdtemp(prefix="tjbench_"))
    log = work / "calls.jsonl"
    log.touch()
    mcp = write_configs(work, specs, condition, log)
    # "native": every tool exposed, plus Claude Code's own deferred-loading ToolSearch,
    # which is what a real Claude Code user gets once tool definitions outgrow the context.
    exposed = condition in ("direct", "native")
    allowed = ",".join(([f"mcp__{s['name']}" for s in specs] if exposed else ["mcp__tooljev"])
                       + (["ToolSearch"] if condition == "native" else []))
    cmd = ["claude", "-p", prompt, "--system-prompt", SYSTEM, "--strict-mcp-config", "--mcp-config", str(mcp),
           "--setting-sources", "", "--no-session-persistence", "--output-format", "stream-json", "--verbose",
           "--model", model, "--allowedTools", allowed,
           "--tools", "ToolSearch" if condition == "native" else ""]
    t0 = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.DEVNULL,
                                                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                cwd=str(work))
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
        timed_out = False
    except asyncio.TimeoutError:
        proc.kill()
        out, timed_out = b"", True
    wall = time.perf_counter() - t0
    events = [json.loads(l) for l in out.decode().splitlines() if l.strip().startswith("{")]
    init = next((e for e in events if e.get("type") == "system"), {})
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    tool_uses = [c for e in events if e.get("type") == "assistant"
                 for c in e["message"]["content"] if c.get("type") == "tool_use"]
    results = {c["tool_use_id"]: c for e in events if e.get("type") == "user"
               for c in (e["message"]["content"] if isinstance(e["message"]["content"], list) else [])
               if isinstance(c, dict) and c.get("type") == "tool_result"}
    steps = []
    for t in tool_uses:
        res = results.get(t["id"], {}).get("content")
        steps.append({"tool": t["name"], "input": json.dumps(t["input"])[:1500],
                      "result": json.dumps(res)[:1500] if res is not None else None})
    usage = result.get("usage", {})
    calls = [json.loads(l) for l in log.read_text().splitlines() if l.strip()]
    shutil.rmtree(work, ignore_errors=True)
    return {
        "condition": condition, "timed_out": timed_out, "wall_s": round(wall, 1),
        "mcp_status": [s.get("status") for s in init.get("mcp_servers", [])],
        "n_tools_visible": len([t for t in init.get("tools", []) if t.startswith("mcp__")]),
        "turns": result.get("num_turns"), "cost_usd": result.get("total_cost_usd"),
        # distinct model responses; turns counts tool rounds, and parallel tool calls share one response
        "llm_calls": len({e["message"].get("id") for e in events if e.get("type") == "assistant"}),
        "input_tokens": sum(usage.get(k, 0) for k in
                            ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")),
        "output_tokens": usage.get("output_tokens", 0),
        "agent_tool_calls": [t["name"] for t in tool_uses],
        "steps": steps,
        "upstream_calls": calls,
        "final": (result.get("result") or "")[:500],
        "is_error": result.get("is_error"),
    }


# ---------------------------------------------------------------- scoring

def _norm(v) -> str:
    return json.dumps(v, sort_keys=True).lower().strip('"').strip() if not isinstance(v, str) else v.lower().strip()


def score_mcptb(task: dict, run: dict) -> dict:
    hits = [c for c in run["upstream_calls"] if c["server"] == task["server"] and c["tool"] == task["tool"]]
    arg_acc = None
    if hits and task["input"]:
        a = hits[0]["args"]
        arg_acc = sum(_norm(a.get(k)) == _norm(v) for k, v in task["input"].items()) / len(task["input"])
    return {"tool_correct": bool(hits), "arg_acc": arg_acc,
            "first_call_correct": bool(run["upstream_calls"]) and run["upstream_calls"][0]["tool"] == task["tool"],
            "n_upstream_calls": len(run["upstream_calls"])}


def score_triage(task: dict, run: dict) -> dict:
    routed: dict[int, list[str]] = defaultdict(list)
    for c in run["upstream_calls"]:
        if c["tool"] == "route_ticket":
            try:
                routed[int(c["args"].get("ticket_id"))].append(c["args"].get("queue"))
            except (TypeError, ValueError):
                pass
    gold = {int(k): v for k, v in task["gold"].items()}
    correct = sum(1 for tid, q in routed.items() if tid in gold and q[-1] == gold[tid])
    return {"accuracy": correct / len(gold), "coverage": len([t for t in routed if t in gold]) / len(gold),
            "duplicate_routes": sum(len(q) - 1 for q in routed.values()),
            "n_upstream_calls": len(run["upstream_calls"])}


# ---------------------------------------------------------------- main

async def main(task: str, conditions: list[str], n: int, reps: int, model: str, parallel: int,
               distractors: bool = False) -> None:
    RESULTS.mkdir(exist_ok=True)
    suffix = "_big" if distractors else ""
    out = RESULTS / (f"agent_{task}{suffix}_{model}.jsonl" if task == "mcptb"
                     else f"agent_{task}{n * 10}_{model}.jsonl")
    sem = asyncio.Semaphore(parallel)
    if task == "mcptb":
        tasks, specs = mcptb_tasks(n)
        if distractors:
            specs = specs + [distractor_spec()]
        jobs = [(t, c, r) for t in tasks for c in conditions for r in range(reps)]
        scorer, timeout = score_mcptb, 240
    else:
        t, specs = triage_task(per_intent=n)
        jobs = [(t, c, r) for c in conditions for r in range(reps)]
        scorer, timeout = score_triage, 900

    async def one(t, cond, rep):
        async with sem:
            prompt = t["prompt"] + (HINT if cond.endswith("-hint") else "")
            run = await run_claude(prompt, specs, cond, model, timeout)
            row = {"task": t["id"], "rep": rep, "model": model, **run, **scorer(t, run)}
            if task == "mcptb":
                row.update({k: t[k] for k in ("category", "server", "tool", "query")})
            with out.open("a") as f:
                f.write(json.dumps(row) + "\n")
            keys = ["tool_correct", "arg_acc"] if task == "mcptb" else ["accuracy", "coverage"]
            print(f"{t['id']:10s} {cond:12s} rep{rep} " + " ".join(f"{k}={row[k]}" for k in keys) +
                  f" llm_calls={row['llm_calls']} out_tok={row['output_tokens']} in_tok={row['input_tokens']} ${row['cost_usd']} {row['wall_s']}s",
                  flush=True)

    await asyncio.gather(*(one(*j) for j in jobs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["mcptb", "triage"], required=True)
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--n", type=int, default=36, help="mcptb: tasks; triage: tickets per intent (x10 total)")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--model", default="haiku")
    p.add_argument("--parallel", type=int, default=3)
    p.add_argument("--distractors", action="store_true", help="mcptb: add LiveMCPBench's 525 tools")
    a = p.parse_args()
    conds = a.conditions or (["direct", "tooljev"] if a.task == "mcptb" else ["direct", "tooljev", "tooljev-hint"])
    asyncio.run(main(a.task, conds, a.n, a.reps, a.model, a.parallel, a.distractors))
