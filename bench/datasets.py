"""Benchmark loaders, all reduced to one shape.

A Case is a query, the catalog it is asked against, and the gold tool paths.
An empty gold set means the right answer is to abstain.
"""

from __future__ import annotations

import ast
import glob
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from tooljev.catalog import ToolInfo, py_name
from tooljev.config import ServerConfig

DATA = Path(__file__).parent / "data"


@dataclass
class StaticCatalog:
    """Same surface the Searcher uses from Catalog, minus the live connections."""

    servers: dict[str, ServerConfig]
    _tools: dict[str, list[ToolInfo]]

    async def refresh(self, force: bool = False) -> None:
        pass

    def tools(self, server: str | None = None) -> list[ToolInfo]:
        if server is not None:
            return list(self._tools.get(server, []))
        return [t for ts in self._tools.values() for t in ts]

    def server_names(self) -> list[str]:
        return [n for n in self.servers if self._tools.get(n)]

    def without(self, drop: set[str]) -> StaticCatalog:
        keep = [n for n in self.servers if n not in drop]
        return StaticCatalog({n: self.servers[n] for n in keep}, {n: self._tools[n] for n in keep})


def build_catalog(spec: dict[str, tuple[str, list[dict]]]) -> StaticCatalog:
    """spec: server -> (description, [{"name", "description", "input_schema"}])."""
    servers, tools = {}, {}
    for name, (desc, ts) in spec.items():
        servers[name] = ServerConfig(name, transport=None, description=desc)
        seen, tools[name] = set(), []
        for t in ts:
            if t["name"] in seen:
                continue
            seen.add(t["name"])
            tools[name].append(ToolInfo(name, t["name"], t.get("description") or "",
                                        t.get("input_schema") or {}))
    return StaticCatalog(servers, tools)


def path(server: str, tool: str) -> str:
    return f"{py_name(server)}.{py_name(tool)}"


@dataclass
class Case:
    dataset: str
    query: str
    catalog: StaticCatalog
    gold: set[str]  # tool paths; empty = should abstain
    meta: dict = field(default_factory=dict)


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def mcptoolbench(neg_per_server: int = 30, seed: int = 0) -> list[Case]:
    """MCPToolBench++ single-call queries against one merged 10-server catalog.

    Negatives: leave-category-out. Every server in the gold server's category is
    removed, so nothing fits. Removing only the gold server is not enough: the
    catalog has near-duplicate siblings (three map providers, two web-search
    servers, two browser automators), so the query often stays answerable.
    """
    recs = []
    for f in sorted(glob.glob(str(DATA / "mcptoolbenchpp" / "*.json"))):
        recs += json.load(open(f))
    spec: dict[str, tuple[str, list[dict]]] = {}
    category = {}
    for r in recs:
        tools = {t["name"]: t for t in _lit(r["tools"])}
        for server, names in _lit(r["mcp_tools_dict"]).items():
            desc, ts = spec.setdefault(server, ("", []))
            ts += [tools[n] for n in names if n in tools]
            category[server] = r["category"]
    cat = build_catalog(spec)
    cases, by_server = [], {}
    for r in recs:
        labels = _lit(r["function_call_label"])
        gold = {path(l["mcp_server"], l["name"]) for l in labels}
        servers = {l["mcp_server"] for l in labels}
        c = Case("mcptoolbench", r["query"], cat, gold,
                 {"category": r["category"], "servers": servers, "category_of": category})
        cases.append(c)
        if len(servers) == 1:
            by_server.setdefault(next(iter(servers)), []).append(c)
    rng = random.Random(seed)
    for server, cs in sorted(by_server.items()):
        siblings = {s for s, c in category.items() if c == category[server]}
        reduced = cat.without(siblings)
        for c in rng.sample(cs, min(neg_per_server, len(cs))):
            cases.append(Case("mcptoolbench-lso", c.query, reduced, set(), {"removed": sorted(siblings)}))
    return cases


def livemcpbench() -> list[Case]:
    """LiveMCPBench: 95 multi-tool tasks over 69 real servers / 525 tools.

    Negatives: every server holding a gold tool is removed.
    """
    spec, owner = {}, {}
    for entry in json.load(open(DATA / "livemcpbench" / "tools.json")):
        for srv, block in entry["tools"].items():
            ts = [{"name": t["name"], "description": t.get("description"),
                   "input_schema": t.get("inputSchema")} for t in block["tools"]]
            spec[srv] = (entry.get("description") or entry["name"], ts)
            for t in ts:
                owner.setdefault(t["name"], srv)
    cat = build_catalog(spec)
    cases = []
    for a in json.load(open(DATA / "livemcpbench" / "all_annotations.json")):
        names = [ln.split(".", 1)[-1].strip() for ln in a["Annotator Metadata"]["Tools"].splitlines()]
        gold = {path(owner[n], n) for n in names if n in owner}
        if not gold:
            continue
        servers = {owner[n] for n in names if n in owner}
        cases.append(Case("livemcpbench", a["Question"], cat, gold, {"servers": servers}))
        cases.append(Case("livemcpbench-lso", a["Question"], cat.without(servers), set(),
                          {"removed": sorted(servers)}))
    return cases


def _one_server_per_tool(tools: list[dict]) -> StaticCatalog:
    return build_catalog({t["name"]: (t.get("description") or t["name"], [t]) for t in tools})


def when2call(n_each: int = 250, seed: int = 0) -> list[Case]:
    """When2Call MCQ: 'tool_call' rows are positives, 'cannot_answer' rows are negatives.

    Tools are BFCL functions with no server, so each becomes its own server.
    """
    rows = [json.loads(l) for l in open(DATA / "when2call" / "test_mcq.jsonl")]
    rng = random.Random(seed)
    cases = []
    for label in ("tool_call", "cannot_answer"):
        pool = [r for r in rows if r["correct_answer"] == label and r["tools"]]
        for r in rng.sample(pool, min(n_each, len(pool))):
            tools = [json.loads(t) if isinstance(t, str) else t for t in r["tools"]]
            tools = [{"name": t["name"], "description": t.get("description"),
                      "input_schema": t.get("parameters")} for t in tools]
            gold = set()
            if label == "tool_call":
                name = json.loads(r["answers"]["tool_call"])["name"]
                gold = {path(name, name)}
            cases.append(Case("when2call", r["question"], _one_server_per_tool(tools), gold,
                              {"label": label, "n_tools": len(tools)}))
    return cases


def bfcl_irrelevance() -> list[Case]:
    """BFCL v4 irrelevance: every query is unanswerable with the functions given."""
    cases = []
    for r in map(json.loads, open(DATA / "bfcl" / "irrelevance.jsonl")):
        q = " ".join(m["content"] for turn in r["question"] for m in turn if m["role"] == "user")
        tools = [{"name": f["name"], "description": f.get("description"),
                  "input_schema": f.get("parameters")} for f in r["function"]]
        cases.append(Case("bfcl-irrelevance", q, _one_server_per_tool(tools), set()))
    return cases


LOADERS = {"mcptoolbench": mcptoolbench, "livemcpbench": livemcpbench,
           "when2call": when2call, "bfcl": bfcl_irrelevance}
