"""`search(query)`: find the tools a request needs, and say whether any fits.

1. Retrieval (BM25 + bge-base embeddings, fused by reciprocal rank) shortlists
   `recall_k` tools from the whole catalog in a few milliseconds.
2. Jev looks at the shortlist in one call:
   - a Choice over the candidates, which reranks them (on by default with hosted
     Jev; the local nanojev encoder is worse than retrieval order, so it skips this);
   - one Noul per candidate, "this request asks to <tool>", which becomes each
     tool's `fit`. `in_catalog` is the highest fit; under `abstain_below` the
     result carries a warning (or, with abstain="hard", no tools).

Measured with hosted Jev, the rerank lifts top-1 over retrieval alone from 0.73
to 0.83 on MCPToolBench++, 0.40 to 0.54 on LiveMCPBench and 0.92 to 0.99 on
When2Call (bench/RESULTS.md). `mode="hierarchical"` (Jev alone picks a server,
then a tool) is the original design, kept as an ablation.
"""

from __future__ import annotations

import asyncio
from typing import Any

from .catalog import Catalog, ToolInfo
from .config import SearchConfig
from .decider import Decider
from .retrieve import Retriever

# Per-tool Nouls are only asked when the candidate set is this small, so a
# catalog with a huge server doesn't turn every search into hundreds of questions.
MAX_TOOL_NOULS = 64


def take_by_mass(probs: dict[str, float], mass: float, cap: int) -> list[tuple[str, float]]:
    """Highest-probability keys until `mass` is covered or `cap` is hit. Never empty."""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    out, acc = [], 0.0
    for key, p in ranked[:max(1, cap)]:
        out.append((key, p))
        acc += p
        if acc >= mass:
            break
    return out


def _server_criteria(catalog: Catalog, names: list[str], tool_list_chars: int = 400) -> dict[str, str]:
    criteria = {}
    for name in names:
        desc = catalog.servers[name].description
        tools = ", ".join(t.name for t in catalog.tools(name))
        if len(tools) > tool_list_chars:
            tools = tools[:tool_list_chars].rsplit(", ", 1)[0] + ", ..."
        criteria[name] = f"{desc} Tools: {tools}".strip()
    return criteria


def _noul(instructions: str) -> dict[str, Any]:
    return {"type": "noul", "instructions": instructions}


class Searcher:
    def __init__(self, catalog: Catalog, decider: Decider, cfg: SearchConfig,
                 retriever: Retriever | None = None):
        self.catalog = catalog
        self.decider = decider
        self.cfg = cfg
        self.retriever = retriever
        self.rerank = cfg.rerank if cfg.rerank is not None else getattr(decider, "reranks_well", False)

    async def search(self, query: str) -> dict[str, Any]:
        await self.catalog.refresh()
        servers = self.catalog.server_names()
        if not servers:
            return {"query": query, "tools": [], "reason": "no upstream servers have tools"}

        if self.cfg.mode == "hierarchical":
            candidates, tool_probs, confidence, nouls = await self._hierarchical(query, servers)
        else:
            candidates, tool_probs, confidence, nouls = await self._retrieve(query, servers)
        fits = {k.removeprefix("tool:"): v for k, v in nouls.items() if k.startswith("tool:")}
        in_catalog = float(max(nouls.values())) if nouls else 1.0

        server_probs: dict[str, float] = {}
        by_path = {t.path: t for t in candidates}
        for path, p in tool_probs.items():
            server_probs[by_path[path].server] = server_probs.get(by_path[path].server, 0.0) + p
        result: dict[str, Any] = {"query": query, "in_catalog": round(in_catalog, 3),
                                  "servers": _rounded(server_probs)}
        weak = in_catalog < self.cfg.abstain_below
        if weak and self.cfg.abstain == "hard":
            result["tools"] = []
            result["reason"] = "no available tool looks able to serve this request"
            return result
        if weak:
            # Soft by default: measured on MCPToolBench++, hiding tools on a low score cost
            # the agent more (wrong detours) than showing a warned shortlist.
            result["warning"] = ("none of these tools looks like a strong fit; check the "
                                 "signatures before relying on them")
        ranked = self.cfg.mode == "hierarchical" or self.rerank
        if ranked:
            picked = take_by_mass(tool_probs, self.cfg.tool_mass, self.cfg.max_tools)
            result["confidence"] = round(confidence, 3)
        else:
            # Retrieval order; its fused scores are not probabilities, so none are shown.
            picked = [(t.path, None) for t in candidates[:self.cfg.max_tools]]
        tools = []
        for path, p in picked:
            entry: dict[str, Any] = {"path": path}
            if p is not None:
                entry["probability"] = round(p, 3)
            if path in fits:
                entry["fit"] = round(fits[path], 3)  # P(the request asks for what this tool does)
            entry["signature"] = by_path[path].stub()
            tools.append(entry)
        result["tools"] = tools
        return result

    def _tool_nouls(self, tools: list[ToolInfo]) -> dict[str, Any]:
        out = {}
        for t in tools[:MAX_TOOL_NOULS]:
            what = (t.summary() or t.name).rstrip(".")
            out[f"tool:{t.path}"] = _noul(f"This request asks to: {what[:1].lower()}{what[1:]}")
        return out

    async def _retrieve(self, query: str, servers: list[str]):
        """Retrieval shortlists `recall_k` tools; Jev reranks them and judges fit."""
        if self.retriever is None:
            self.retriever = Retriever(self.cfg.dense_model or None)
        self.retriever.index(self.catalog.tools(),
                             {n: self.catalog.servers[n].description for n in servers})
        hits = self.retriever.top(query, self.cfg.recall_k)
        candidates = [t for t, _ in hits]
        nouls_q = self._tool_nouls(candidates)
        if self.rerank:
            probs, confidence, extra = await self._choice(
                query, "Which tool should be called to carry out this request?",
                {t.path: t.summary() or t.name for t in candidates}, nouls_q)
        else:
            # Keep the retrieval order; Jev only answers the in-catalog question.
            extra = await self.decider.decide(query, nouls_q) if nouls_q else {}
            total = sum(s for _, s in hits) or 1.0
            probs = {t.path: s / total for t, s in hits}
            confidence = max(probs.values())
        return candidates, probs, confidence, {k: a["noul"] for k, a in extra.items()}

    async def _hierarchical(self, query: str, servers: list[str]):
        """Jev only: pick servers, then tools within them. Kept as an ablation."""
        server_nouls = {
            f"srv:{name}": _noul(f"This request can be handled by a tool for: "
                                 f"{self.catalog.servers[name].description or name}")
            for name in servers
        }
        server_probs, _, extra = await self._choice(
            query, "Which tool server is needed for this request?",
            _server_criteria(self.catalog, servers), server_nouls)
        nouls = {k: a["noul"] for k, a in extra.items()}
        chosen = take_by_mass(server_probs, self.cfg.server_mass, self.cfg.max_servers)
        candidates = [t for name, _ in chosen for t in self.catalog.tools(name)]
        tool_nouls = self._tool_nouls(candidates) if len(candidates) <= MAX_TOOL_NOULS else {}
        tool_probs, confidence, extra = await self._choice(
            query, "Which tool should be called to carry out this request?",
            {t.path: t.summary() or t.name for t in candidates}, tool_nouls)
        return candidates, tool_probs, confidence, {**nouls, **{k: a["noul"] for k, a in extra.items()}}

    async def _choice(self, query: str, instructions: str, criteria: dict[str, str],
                      extra: dict[str, Any]) -> tuple[dict[str, float], float, dict[str, Any]]:
        """One Choice over `criteria`, asked alongside the `extra` questions.

        Past the backend's option cap, knockout rounds narrow the field first;
        `extra` rides along only on the final call. Returns (probabilities,
        confidence, answers to extra).
        """
        cap = self.decider.max_options
        keys = list(criteria)
        if len(keys) > cap:
            chunks = [keys[i:i + cap] for i in range(0, len(keys), cap)]
            rounds = await asyncio.gather(*(
                self._choice(query, instructions, {k: criteria[k] for k in c}, {}) for c in chunks))
            # Each chunk sends at most half its size forward, so the field shrinks every round.
            per_chunk = max(1, min(self.cfg.max_tools, cap // 2))
            keep = {k for probs, _, _ in rounds for k, _ in take_by_mass(probs, 1.0, per_chunk)}
            return await self._choice(query, instructions,
                                      {k: v for k, v in criteria.items() if k in keep}, extra)

        questions = dict(extra)
        if len(keys) > 1:
            questions["choice"] = {"type": "choice", "instructions": instructions, "criteria": criteria}
        if not questions:
            return {keys[0]: 1.0}, 1.0, {}
        ans = await self.decider.decide(query, questions)
        answered = {k: v for k, v in ans.items() if k != "choice"}
        if "choice" not in ans:
            return {keys[0]: 1.0}, 1.0, answered
        probs = dict(ans["choice"]["probabilities"])
        return probs, float(ans["choice"].get("confidence", max(probs.values()))), answered


def _rounded(probs: dict[str, float]) -> dict[str, float]:
    return {k: round(float(v), 3) for k, v in sorted(probs.items(), key=lambda kv: -kv[1])}
