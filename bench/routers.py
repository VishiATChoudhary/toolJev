"""Routers under test. Each maps (query, catalog) to a ranked tool list plus an
in-catalog score, which is what an abstain decision would threshold on."""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from typing import Any

from tooljev.config import SearchConfig
from tooljev.decider import make_decider
from tooljev.retrieve import Retriever
from tooljev.search import Searcher

from .datasets import StaticCatalog


def tool_text(catalog: StaticCatalog, t) -> str:
    return f"{t.server} {catalog.servers[t.server].description} {t.name} {t.description}"


class JevRouter:
    """toolJev's own search, with the shortlist opened up so recall@k is measurable.

    variant: "" (the default: retrieval order, Jev only for abstention),
    "rerank" (Jev reranks the retrieved tools), "hier" (Jev alone, server then tool).
    """

    def __init__(self, kind: str, variant: str = ""):
        self.name = f"tooljev-{kind}" + (f"-{variant}" if variant else "")
        if kind == "hosted":
            self.decider = make_decider("hosted")  # TYPESAFE_API_KEY from the environment
        else:
            # fp16 on Apple MPS is ~2.2x faster than fp32 for the encoder, same answers on spot checks.
            self.decider = make_decider("nanojev", kind=kind, dtype="float16")
        # abstain_below=0 so every case yields a ranking; abstention is scored from in_catalog.
        self.cfg = SearchConfig(tool_mass=1.0, max_tools=10, abstain_below=0.0,
                                mode="hierarchical" if variant == "hier" else "retrieve",
                                rerank=variant == "rerank")
        self.retriever = Retriever(self.cfg.dense_model)

    async def route(self, query: str, catalog: StaticCatalog) -> dict[str, Any]:
        t0 = time.perf_counter()
        r = await Searcher(catalog, self.decider, self.cfg, self.retriever).search(query)
        ms = (time.perf_counter() - t0) * 1000
        # score: Jev's probability when it ranked, else its per-tool fit (retrieval order kept)
        return {"ranked": [(t["path"], t.get("probability", t.get("fit", 0.0))) for t in r["tools"]],
                "in_catalog": r["in_catalog"], "ms": ms}


_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("_", " ").replace("-", " "))


class BM25Router:
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self._index: dict[int, tuple] = {}

    def _build(self, catalog: StaticCatalog):
        key = id(catalog)
        if key not in self._index:
            tools = catalog.tools()
            docs = [_tokens(tool_text(catalog, t)) for t in tools]
            df = Counter(w for d in docs for w in set(d))
            n = len(docs)
            idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
            avg = sum(map(len, docs)) / max(1, n)
            self._index[key] = (catalog, tools, [Counter(d) for d in docs], [len(d) for d in docs], idf, avg)
        return self._index[key]

    async def route(self, query: str, catalog: StaticCatalog) -> dict[str, Any]:
        t0 = time.perf_counter()
        _, tools, tfs, lens, idf, avg = self._build(catalog)
        q = _tokens(query)
        scores = []
        for tf, dl in zip(tfs, lens):
            s = 0.0
            for w in q:
                if w in tf:
                    s += idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * dl / avg))
            scores.append(s)
        ranked = sorted(zip((t.path for t in tools), scores), key=lambda x: -x[1])[:10]
        return {"ranked": ranked, "in_catalog": ranked[0][1] if ranked else 0.0,
                "ms": (time.perf_counter() - t0) * 1000}


class DenseRouter:
    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = "dense-" + model.split("/")[-1].split("-")[1].lower()
        self.model = SentenceTransformer(model)
        self._cache: dict[str, Any] = {}

    def _embed(self, texts: list[str]):
        missing = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if missing:
            for t, v in zip(missing, self.model.encode(missing, normalize_embeddings=True, batch_size=64)):
                self._cache[t] = v
        return [self._cache[t] for t in texts]

    async def route(self, query: str, catalog: StaticCatalog) -> dict[str, Any]:
        import numpy as np

        tools = catalog.tools()
        texts = [tool_text(catalog, t) for t in tools]
        self._embed(texts)  # catalog embeddings are cached, as a gateway would; not timed
        t0 = time.perf_counter()
        qv = self.model.encode([query], normalize_embeddings=True)[0]
        sims = np.stack(self._embed(texts)) @ qv
        order = np.argsort(-sims)[:10]
        ranked = [(tools[i].path, float(sims[i])) for i in order]
        return {"ranked": ranked, "in_catalog": ranked[0][1], "ms": (time.perf_counter() - t0) * 1000}


class HybridRouter:
    """Retrieval alone, BM25 + dense fused by reciprocal rank: toolJev's first stage.

    "hybrid-rrf" is the original first stage (MiniLM, equal weights); "hybrid-bge" is the current one.
    """

    def __init__(self, name: str, model: str, dense_weight: float):
        self.name = name
        self.retriever = Retriever(model, dense_weight)

    async def route(self, query: str, catalog: StaticCatalog) -> dict[str, Any]:
        self.retriever.index(catalog.tools(), {n: s.description for n, s in catalog.servers.items()})
        t0 = time.perf_counter()
        hits = self.retriever.top(query, 10)
        ranked = [(t.path, s) for t, s in hits]
        return {"ranked": ranked, "in_catalog": ranked[0][1] if ranked else 0.0,
                "ms": (time.perf_counter() - t0) * 1000}


def make_router(name: str):
    if name.startswith("tooljev-"):
        parts = name.split("-")
        return JevRouter(parts[1], parts[2] if len(parts) > 2 else "")
    if name == "hybrid-rrf":
        return HybridRouter(name, "sentence-transformers/all-MiniLM-L6-v2", 1.0)
    if name == "hybrid-bge":
        return HybridRouter(name, "BAAI/bge-base-en-v1.5", 2.0)
    if name == "bm25":
        return BM25Router()
    if name == "dense-minilm":
        return DenseRouter("sentence-transformers/all-MiniLM-L6-v2")
    if name == "dense-mpnet":
        return DenseRouter("sentence-transformers/all-mpnet-base-v2")
    raise ValueError(name)
