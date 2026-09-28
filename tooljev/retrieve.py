"""Cheap first-stage recall over tool descriptions: BM25, plus dense embeddings
when sentence-transformers is installed, fused by reciprocal rank.

Measured on MCPToolBench++, picking a tool is mostly a retrieval problem: a
small embedding model beat a zero-shot decision model at top-1 while costing a
few milliseconds. So retrieval narrows the catalog, and Jev does what retrieval
scores cannot: decide among the survivors, and decide whether any of them fits.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .catalog import ToolInfo

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def tool_text(t: ToolInfo, server_description: str = "") -> str:
    return f"{t.server} {server_description} {t.name} {t.description}"


# Function words carry no tool intent but are rare enough in short descriptions
# to dominate BM25: "multiply 6 by 7" matched "numerator by denominator".
_STOP = frozenset("""a an the of to in on at by for from with and or is are be it this that
what which who how can could would should will me my i you your please""".split())


def _tokens(text: str) -> list[str]:
    return [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        toks = [_tokens(d) for d in docs]
        self.tfs = [Counter(t) for t in toks]
        self.lens = [len(t) for t in toks]
        self.avg = sum(self.lens) / max(1, len(toks))
        df = Counter(w for t in toks for w in set(t))
        n = len(toks)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: str) -> list[float]:
        q = _tokens(query)
        out = []
        for tf, dl in zip(self.tfs, self.lens):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avg))
            out.append(s)
        return out


class Retriever:
    """Index over a tool list. Rebuilt only when the tool set changes."""

    def __init__(self, dense_model: str | None = "sentence-transformers/all-MiniLM-L6-v2"):
        self.dense_model = dense_model
        self._model: Any = None
        self._key: tuple | None = None
        self._tools: list[ToolInfo] = []
        self._bm25: BM25 | None = None
        self._emb: Any = None
        self._doc_cache: dict[str, Any] = {}

    def _encoder(self):
        if self._model is None and self.dense_model:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self.dense_model = None  # BM25 only
                return None
            self._model = SentenceTransformer(self.dense_model)
        return self._model

    def index(self, tools: list[ToolInfo], server_descriptions: dict[str, str]) -> None:
        key = tuple(t.path for t in tools) + tuple(sorted(server_descriptions.items()))
        if key == self._key:
            return
        docs = [tool_text(t, server_descriptions.get(t.server, "")) for t in tools]
        self._tools, self._bm25, self._key = tools, BM25(docs), key
        enc = self._encoder()
        if enc is None:
            self._emb = None
            return
        import numpy as np

        # Cache per document, so a catalog refresh only encodes tools that changed.
        missing = [d for d in dict.fromkeys(docs) if d not in self._doc_cache]
        if missing:
            for d, v in zip(missing, enc.encode(missing, normalize_embeddings=True, batch_size=64)):
                self._doc_cache[d] = v
        self._emb = np.stack([self._doc_cache[d] for d in docs])

    def top(self, query: str, k: int, rrf_k: int = 60) -> list[tuple[ToolInfo, float]]:
        """Top-k tools by reciprocal-rank fusion of BM25 and dense rankings."""
        n = len(self._tools)
        if n == 0:
            return []
        rankings = [self._bm25.scores(query)]
        if self._emb is not None:
            qv = self._encoder().encode([query], normalize_embeddings=True)[0]
            rankings.append(list(self._emb @ qv))
        fused = [0.0] * n
        for scores in rankings:
            for rank, i in enumerate(sorted(range(n), key=lambda i: -scores[i])):
                fused[i] += 1.0 / (rrf_k + rank + 1)
        order = sorted(range(n), key=lambda i: -fused[i])[:k]
        return [(self._tools[i], fused[i]) for i in order]
