"""Retrieval lab: sweep embedders, fusion and cross-encoder rerankers on all three
routing benchmarks at once, scoring top-1, recall@5 and abstention AUROC.

    uv run python -m bench.lab --embed minilm bge-base --rerank msmarco-l6

Scores are cached per (model, text) under bench/results/lab_cache/, so adding a
model only encodes what is new. This is for choosing defaults; the numbers that
get reported come from bench/run.py with the shipped Searcher.
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import random
from pathlib import Path

import numpy as np

from tooljev.retrieve import BM25, tool_text

from .datasets import LOADERS
from .report import auroc

CACHE = Path(__file__).parent / "results" / "lab_cache"

# name -> (hf id, query prefix, doc prefix)
EMBED = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", "", ""),
    "bge-small": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "bge-base": ("BAAI/bge-base-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5-base": ("intfloat/e5-base-v2", "query: ", "passage: "),
    "arctic-m": ("Snowflake/snowflake-arctic-embed-m-v1.5",
                 "Represent this sentence for searching relevant passages: ", ""),
}
RERANK = {
    "msmarco-l6": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "bge-rr-base": "BAAI/bge-reranker-base",
    "mxbai-xs": "mixedbread-ai/mxbai-rerank-xsmall-v1",
    "bge-rr-m3": "BAAI/bge-reranker-v2-m3",
}


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()


class Store:
    """Pickled dict per model: sha1(text) -> vector or score."""

    def __init__(self, name: str):
        CACHE.mkdir(parents=True, exist_ok=True)
        self.path = CACHE / f"{name.replace('/', '_')}.pkl"
        self.d = pickle.loads(self.path.read_bytes()) if self.path.exists() else {}
        self.dirty = False

    def missing(self, keys):
        return [k for k in dict.fromkeys(keys) if _h(k) not in self.d]

    def put(self, k, v):
        self.d[_h(k)] = v
        self.dirty = True

    def get(self, k):
        return self.d[_h(k)]

    def save(self):
        if self.dirty:
            self.path.write_bytes(pickle.dumps(self.d))


def embed_all(name: str, queries: list[str], docs: list[str]) -> Store:
    hf, qp, dp = EMBED[name]
    st = Store(f"emb_{name}")
    todo = st.missing([qp + q for q in queries] + [dp + d for d in docs])
    if todo:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer(hf, device="mps")
        m.max_seq_length = min(m.max_seq_length or 512, 256)
        texts = todo
        vecs = m.encode(texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False)
        for t, v in zip(texts, vecs):
            st.put(t, v.astype(np.float32))
        st.save()
        print(f"  embedded {len(texts)} texts with {name}", flush=True)
    return st


def rerank_all(name: str, pairs: list[tuple[str, str]]) -> Store:
    st = Store(f"ce_{name}")
    keys = [q + "\x00" + d for q, d in pairs]
    todo = st.missing(keys)
    if todo:
        from sentence_transformers import CrossEncoder

        m = CrossEncoder(RERANK[name], device="mps", max_length=256)
        scores = m.predict([tuple(k.split("\x00", 1)) for k in todo], batch_size=64,
                           show_progress_bar=False, activation_fn=None) if _accepts_act(m) else \
            m.predict([tuple(k.split("\x00", 1)) for k in todo], batch_size=64, show_progress_bar=False)
        for k, s in zip(todo, scores):
            st.put(k, float(np.asarray(s).reshape(-1)[0]))
        st.save()
        print(f"  reranked {len(todo)} pairs with {name}", flush=True)
    return st


def _accepts_act(m) -> bool:
    import inspect

    return "activation_fn" in inspect.signature(m.predict).parameters


def rrf(rankings: list[np.ndarray], weights: list[float], k: int = 60) -> np.ndarray:
    n = len(rankings[0])
    fused = np.zeros(n)
    for scores, w in zip(rankings, weights):
        order = np.argsort(-scores, kind="stable")
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        fused += w / (k + ranks + 1)
    return fused


def main(datasets, embeds, reranks, limit, depth):
    cases = []
    for d in datasets:
        cs = LOADERS[d]()
        if limit:
            rng = random.Random(0)
            pos = [c for c in cs if c.gold]
            neg = [c for c in cs if not c.gold]
            cs = rng.sample(pos, min(limit, len(pos))) + rng.sample(neg, min(limit, len(neg)))
        cases += cs
    # per catalog: tools, docs, bm25
    cats = {}
    for c in cases:
        if id(c.catalog) not in cats:
            tools = c.catalog.tools()
            docs = [tool_text(t, c.catalog.servers[t.server].description) for t in tools]
            cats[id(c.catalog)] = (tools, docs, BM25(docs))
    all_docs = list({d for _, docs, _ in cats.values() for d in docs})
    queries = list({c.query for c in cases})
    stores = {e: embed_all(e, queries, all_docs) for e in embeds}

    def dense(e, c):
        _, qp, dp = EMBED[e]
        tools, docs, _ = cats[id(c.catalog)]
        D = np.stack([stores[e].get(dp + d) for d in docs])
        return D @ stores[e].get(qp + c.query)

    # signals per case
    rows = []
    for c in cases:
        tools, docs, bm = cats[id(c.catalog)]
        r = {"c": c, "paths": [t.path for t in tools], "docs": docs,
             "bm25": np.array(bm.scores(c.query))}
        for e in embeds:
            r[e] = dense(e, c)
        rows.append(r)

    configs = {"bm25": lambda r: r["bm25"]}
    for e in embeds:
        configs[e] = (lambda e: lambda r: r[e])(e)
        for w in (1.0, 2.0):
            configs[f"rrf(bm25,{e})x{w:g}"] = (lambda e, w: lambda r: rrf([r["bm25"], r[e]], [1.0, w]))(e, w)
    base_first = f"rrf(bm25,{embeds[0]})x1" if embeds else "bm25"

    # cross-encoders rerank the top `depth` of the best first stage (by MCPTB top-1 below)
    def score(sfun, split_pos):
        top1, r5 = [], []
        for r in split_pos:
            s = sfun(r)
            order = np.argsort(-s, kind="stable")
            gold = r["c"].gold
            top1.append(r["paths"][order[0]] in gold)
            r5.append(len({r["paths"][i] for i in order[:5]} & gold) / len(gold))
        return np.mean(top1), np.mean(r5)

    splits = {}
    for r in rows:
        splits.setdefault((r["c"].dataset.split("-")[0], bool(r["c"].gold)), []).append(r)
    names = sorted({k[0] for k in splits})

    def table(configs, abst):
        hdr = "| config | " + " | ".join(f"{n} top1 | r@5 | auc" for n in names) + " |"
        print(hdr)
        print("|---" * (1 + 3 * len(names)) + "|")
        for cname, f in configs.items():
            cells = []
            for n in names:
                pos, neg = splits.get((n, True), []), splits.get((n, False), [])
                t1, r5 = score(f, pos)
                a = abst.get(cname)
                auc = auroc([a(r) for r in pos], [a(r) for r in neg]) if a and neg else float("nan")
                cells.append(f"{t1:.3f} | {r5:.3f} | {auc:.3f}")
            print(f"| {cname} | " + " | ".join(cells) + " |", flush=True)

    abst = {e: (lambda e: lambda r: float(r[e].max()))(e) for e in embeds}
    abst["bm25"] = lambda r: float(r["bm25"].max())
    table(configs, abst)

    if not reranks:
        return
    first = configs[args.first or base_first]
    for r in rows:
        s = first(r)
        r["cand"] = list(np.argsort(-s, kind="stable")[:depth])
    ce_cfg, ce_abst = {}, {}
    for rr in reranks:
        pairs = [(r["c"].query, r["docs"][i]) for r in rows for i in r["cand"]]
        st = rerank_all(rr, pairs)
        for r in rows:
            r[f"ce_{rr}"] = {i: st.get(r["c"].query + "\x00" + r["docs"][i]) for i in r["cand"]}

        def ce_only(r, rr=rr):
            s = np.full(len(r["paths"]), -1e9)
            for rank, i in enumerate(r["cand"]):
                s[i] = r[f"ce_{rr}"][i] - 1e-6 * rank
            return s

        def ce_fuse(r, rr=rr):
            ce = np.full(len(r["paths"]), -1e9)
            for i in r["cand"]:
                ce[i] = r[f"ce_{rr}"][i]
            return rrf([first(r), ce], [1.0, 1.0])

        ce_cfg[f"ce:{rr}"] = ce_only
        ce_cfg[f"rrf(first,ce:{rr})"] = ce_fuse
        ce_abst[f"ce:{rr}"] = lambda r, rr=rr: max(r[f"ce_{rr}"].values())
        ce_abst[f"rrf(first,ce:{rr})"] = ce_abst[f"ce:{rr}"]
    print(f"\nrerank depth {depth} over {args.first or base_first}")
    table(ce_cfg, ce_abst)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["mcptoolbench", "livemcpbench", "when2call"])
    p.add_argument("--embed", nargs="*", default=["minilm"])
    p.add_argument("--rerank", nargs="*", default=[])
    p.add_argument("--first", default=None, help="first-stage config name for reranking")
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    main(args.datasets, args.embed, args.rerank, args.limit, args.depth)
