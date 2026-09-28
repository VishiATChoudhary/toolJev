"""Run routers over routing/abstention benchmarks. One JSONL row per (case, router).

    uv run python -m bench.run --datasets mcptoolbench livemcpbench when2call bfcl \\
        --routers bm25 dense-minilm tooljev-encoder --limit 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from tooljev.catalog import py_name

from .datasets import LOADERS
from .routers import make_router

RESULTS = Path(__file__).parent / "results"


def sample(cases, limit: int | None, seed: int = 0):
    """Keep positives and negatives balanced when subsampling."""
    if not limit:
        return cases
    rng = random.Random(seed)
    pos = [c for c in cases if c.gold]
    neg = [c for c in cases if not c.gold]
    if pos and neg:
        return rng.sample(pos, min(limit, len(pos))) + rng.sample(neg, min(limit, len(neg)))
    return rng.sample(cases, min(limit, len(cases)))


async def main(datasets: list[str], routers: list[str], limit: int | None, concurrency: int = 1) -> None:
    RESULTS.mkdir(exist_ok=True)
    loaded = {d: sample(LOADERS[d](), limit) for d in datasets}
    for rname in routers:
        router = make_router(rname)
        for d, cases in loaded.items():
            out = RESULTS / f"{d}__{rname}.jsonl"
            t0 = time.perf_counter()
            sem = asyncio.Semaphore(concurrency)

            async def routed(c):
                async with sem:
                    for attempt in range(3):
                        try:
                            return await router.route(c.query, c.catalog)
                        except Exception as e:  # remote backends: timeouts under load
                            err = str(e)[:200]
                            await asyncio.sleep(5 * (attempt + 1))
                    return {"ranked": [], "in_catalog": 0.0, "ms": 0.0, "error": err}

            # Results come back in case order. Per-case ms starts after the semaphore, so it
            # excludes queueing but not contention between in-flight requests.
            outs = await asyncio.gather(*(routed(c) for c in cases)) if concurrency > 1 else None
            with out.open("w") as f:
                for i, c in enumerate(cases):
                    r = outs[i] if outs is not None else await routed(c)
                    if "error" in r:
                        print(f"  case failed after retries: {r['error']}", file=sys.stderr, flush=True)
                    f.write(json.dumps({
                        "error": r.get("error"),
                        "dataset": c.dataset, "router": rname, "query": c.query,
                        "gold": sorted(c.gold), "ranked": r["ranked"], "in_catalog": r["in_catalog"],
                        "ms": round(r["ms"], 2), "n_tools": len(c.catalog.tools()),
                        "meta": {k: (sorted(v) if isinstance(v, set) else v)
                                 for k, v in c.meta.items() if k != "category_of"},
                        "top_category": {py_name(k): v for k, v in c.meta.get("category_of", {}).items()}.get(
                            r["ranked"][0][0].split(".")[0] if r["ranked"] else "", None),
                    }) + "\n")
                    if (i + 1) % 50 == 0:
                        print(f"  {rname} {d} {i + 1}/{len(cases)}", file=sys.stderr, flush=True)
            print(f"{rname:16s} {d:14s} {len(cases):5d} cases {time.perf_counter() - t0:7.1f}s -> {out.name}",
                  flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(LOADERS))
    p.add_argument("--routers", nargs="+", default=["bm25", "dense-minilm", "tooljev-encoder"])
    p.add_argument("--limit", type=int, default=None, help="max positives (and negatives) per dataset")
    p.add_argument("--concurrency", type=int, default=1, help="parallel cases (remote backends)")
    a = p.parse_args()
    asyncio.run(main(a.datasets, a.routers, a.limit, a.concurrency))
