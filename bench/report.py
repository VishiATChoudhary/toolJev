"""Score bench/results/*.jsonl into bench/results/summary.json and print markdown tables.

    uv run python -m bench.report
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from tooljev.search import take_by_mass

RESULTS = Path(__file__).parent / "results"

# (positive split, negative split) pairs for abstention
ABSTAIN_PAIRS = {
    "MCPToolBench++ (leave-category-out)": ("mcptoolbench", "mcptoolbench-lso"),
    "LiveMCPBench (leave-server-out)": ("livemcpbench", "livemcpbench-lso"),
    "When2Call": ("when2call+", "when2call-"),
}


def load() -> dict[str, dict[str, list[dict]]]:
    """router -> split -> rows. When2Call is split into + / - by gold."""
    out: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for f in sorted(RESULTS.glob("*__*.jsonl")):  # routing runs only, not agent_*
        for line in f.open():
            r = json.loads(line)
            if r.get("error"):
                continue  # failed after retries; counted separately in the run log
            split = r["dataset"]
            if split == "when2call":
                split += "+" if r["gold"] else "-"
            out[r["router"]][split].append(r)
    return out


def auroc(pos: list[float], neg: list[float]) -> float:
    """P(score of a random positive > random negative), ties count half."""
    if not pos or not neg:
        return float("nan")
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i = 0.0, 0
    while i < len(ranked):
        j = i
        while j < len(ranked) and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2  # 1-based average rank of the tie block
        rank_sum += avg_rank * sum(lbl for _, lbl in ranked[i:j])
        i = j
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def threshold_at_tpr(pos: list[float], tpr: float) -> float:
    s = sorted(pos, reverse=True)
    return s[min(len(s) - 1, int(tpr * len(s)))]


def routing(rows: list[dict], adaptive: bool) -> dict[str, float]:
    top1 = [bool(r["ranked"]) and r["ranked"][0][0] in r["gold"] for r in rows]
    r5 = [len({p for p, _ in r["ranked"][:5]} & set(r["gold"])) / len(r["gold"]) for r in rows]
    srv = [bool(r["ranked"]) and r["ranked"][0][0].split(".")[0] in {g.split(".")[0] for g in r["gold"]}
           for r in rows]
    m = {"n": len(rows), "top1": _mean(top1), "recall@5": _mean(r5), "server@1": _mean(srv),
         "p50_ms": statistics.median(r["ms"] for r in rows),
         "p95_ms": sorted(r["ms"] for r in rows)[int(0.95 * (len(rows) - 1))]}
    if rows and "category" in rows[0]["meta"]:
        m["category@1"] = _mean([r["top_category"] == r["meta"]["category"] for r in rows])
    if adaptive:
        # the shortlist toolJev returns by default: 0.9 probability mass, at most 5
        lists = [take_by_mass(dict(r["ranked"]), 0.9, 5) if r["ranked"] else [] for r in rows]
        m["shortlist_hit"] = _mean([bool({p for p, _ in l} & set(r["gold"])) for l, r in zip(lists, rows)])
        m["shortlist_size"] = _mean([len(l) for l in lists])
    # selective prediction: accuracy on the half the router is most confident about
    by_conf = sorted(zip((r["ranked"][0][1] if r["ranked"] else 0 for r in rows), top1), reverse=True)
    half = by_conf[: max(1, len(by_conf) // 2)]
    m["top1@50%cov"] = _mean([ok for _, ok in half])
    return m


def _mean(xs) -> float:
    xs = list(xs)
    return sum(map(float, xs)) / len(xs) if xs else float("nan")


def match(data, ref: str):
    """Restrict every router to the cases `ref` ran, so sampled runs compare like for like."""
    keys = {split: {(r["query"], r["n_tools"]) for r in rows} for split, rows in data[ref].items()}
    return {router: {split: [r for r in rows if (r["query"], r["n_tools"]) in keys.get(split, set())]
                     for split, rows in splits.items()}
            for router, splits in data.items()}


def main(ref: str | None = None) -> None:
    data = load()
    if ref:
        data = match(data, ref)
    summary: dict = {"routing": {}, "abstention": {}, "transfer": {}}
    for router, splits in data.items():
        for split in ("mcptoolbench", "livemcpbench", "when2call+"):
            if splits.get(split):
                summary["routing"].setdefault(split, {})[router] = routing(
                    splits[split], adaptive=router.startswith("tooljev"))
        for name, (p, n) in ABSTAIN_PAIRS.items():
            pos = [r["in_catalog"] for r in splits.get(p, [])]
            neg = [r["in_catalog"] for r in splits.get(n, [])]
            if pos and neg:
                summary["abstention"].setdefault(name, {})[router] = auroc(pos, neg)
        # Fit a threshold on When2Call (keep 90% of answerable), apply it unchanged elsewhere.
        fit = [r["in_catalog"] for r in splits.get("when2call+", [])]
        if fit:
            t = threshold_at_tpr(fit, 0.9)
            thresholds = {"fit on When2Call @90% TPR": t}
            if router.startswith("tooljev"):
                thresholds["default 0.5"] = 0.5
            for label, th in thresholds.items():
                row = {"threshold": th}
                for split, kind in [("mcptoolbench", "keep"), ("mcptoolbench-lso", "abstain"),
                                    ("livemcpbench", "keep"), ("livemcpbench-lso", "abstain"),
                                    ("bfcl-irrelevance", "abstain")]:
                    rs = splits.get(split, [])
                    if rs:
                        above = _mean([r["in_catalog"] >= th for r in rs])
                        row[f"{split} {kind}"] = above if kind == "keep" else 1 - above
                summary["transfer"][f"{router} @ {label}"] = row
    (RESULTS / (f"summary_matched_{ref}.json" if ref else "summary.json")).write_text(json.dumps(summary, indent=2))
    print(render(summary))


def render(s: dict) -> str:
    out = []
    names = {"mcptoolbench": "MCPToolBench++", "livemcpbench": "LiveMCPBench", "when2call+": "When2Call"}
    for split, routers in s["routing"].items():
        cols = ["top1", "recall@5", "server@1", "category@1", "top1@50%cov", "shortlist_hit",
                "shortlist_size", "p50_ms", "p95_ms"]
        cols = [c for c in cols if any(c in m for m in routers.values())]
        n = next(iter(routers.values()))["n"]
        out.append(f"\n### {names.get(split, split)} routing (n={n})\n")
        out.append("| router | " + " | ".join(cols) + " |")
        out.append("|---" * (len(cols) + 1) + "|")
        for router, m in routers.items():
            out.append(f"| {router} | " + " | ".join(_fmt(c, m.get(c)) for c in cols) + " |")
    out.append("\n### Abstention AUROC (in-catalog score, answerable vs not)\n")
    routers = sorted({r for d in s["abstention"].values() for r in d})
    out.append("| benchmark | " + " | ".join(routers) + " |")
    out.append("|---" * (len(routers) + 1) + "|")
    for name, d in s["abstention"].items():
        out.append(f"| {name} | " + " | ".join(_fmt("auc", d.get(r)) for r in routers) + " |")
    out.append("\n### Threshold transfer (fraction handled correctly)\n")
    cols = sorted({k for row in s["transfer"].values() for k in row if k != "threshold"})
    out.append("| router, threshold | t | " + " | ".join(cols) + " |")
    out.append("|---" * (len(cols) + 2) + "|")
    for label, row in s["transfer"].items():
        out.append(f"| {label} | {row['threshold']:.3g} | " + " | ".join(_fmt('x', row.get(c)) for c in cols) + " |")
    return "\n".join(out)


def _fmt(col: str, v) -> str:
    if v is None or v != v:
        return "-"
    if col.endswith("_ms"):
        return f"{v:.0f}"
    if col == "shortlist_size":
        return f"{v:.2f}"
    return f"{v:.3f}"


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--match", default=None, help="score every router only on this router's cases")
    main(p.parse_args().match)
