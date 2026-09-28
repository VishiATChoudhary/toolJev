"""Confidence gating on the triage task: Jev takes the tickets it is sure about,
the LLM takes the rest. Measured offline on the same tickets as the agent run.

    uv run python -m bench.gating --per-intent 40

Jev answers are real (nanojev encoder). The escalated share is projected at the
direct-LLM condition's measured accuracy and per-ticket cost, read from
bench/results/agent_triage<N>_haiku.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from tooljev.decider import make_decider

from .agent.run_agent import TRIAGE_INTENTS, triage_task

RESULTS = Path(__file__).parent / "results"

# Plain-language queue descriptions, the kind an agent writes (see the agent traces).
DESCRIPTIONS = {
    "activate_my_card": "activating a new card",
    "card_arrival": "when a card will arrive, card not delivered yet",
    "lost_or_stolen_card": "card lost or stolen",
    "exchange_rate": "exchange rates and currency conversion",
    "pin_blocked": "PIN blocked or locked",
    "top_up_failed": "top-up or adding money failed",
    "transfer_not_received_by_recipient": "transfer sent but recipient has not received it",
    "cash_withdrawal_charge": "fee charged for a cash or ATM withdrawal",
    "request_refund": "asking for a refund",
    "terminate_account": "closing or deleting the account",
}


async def main(per_intent: int, backend: str = "nanojev") -> None:
    task, specs = triage_task(per_intent=per_intent)
    tickets = specs[0]["tools"][0]["returns"]
    gold = task["gold"]
    d = (make_decider("hosted") if backend == "hosted"
         else make_decider("nanojev", kind="encoder", dtype="float16"))
    q = {"queue": {"type": "choice", "instructions": "Which support queue should handle this ticket?",
                   "criteria": {k: DESCRIPTIONS[k] for k in TRIAGE_INTENTS}}}
    answers = []
    for t in tickets:
        a = (await d.decide(t["body"], q))["queue"]
        answers.append((max(a["probabilities"].values()), a["choice"] == gold[t["id"]]))

    n = len(answers)
    runs = [json.loads(l) for l in open(RESULTS / f"agent_triage{n}_haiku.jsonl")]
    direct = [r for r in runs if r["condition"] == "direct"]
    llm_acc = statistics.mean(r["accuracy"] for r in direct)
    llm_cost = statistics.mean(r["cost_usd"] for r in direct) / n  # per ticket

    rows = []
    for th in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.01]:
        kept = [ok for p, ok in answers if p >= th]
        cov = len(kept) / n
        jev_acc = sum(kept) / len(kept) if kept else float("nan")
        hybrid = (sum(kept) + llm_acc * (n - len(kept))) / n
        rows.append({"threshold": th, "jev_coverage": cov, "jev_accuracy_on_kept": jev_acc,
                     "hybrid_accuracy": hybrid, "llm_cost_usd": llm_cost * (n - len(kept))})
    out = {"n": n, "llm_accuracy": llm_acc, "llm_cost_full_usd": llm_cost * n, "rows": rows}
    out["backend"] = backend
    (RESULTS / f"gating_triage{n}_{backend}.json").write_text(json.dumps(out, indent=2))
    print(f"n={n}  LLM alone: acc {llm_acc:.3f}, ${llm_cost * n:.3f}\n")
    print("| Jev confidence >= | Jev handles | Jev acc on those | hybrid acc | LLM cost for rest |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['threshold']:.2f} | {r['jev_coverage']:.0%} | {r['jev_accuracy_on_kept']:.3f} | "
              f"{r['hybrid_accuracy']:.3f} | ${r['llm_cost_usd']:.3f} |")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-intent", type=int, default=40)
    p.add_argument("--backend", choices=["nanojev", "hosted"], default="nanojev")
    a = p.parse_args()
    asyncio.run(main(a.per_intent, a.backend))
