# toolJev benchmark results

Run 2026-09-28 on an Apple M5 (24 GB). Agent: Claude Haiku 4.5 via headless
`claude -p`. Decision backend: local nanojev (DeBERTa-v3 NLI encoder, fp16)
unless stated. Hosted Jev ran out of API credits partway, so hosted numbers
are limited to one 80-query pilot (section 3).

Everything here is reproducible from `bench/`: `run.py` (routing),
`report.py`, `gating.py`, `agent/run_agent.py`, `agent/analyze.py`, `charts.py`.
Raw per-case JSONL is written to `bench/results/` (git-ignored for size); superseded or contaminated runs are
moved to `bench/results/old/` with the reason in the file name. Summaries (`report.md`, `agents.md`, `summary.json`, `gating_*.json`) are committed.

## Headlines

1. **Picking a tool is retrieval; Jev should not do it alone.** The first design
   (Jev picks a server, then a tool) scored 0.18 top-1 on MCPToolBench++ against
   0.62 for BM25 + MiniLM fusion, at 10x the latency. Reranking retrieval's
   top 15 with local Jev also hurt (0.19). toolJev now retrieves, and Jev judges.
2. **Whether anything fits is a different question, and the right signal depends
   on the kind of miss.** Local Jev's per-tool Nouls win on near-misses (When2Call,
   AUROC 0.88 vs 0.74 for embeddings). Embedding similarity wins on out-of-domain
   requests (MCPToolBench++ with the whole category removed, 0.91 vs 0.72).
   Hosted Jev reached 0.91 on an 80-query MCPToolBench++ pilot, where local Jev
   scored 0.77 on the same queries.
3. **Jev's confidence is calibrated enough to gate on.** On 400 banking77
   tickets, local Jev handles its most confident 29% at 98.3% accuracy, where
   Claude Haiku routing every ticket itself scores 99.3%.
4. **Code Mode pays off at scale, not at small scale.**
   - At 612 tools, toolJev matched Claude Code's native tool search on task
     success (0.806 each, lenient), with 61% fewer median input tokens and 23%
     lower cost.
   - At 87 tools, exposing everything directly was cheaper and more accurate.
   - Bulk triage via `jev.map` was 3.6 to 5.2x cheaper than the LLM routing
     tickets itself, but 12 to 25 points less accurate with the local model.

![abstention](../docs/abstention.png)

## 1. Routing: does the right tool come back?

Datasets:
- [MCPToolBench++](https://github.com/mcp-tool-bench/MCPToolBenchPP): 1,509
  single-call queries against one merged catalog (10 servers, 87 tools).
- [LiveMCPBench](https://github.com/icip-cas/LiveMCPBench): 94 multi-tool tasks
  against 69 servers and 525 tools. Its gold labels are sets of tools.
- [When2Call](https://huggingface.co/datasets/nvidia/When2Call): 250 answerable
  queries, each with its own 1 to 5 tools.

Routers:
- `tooljev-encoder` is the shipped default: retrieval order, with Jev used only
  for fit and abstention. Its top-1 therefore equals `hybrid-rrf` by construction.
- `-hier` and `-rerank` are the two designs it replaced.

| router | MCPToolBench++ top-1 | recall@5 | LiveMCPBench top-1 | recall@5 | When2Call top-1 | p50 ms (MCPTB) |
|---|---|---|---|---|---|---|
| BM25 | 0.535 | 0.712 | 0.266 | 0.306 | 0.716 | 0 |
| dense MiniLM | 0.547 | 0.777 | 0.287 | 0.350 | 0.880 | 5 |
| dense mpnet | 0.557 | 0.794 | 0.309 | 0.362 | **0.908** | 10 |
| hybrid RRF (BM25 + MiniLM) | **0.624** | **0.796** | **0.309** | **0.405** | 0.792 | 5 |
| **toolJev** (hybrid + Jev fit) | **0.624** | **0.796** | **0.309** | **0.405** | 0.792 | 74 |
| toolJev, Jev reranks top 15 | 0.193 | 0.587 | 0.245 | 0.284 | 0.667 | 162 |
| toolJev v1, Jev picks server then tool | 0.181 | 0.341 | 0.106 | 0.076 | 0.684 | 743 |

MCPToolBench++ has three near-duplicate map providers. At the category level,
dense retrieval reaches 0.89 top-1, so a good part of the strict error is
choosing the "wrong" equivalent provider.

## 2. Abstention: does it know when nothing fits?

Scoring is threshold-free AUROC of each router's in-catalog score, answerable
against unanswerable queries.

How the unanswerable queries are built:
- **MCPToolBench++:** the gold server's whole category is removed (maps: all
  three providers). An earlier leave-one-server-out version was discarded,
  because a sibling server could still answer.
- **LiveMCPBench:** every server holding a gold tool is removed.
- **When2Call:** its labelled `cannot_answer` rows.

| benchmark | BM25 | dense MiniLM | dense mpnet | toolJev (local Jev Nouls) | v1 hierarchical | reranking |
|---|---|---|---|---|---|---|
| MCPToolBench++ (category removed) | 0.751 | 0.913 | **0.915** | 0.721 | 0.814 | 0.712 |
| LiveMCPBench (gold servers removed) | 0.604 | **0.685** | 0.663 | 0.592 | 0.553 | 0.592 |
| When2Call (labelled cannot-answer) | 0.628 | 0.735 | 0.726 | **0.884** | 0.869 | **0.901** |

Two kinds of "no tool fits" show up:
- **Out-of-domain** (no map server left for a maps query): embedding similarity
  catches it.
- **Near-miss** (the offered tool is related but cannot do this particular
  thing): an entailment judgement catches it.

I tried combining the two signals. A logistic combination fitted on When2Call
scored 0.867 / 0.801 / 0.637 on the three benchmarks, a mean of 0.768 against
0.777 for dense alone. That is not an improvement, so it did not ship.

Consequence for the product: with the local backend, abstention is a
**warning, not a gate**. Hiding tools on a low score sent Haiku on 8 to 13 call
detours in the first MCPToolBench++ agent run. `abstain = "hard"` still exists.

## 3. Hosted Jev (80-query pilot)

Hosted Jev ran out of credits (HTTP 402) after this pilot. The pilot used the
same code, 40 answerable plus 40 category-removed MCPToolBench++ queries, and
every router was scored on exactly these 80:

| signal | AUROC |
|---|---|
| dense MiniLM | 0.941 |
| **hosted Jev Nouls** | **0.912** |
| dense mpnet | 0.907 |
| BM25 | 0.791 |
| local nanojev Nouls | 0.773 |

Hosted Jev is far better than the local stand-in on the query type where local
Jev was weakest. At n = 80 the gap to dense embeddings is within noise
(roughly ±0.06). Hosted latency was about 250 ms per search call sequentially,
with multi-second tails and timeouts under parallel load.

Everything that did not complete (full hosted routing, hosted rerank, hosted
gating, hosted agent runs) is listed under "not measured" below.

## 4. Confidence gating (400 banking77 tickets, local Jev)

One Jev Choice per ticket over 10 queues. The hybrid column assumes the
escalated tickets are routed at Haiku's measured 99.3% for $0.33 per 400
tickets, pro rata. That is optimistic, because escalated tickets are the hard
ones.

| Jev confidence ≥ | Jev handles | Jev accuracy on those | hybrid accuracy | LLM cost for the rest |
|---|---|---|---|---|
| (none) | 100% | 0.698 | 0.698 | $0.00 |
| 0.7 | 62% | 0.879 | 0.922 | $0.13 |
| 0.8 | 47% | 0.947 | 0.971 | $0.18 |
| 0.9 | 29% | 0.983 | 0.990 | $0.24 |
| 0.95 | 17% | 1.000 | 0.994 | $0.28 |

![gating](../docs/gating.png)

## 5. Agent in the loop (Claude Haiku 4.5, headless Claude Code)

Setup:
- Upstream servers are mocks built from each benchmark's own tool schemas.
  They validate arguments like real servers and log every call; scoring reads
  those logs.
- `direct` exposes every tool. `native` also enables Claude Code's built-in
  deferred-loading ToolSearch. `tooljev` exposes only toolJev's `search` and
  `execute`.
- MCPToolBench++ ships no real tool outputs, so mocks return a generic
  success. That is identical across conditions, but it makes agents probe on
  data-seeking tasks.

### MCPToolBench++, 36 tasks, catalog of 87 tools

| condition | right tool (strict) | lenient | median input tokens | total cost | LLM calls | median wall |
|---|---|---|---|---|---|---|
| direct | **0.917** | **0.944** | 59,304 | **$0.63** | 3.1 | 11 s |
| toolJev | 0.861 | 0.889 | **20,376** | $1.43 | 7.6 | 31 s |

At this size, Code Mode's extra round trips cost more than its smaller context
saves.

### Same 36 tasks, catalog of 612 tools

The catalog is the same, plus LiveMCPBench's 525 tools as distractors.

| condition | right tool (strict) | lenient | median input tokens | total cost | LLM calls | median wall |
|---|---|---|---|---|---|---|
| Claude Code native ToolSearch | 0.750 | **0.806** | 51,660 | $1.60 | 6.3 | 18 s |
| **toolJev** | 0.611 | **0.806** | **20,236** | **$1.23** | 7.6 | 32 s |
| direct, all tools (12-task subset) | 0.833 | 0.833 | 586,835 | $0.98 (for 12) | 4.2 | 16 s |

How to read this:
- **Lenient scoring** accepts functional equivalents, because LiveMCPBench
  ships real duplicates (its own filesystem server, a yahoo-finance
  stock-price tool). The rules were fixed before scores were read; see
  `agent/analyze.py`.
- **Naive direct** is accurate but costs about $0.08 per task. In one pilot
  run it also invented tool names that do not exist.
- **toolJev vs native search:** toolJev matches native on success at 61% fewer
  median input tokens and 23% lower cost, but it is slower.
- **One toolJev run hit the 240 s benchmark timeout.** It is counted as a run,
  but its cost is missing from the total.

### Bulk triage: route 400 banking77 tickets into 10 queues

| condition | accuracy per rep | mean accuracy | mean cost | LLM calls | wall |
|---|---|---|---|---|---|
| direct (Haiku routes each ticket itself) | 0.993, 0.993 | **0.993** | $0.332 | 7.5 | 268 s |
| toolJev (agent writes `jev.map` code) | 0.805, 0.672 | 0.739 | $0.091 | 11.5 | **98 s** |
| toolJev + gating hint (`min_confidence`) | 0.912, 0.833 | 0.873 | **$0.064** | 10.5 | 122 s |

Accuracy swings by rep because the agent writes the queue descriptions Jev
reads. With a zero-shot decider, the agent's prompt wording is the dominant
factor.

## What the benchmarks changed in toolJev

Every item below came from a benchmark failure, not from the plan:

- Search went from Jev-only to retrieval-first. Jev reranking is off by default.
- Abstention became a warning, and each tool carries a Jev `fit` score instead
  of a meaningless pseudo-probability.
- `jev.map` counts as one call, not one per item. It gained
  `min_confidence` gating.
- `execute(session=...)` keeps variables between calls, like RLM's REPL.
  Agents kept hitting `NameError` without it.
- Sandbox defaults were resized for bulk work: 120 s and 2,000 calls, and
  Monty's await-suspension cap was raised to match.
- Tool arguments are validated against the tool's schema before calling
  upstream. A mock that skipped validation accepted a whole Jev answer dict as
  an enum value.
- Fixes to gateway startup, the hosted backend and retrieval:
  - A broken upstream no longer kills the gateway.
  - Stdio paths resolve relative to the config file.
  - The hosted backend retries 503s and timeouts.
  - BM25 drops stopwords.
  - Tool summaries are budgeted in tokens, not characters (Chinese
    descriptions were padding every batch).

## Data

Put the files under `bench/data/<name>/`. They are not committed.

| name | source | files used |
|---|---|---|
| `mcptoolbenchpp` | [MCPToolBench++](https://github.com/mcp-tool-bench/MCPToolBenchPP) `data/*/` | the six `*_single*.json` / category files |
| `livemcpbench` | [LiveMCPBench](https://github.com/icip-cas/LiveMCPBench) | `tools/LiveMCPTool/tools.json`, `annotated_data/all_annotations.json` |
| `when2call` | [nvidia/When2Call](https://huggingface.co/datasets/nvidia/When2Call) | `test/when2call_test_mcq.jsonl` saved as `test_mcq.jsonl` |
| `bfcl` | [gorilla BFCL v4](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard/bfcl_eval/data) | `BFCL_v4_irrelevance.json` saved as `irrelevance.jsonl` |

The triage task loads banking77 through `datasets` (`legacy-datasets/banking77`).

## Not measured

- Hosted Jev beyond the 80-query pilot: full routing, reranking, gating, and
  all agent runs (credits ran out; partial runs are archived, not reported).
- Models other than Claude Haiku 4.5 as the agent.
- Real upstream servers with real outputs (every agent run used mocks).
- Variance: the agent runs are 1 to 2 reps per condition.
