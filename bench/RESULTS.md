# toolJev benchmark results

Run 2026-09-28 on an Apple M5 (24 GB). Agent: Claude Haiku 4.5 via headless
`claude -p`. Decision backend: **hosted Jev** (`jev-latest`) unless stated.
Results for [nanojev](https://github.com/VishiATChoudhary/nanojev), the local
stand-in, are kept beside them, because several design decisions were made on
it first.

Everything here is reproducible from `bench/`: `run.py` (routing),
`report.py`, `gating.py`, `agent/run_agent.py`, `agent/analyze.py`, `charts.py`,
and `hosted.sh` for every hosted-Jev run.
Raw per-case JSONL is written to `bench/results/` (git-ignored for size); superseded or contaminated runs are
moved to `bench/results/old/` with the reason in the file name. Summaries
(`report.md`, `report_matched_hosted.md`, `agents.md`, `summary*.json`, `gating_*.json`) are committed.

## Headlines

1. **Retrieval shortlists, hosted Jev picks.** Hosted Jev reranking
   retrieval's top 15 put the right tool first on 0.825 of MCPToolBench++
   queries, 0.543 of LiveMCPBench tasks and 0.985 of When2Call queries.
   Retrieval alone scored 0.725, 0.404 and 0.915 on the same queries. Jev with
   no retrieval (it picks a server, then a tool) scored 0.18, so the order
   matters: retrieve first, then decide.
2. **Hosted Jev knows when nothing fits.** Its best per-tool Noul separates
   answerable from unanswerable requests at AUROC 0.942 on When2Call near-misses
   (embeddings: 0.739) and 0.724 on LiveMCPBench (embeddings: 0.685). On
   MCPToolBench++ with the whole category removed it ties embeddings (0.894 vs
   0.906). The local stand-in only managed the near-misses.
3. **Jev handles per-item decisions nearly as well as an LLM.** Routing 400
   banking77 tickets into 10 queues, hosted Jev alone was 98.8% right; Claude
   Haiku doing every ticket itself was 99.3%. Its confidence is usable as a
   gate: the 97% of tickets it was at least 0.9 sure of were 99.7% right.
4. **In an agent, Code Mode with Jev is cheaper at scale, not faster.**
   - Routing 400 tickets: 98.8% for $0.063 with toolJev, against 99.3% for
     $0.332 with Haiku routing each ticket itself (5.3x cheaper).
   - 612 MCP tools: the right tool was called in 0.889 of tasks (lenient), against
     0.806 for Claude Code's own tool search, with 77% fewer median input tokens
     and 28% lower cost. The median task took 34 s against 18 s.
   - 87 tools (measured with the local backend): exposing every tool directly
     was cheaper and more accurate.

![abstention](../docs/abstention.png)

## 1. Routing: does the right tool come back?

Datasets:
- [MCPToolBench++](https://github.com/mcp-tool-bench/MCPToolBenchPP): 1,509
  single-call queries against one merged catalog (10 servers, 87 tools).
- [LiveMCPBench](https://github.com/icip-cas/LiveMCPBench): 94 multi-tool tasks
  against 69 servers and 525 tools. Its gold labels are sets of tools.
- [When2Call](https://huggingface.co/datasets/nvidia/When2Call): 250 answerable
  queries, each with its own 1 to 5 tools.

Hosted Jev ran on a sample: up to 200 answerable and 200 unanswerable queries
per benchmark (all 94 LiveMCPBench tasks), sequentially, with no failed calls.
Every router in this table is scored on exactly those queries
(`bench.report --match tooljev-hosted-rerank`). The MCPToolBench++ sample
came out a little easier than the full set: the current first stage scores
0.725 here and 0.698 on all 1,509 queries (`report.md`).

| router | MCPToolBench++ top-1 | recall@5 | LiveMCPBench top-1 | recall@5 | When2Call top-1 | p50 ms (MCPTB) |
|---|---|---|---|---|---|---|
| BM25 | 0.564 | 0.763 | 0.266 | 0.306 | 0.721 | 0 |
| dense mpnet | 0.630 | 0.853 | 0.309 | 0.362 | 0.900 | 10 |
| hybrid RRF, BM25 + MiniLM (first version) | 0.630 | 0.839 | 0.309 | 0.405 | 0.796 | 5 |
| hybrid RRF, BM25 + bge-base, dense 2x (current first stage) | 0.725 | 0.877 | 0.404 | 0.424 | 0.915 | 8 |
| **toolJev, hosted Jev reranks the top 15 (default)** | **0.825** | **0.955** | **0.543** | **0.499** | **0.985** | 284 |
| toolJev, local nanojev, retrieval order kept | 0.630 | 0.839 | 0.309 | 0.405 | 0.796 | 74 |
| toolJev, local nanojev reranks | 0.193 | 0.587 | 0.245 | 0.284 | 0.667 | 162 |
| toolJev v1, local Jev picks server then tool | 0.156 | 0.327 | 0.106 | 0.076 | 0.667 | 719 |

Notes:
- The local-nanojev rows ran on the MiniLM first stage, before bge-base was
  chosen, so they compare with the "first version" row.
- The first stage was chosen with `bench/lab.py`, which sweeps embedders
  (MiniLM, mpnet, bge-small, bge-base, e5-base) and fusion weights on all three
  benchmarks at once.
- MCPToolBench++ has three near-duplicate map providers, so part of the strict
  error is choosing an equivalent provider. At the category level hosted Jev is
  right on 0.990.
- The shortlist toolJev returns (0.9 probability mass, at most 5 tools) holds
  the right tool on 0.925 of MCPToolBench++ queries with 1.5 tools on average.

## 2. Abstention: does it know when nothing fits?

Scoring is threshold-free AUROC of each router's in-catalog score, answerable
against unanswerable queries, on the same sample as section 1.

How the unanswerable queries are built:
- **MCPToolBench++:** the gold server's whole category is removed (maps: all
  three providers). An earlier leave-one-server-out version was discarded,
  because a sibling server could still answer.
- **LiveMCPBench:** every server holding a gold tool is removed.
- **When2Call:** its labelled `cannot_answer` rows.

| benchmark | BM25 | dense MiniLM | dense mpnet | **hosted Jev Nouls** | local nanojev Nouls |
|---|---|---|---|---|---|
| MCPToolBench++ (category removed) | 0.772 | **0.906** | **0.907** | 0.894 | 0.688 |
| LiveMCPBench (gold servers removed) | 0.604 | 0.685 | 0.663 | **0.724** | 0.592 |
| When2Call (labelled cannot-answer) | 0.611 | 0.739 | 0.725 | **0.942** | 0.890 |

Two kinds of "no tool fits" show up:
- **Out-of-domain** (no map server left for a maps query): embedding similarity
  catches it, and so does hosted Jev.
- **Near-miss** (the offered tool is related but cannot do this particular
  thing): an entailment judgement catches it, and embeddings mostly do not.

Hosted Jev is the only signal that is good at both. With the local backend, a
logistic combination of Jev and embeddings did not beat embeddings alone (mean
0.768 vs 0.777), so it did not ship.

Abstention is a **warning, not a gate**, by default. Hiding tools on a low
score sent Haiku on 8 to 13 call detours in an early MCPToolBench++ agent run.
`abstain = "hard"` still exists.

## 3. Hosted Jev in practice

- Latency: 284 ms p50, 345 ms p95 per `search` on MCPToolBench++, run
  sequentially. Earlier probes timed out under heavy parallel load, so the
  benchmarks run hosted calls one at a time and the agent gateway caps sandbox
  concurrency at 4.
- The first key ran out of credits (HTTP 402) after an 80-query pilot. That
  pilot's runs are archived; `bench/run.py --resume` now stops on 402 and
  resumes where it stopped.
- The 80-query pilot also ran Jev judging fit without reranking
  (`tooljev-hosted-fit`). Its abstention AUROC was close to the reranking
  run's; see `report_matched_hosted.md`.

## 4. Confidence gating (400 banking77 tickets)

One Jev Choice per ticket over 10 queues, with plain-language queue
descriptions. The hybrid column assumes the escalated tickets are routed at
Haiku's measured 99.3% for $0.33 per 400 tickets, pro rata. That is optimistic,
because escalated tickets are the hard ones.

Hosted Jev:

| Jev confidence ≥ | Jev handles | Jev accuracy on those | hybrid accuracy | LLM cost for the rest |
|---|---|---|---|---|
| (none) | 100% | 0.988 | 0.988 | $0.00 |
| 0.7 | 99% | 0.992 | 0.992 | $0.004 |
| 0.9 | 97% | 0.997 | 0.997 | $0.009 |
| 0.95 | 96% | 0.997 | 0.997 | $0.012 |

Local nanojev, for comparison:

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
  `execute`; `-hosted` marks hosted Jev, otherwise the local backend.
- MCPToolBench++ ships no real tool outputs, so mocks return a generic
  success. That is identical across conditions, but it makes agents probe on
  data-seeking tasks.

### Bulk triage: route 400 banking77 tickets into 10 queues

| condition | accuracy per rep | mean accuracy | mean cost | LLM calls | wall |
|---|---|---|---|---|---|
| direct (Haiku routes each ticket itself) | 0.993, 0.993 | **0.993** | $0.332 | 7.5 | 268 s |
| **toolJev, hosted Jev** (agent writes `jev.map` code) | 0.988, 0.988 | **0.988** | **$0.063** | 11.0 | 160 s |
| toolJev, hosted Jev + gating hint | 0.983, 0.985 | 0.984 | $0.065 | 9.0 | 137 s |
| toolJev, local nanojev | 0.805, 0.672 | 0.739 | $0.091 | 11.5 | 98 s |
| toolJev, local nanojev + gating hint | 0.912, 0.833 | 0.873 | $0.064 | 10.5 | 122 s |

The agent writes the queue descriptions Jev reads. With the local model that
wording swung accuracy from 0.67 to 0.81 between reps; with hosted Jev both
reps scored 0.988.

### MCPToolBench++, 36 tasks, catalog of 612 tools

The 87-tool MCPToolBench++ catalog plus LiveMCPBench's 525 tools as distractors.

| condition | right tool (strict) | lenient | arg accuracy | median input tokens | total cost | LLM calls | median wall |
|---|---|---|---|---|---|---|---|
| Claude Code native ToolSearch | **0.750** | 0.806 | **0.796** | 51,660 | $1.60 | 6.3 | **18 s** |
| **toolJev, hosted Jev** | 0.722 | **0.889** | 0.703 | **11,742** | **$1.15** | 7.5 | 34 s |
| toolJev, local nanojev (MiniLM first stage) | 0.611 | 0.806 | 0.770 | 20,236 | $1.23 | 7.6 | 32 s |
| direct, all tools (12-task subset) | 0.833 | 0.833 | 0.883 | 586,835 | $0.98 (for 12) | 4.2 | 16 s |

How to read this:
- **Lenient scoring** accepts functional equivalents, because LiveMCPBench
  ships real duplicates (its own filesystem server, a yahoo-finance
  stock-price tool). The rules were fixed before scores were read; see
  `agent/analyze.py`.
- At n = 36 and one rep, 0.889 vs 0.806 is three tasks: suggestive, not
  settled. The token and cost differences are the robust part.
- toolJev is slower (a search round trip plus the extra `execute` turn), and
  its argument accuracy was lower (0.70 vs 0.80).
- One run in each toolJev condition hit the 240 s benchmark timeout. It is
  counted as a run; its cost is missing from the total.

### MCPToolBench++, 36 tasks, catalog of 87 tools (local backend)

| condition | right tool (strict) | lenient | median input tokens | total cost | LLM calls | median wall |
|---|---|---|---|---|---|---|
| direct | **0.917** | **0.944** | 59,304 | **$0.63** | 3.1 | 11 s |
| toolJev, local nanojev | 0.861 | 0.889 | **20,376** | $1.43 | 7.6 | 31 s |

At this size, Code Mode's extra round trips cost more than its smaller context
saves. Not rerun with hosted Jev.

## What the benchmarks changed in toolJev

Every item below came from a benchmark failure, not from the plan:

- Search went from Jev-only to retrieval-first. Jev reranks retrieval's top 15
  when the backend is hosted Jev, and not with the local stand-in, which reranks
  worse than retrieval.
- The first stage moved from BM25 + MiniLM to BM25 + bge-base with the dense
  ranking weighted 2x.
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

- Hosted Jev on the full routing sets (a 200 + 200 sample per benchmark was run)
  and on the 87-tool agent task.
- Models other than Claude Haiku 4.5 as the agent.
- Real upstream servers with real outputs (every agent run used mocks).
- Variance: the agent runs are 1 to 2 reps per condition.
