<div align="center">

# toolJev

### Code Mode for MCP, where the sub-model is a decision model, not an LLM.

[![ci](https://github.com/VishiATChoudhary/toolJev/actions/workflows/ci.yml/badge.svg)](https://github.com/VishiATChoudhary/toolJev/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

## Results

I benchmarked toolJev on MCPToolBench++, LiveMCPBench, When2Call, BFCL and live
Claude Haiku agents. Several results went against my first design, and the
design changed to match.

<p align="center"><img src="docs/results_card.png" alt="results summary" width="100%"></p>

| Question | Result | Takeaway |
|---|---|---|
| Can a decision model pick the right tool? | Jev alone: **18%** right first. BM25 + embeddings: **70%** (MCPToolBench++, 1,509 queries) | Retrieval picks the tools; Jev judges them |
| Does it know when no tool fits? | Near-misses: Jev **0.88** AUROC vs 0.74 for embeddings. Out-of-domain: embeddings **0.91** vs 0.72 | The right signal depends on the kind of miss |
| Is Jev's confidence honest? | Its surest **29%** of 400 tickets were **98.3%** right. Claude Haiku on all of them: 99.3% | Safe to act on confident answers in code |
| Is it worth it at 612 tools? | Same task success as Claude Code's own tool search (0.81), **61% fewer** input tokens, **23% cheaper**, but slower | Pays off at scale |
| Is it worth it at 87 tools? | Direct tool calling was more accurate (0.94 vs 0.89) and cheaper | Don't bother below ~100 tools |
| Bulk work: route 400 tickets | **3.6 to 5.2x cheaper** than the LLM doing each one, but 12 to 25 points less accurate with the local model | Use confidence gating, not blind automation |

<p align="center">
<img src="docs/gating.png" alt="Jev confidence vs accuracy" width="49%">
<img src="docs/abstention.png" alt="abstention AUROC by benchmark" width="49%">
</p>

Sample sizes, methods and every caveat are in **[bench/RESULTS.md](bench/RESULTS.md)**.
The short version of the caveats:
- the agent runs use 1 to 2 reps per setup
- upstream servers are mocks
- most runs use [nanojev](https://github.com/VishiATChoudhary/nanojev), a local stand-in for Jev, because the hosted Jev credits ran out after an 80-query pilot

## What it is

<img src="docs/assets/showcase.gif" alt="toolJev: search over 612 MCP tools, then 400 support tickets routed in one execute call" width="100%">

*A real run with local models and no API key: `uv run python examples/showcase.py`*

Give an agent 600 MCP tools and it drowns in tool schemas. Give it one tool
call per turn, and a 400-item job takes 400 calls. toolJev sits in front of
every MCP server you have and gives the agent **two tools** instead:

- **`search(query)`** finds the few tools a request needs. Retrieval shortlists
  them, and [Jev](https://typesafe.ai) says whether any of them actually fits.
  Jev is a decision model: it returns calibrated probabilities over options you
  give it, never text.
- **`execute(code)`** runs Python the agent writes. The code calls tools as
  `mcp.<server>.<tool>()`, and makes per-item judgements with
  `jev.choice / noul / map`: typed answers in about 100 ms, no LLM turn. This is
  [Code Mode](https://blog.cloudflare.com/code-mode-mcp/) crossed with a
  [Recursive Language Model](https://arxiv.org/abs/2512.24601), with Jev in the
  place of the sub-LLM.

<p align="center"><img src="docs/architecture.png" alt="toolJev architecture" width="100%"></p>

The LLM writes the plan once, Jev makes the per-item calls, and only the items
Jev is unsure about come back to the LLM.

## How to use it

### 1. Install

```bash
git clone https://github.com/VishiATChoudhary/toolJev && cd toolJev
uv venv && uv pip install -e ".[local,retrieval]"
```

The two extras:
- `local` installs [nanojev](https://github.com/VishiATChoudhary/nanojev), which
  runs Jev-style decisions on your machine with no key. The model downloads
  once.
- `retrieval` adds embedding search next to BM25.

To use hosted Jev instead, set `TYPESAFE_API_KEY` and `backend = "hosted"`
(step 3).

### 2. Try it

```bash
uv run python examples/showcase.py   # the GIF above: 612 tools, 400 tickets, all local
uv run python examples/demo.py       # the real gateway over MCP stdio, 3 toy upstream servers
```

### 3. Point it at your MCP servers

Write a TOML file with one table per upstream server. Anything you'd give an MCP
client goes in it: `command`/`args`/`env` for local servers, `url`/`headers`
for remote ones.

```toml
# tooljev.toml
[decider]
backend = "nanojev"     # or "hosted", with TYPESAFE_API_KEY set
kind = "encoder"

[servers.github]
description = "GitHub repos, issues and pull requests"
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "ghp_..." }

[servers.tickets]
description = "Customer support tickets: list, assign, close"
url = "https://support.example.com/mcp"
headers = { Authorization = "Bearer ..." }
allow = ["list_tickets", "assign_ticket"]   # optional: expose only these tools
```

A one-line `description` per server helps search a lot. Every option, with its
default, is in [`examples/config.toml`](examples/config.toml).

### 4. Connect your agent

toolJev is an ordinary MCP server, so any MCP client works. For Claude Code:

```bash
claude mcp add tooljev -- uv --directory /path/to/toolJev run tooljev --config /path/to/tooljev.toml
```

For remote clients, serve streamable HTTP with
`uv run tooljev --config tooljev.toml --transport http --port 8765`.

### 5. What your agent does with it

The agent never sees your upstream tools directly. It works in two moves.

**Find tools:** `search("refund order 1234 on paypal")` returns up to 5 tools,
each with a Python signature and a `fit` score. If nothing in the catalog fits,
the result carries a warning so the agent answers on its own.

**Run code:** `execute(code)` runs Python with the found tools and `jev` in
scope:

```python
tickets = await mcp.support.list_tickets()
answers = await jev.map([t["body"] for t in tickets],
                        {"queue": jev.Choice("Which queue handles this?", QUEUES)},
                        min_confidence=0.8)
unsure = []
for t, a in zip(tickets, answers):
    if a["confident"]:                    # Jev is sure: act in code
        await mcp.support.route_ticket(ticket_id=t["id"], queue=a["queue"]["choice"])
    else:                                 # Jev is not: hand back to the agent
        unsure.append(t["id"])
FINAL({"unsure": unsure})                 # only this reaches the agent's context
```

What's in scope inside `execute`:

| call | returns |
|---|---|
| `await mcp.<server>.<tool>(**kwargs)` | the tool's result; arguments are checked against its schema first |
| `await jev.choice(state, instructions, options)` | `{"choice", "probabilities", "confidence"}` |
| `await jev.noul(state, statement)` | probability the statement is true |
| `await jev.score(state, instructions, levels)` | `{"score", "probabilities", "confidence"}` |
| `await jev.map(states, questions, min_confidence=)` | one answer dict per state, plus `"confident"` when a threshold is given |
| `FINAL(value)` / `print(...)` | what goes back to the agent (the last expression also works) |

Pass `execute(code, session="work")` to keep variables between calls, like a
REPL: fetch once, inspect, then act. You don't need to teach your agent any of
this; the tool descriptions do.

### 6. When to use it, and when not

**Use it when:**
- You connect **hundreds of tools**. At 612 tools it cut input tokens by 61%
  against Claude Code's own tool search, at the same success rate.
- Your agent does **the same judgement over many items**: triage, routing,
  filtering, labelling. Gate on Jev's confidence and let the LLM handle the rest.

**Skip it when:**
- You have **fewer than about 100 tools**. Direct tool calling was more
  accurate and cheaper at 87 tools.
- You need **every item right** and cost doesn't matter. The LLM alone was more
  accurate on triage.

## How it works

### Backends

| `[decider] backend` | What | Needs |
|---|---|---|
| `hosted` | TypeSafe Jev via `typesafe-sdk` | `TYPESAFE_API_KEY` |
| `nanojev` | [nanojev](https://github.com/VishiATChoudhary/nanojev), local, `kind = "encoder"` or `"decoder"` | nothing (downloads a model once) |

Both implement one method, `decide(state, questions) -> answers`, in TypeSafe's
wire format, so everything above the backend is shared. Use the encoder for
nanojev: the decoder could not tell in-catalog requests from out-of-catalog
ones.

### How `search` decides

1. **Recall:** BM25 and bge-base embeddings over every tool's description, fused
   by reciprocal rank. The top 15 go on. This takes a few milliseconds, and the
   embeddings are cached per tool.
2. **Fit:** one Jev call asks a Noul per candidate, "this request asks to <what
   the tool does>". Each returned tool carries that probability as `fit`.
3. **Nothing fits:** `in_catalog` is the best fit. Below `abstain_below`
   (default 0.5), the result carries a warning but still lists the tools.
   `abstain = "hard"` returns none instead. Hiding tools cost agents more in
   wrong detours than a warned shortlist did.
4. The top `max_tools` (default 5) come back in retrieval order, each with a
   Python signature.

Two alternatives are kept behind config because the benchmarks rejected them:
- `rerank = true` lets Jev reorder the candidates.
- `mode = "hierarchical"` has Jev pick a server, then a tool, with knockout
  rounds past the 255-option limit.

Search latency on MCPToolBench++ is 74 ms p50 with the local encoder.

### Sandbox

Each `execute` runs in a [Monty](https://github.com/pydantic/monty) worker with
no filesystem, network or environment access. It gets a fresh worker unless it
names a `session`; up to 8 sessions are kept, and the least recently used one
goes first.

Limits, all set under `[sandbox]`, are sized for bulk jobs of hundreds of items:
- 120 s wall clock and 20 s of sandbox CPU
- 256 MB of memory
- 2,000 host calls; a whole `jev.map` counts as one call, up to 5,000 items
- 16 concurrent calls
- stdout truncated at 4k characters

Behaviour worth knowing:
- A session that hits a time or memory limit is discarded, as Monty advises.
- When an upstream tool fails, sandbox code sees a catchable `RuntimeError`.
- Each search and execute, and every host call inside it, is logged to
  `~/.tooljev/traces/YYYY-MM-DD.jsonl`.

**The sandbox is not authorization.** Anything a listed upstream tool can do,
sandbox code can do. Use per-server `allow = [...]` to expose less.

## Not yet

- **Free-text arguments.** Jev cannot write them; the agent writes them in code.
- **An `llm_query` fallback** inside the sandbox.
- **Hosted Jev beyond an 80-query pilot.** The API credits ran out mid-benchmark.
- **One combined abstention score.** Fusing embedding similarity with Jev's fit
  is not done yet; a simple fitted combination did not beat embeddings alone.

## Reproduce

```bash
uv pip install -e ".[local,retrieval,dev,bench]"
uv run pytest -m "not slow"                                      # offline: fake decider, in-process servers
uv run pytest -m slow                                            # real nanojev
uv run python -m bench.run && uv run python -m bench.report      # routing + abstention benchmarks
uv run python -m bench.agent.run_agent --task triage --n 40      # agent in the loop (uses `claude -p`)
```

Benchmark data is not committed. Fetch it into `bench/data/` as listed in
[bench/RESULTS.md](bench/RESULTS.md#data).

## Prior art and credit

- **Code Mode:** [Cloudflare](https://blog.cloudflare.com/code-mode-mcp/),
  [Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp)
  and [FastMCP](https://gofastmcp.com/servers/transforms/code-mode).
- **Recursive Language Models:** [Zhang, Kraska and Khattab](https://arxiv.org/abs/2512.24601).
- **Jev:** [TypeSafe](https://typesafe.ai).
  [nanojev](https://github.com/VishiATChoudhary/nanojev) is a small local
  stand-in with the same interface.
- **Sandbox:** [Monty](https://github.com/pydantic/monty), by Pydantic.
- **Tool retrieval:** [RAG-MCP](https://arxiv.org/abs/2505.03275),
  [MCP-Zero](https://arxiv.org/abs/2506.01056) and "Selection Is Retrieval,
  Abstention Is Not" ([arXiv 2609.18672](https://arxiv.org/abs/2609.18672)).
  This repo reached the same conclusion the hard way.

The full prior-art survey is in [RESEARCH.md](RESEARCH.md).
