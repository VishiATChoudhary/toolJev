<div align="center">

# toolJev

### Code Mode for MCP, where the sub-model is a decision model, not an LLM.

[![ci](https://github.com/VishiATChoudhary/toolJev/actions/workflows/ci.yml/badge.svg)](https://github.com/VishiATChoudhary/toolJev/actions/workflows/ci.yml)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<img src="docs/assets/showcase.gif" alt="toolJev: search over 612 MCP tools, then 400 support tickets routed in one execute call" width="100%">

*Real run, local models, no API key: `uv run python examples/showcase.py`*

</div>

---

Give an agent 600 MCP tools and it drowns in schemas. Give it one tool call per
turn and a 400-item job takes 400 calls. toolJev fronts every MCP server you
have and gives the agent two tools instead:

- **`search(query)`**: retrieval (BM25 + embeddings) shortlists the tools, and
  [Jev](https://typesafe.ai), a decision model that returns calibrated
  probabilities instead of text, says whether any of them actually fits.
- **`execute(code, session=None)`**: the agent writes Python that calls tools as
  `mcp.<server>.<tool>()`. Where a
  [Recursive Language Model](https://arxiv.org/abs/2512.24601) would call an LLM
  inside that code, toolJev calls **`jev.choice / noul / score / map`**: typed
  answers, about 100 ms each, no generation. The code runs in a
  [Monty](https://github.com/pydantic/monty) sandbox with no files, network or
  environment.

<p align="center"><img src="docs/architecture.png" alt="toolJev architecture" width="100%"></p>

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
FINAL({"unsure": unsure})
```

That is the whole idea: the LLM writes the plan once, Jev makes the per-item
calls, and only the cases Jev is unsure about come back to the LLM.

## What the benchmarks say

I ran it against MCPToolBench++, LiveMCPBench, When2Call, BFCL and live Claude
Haiku agents. Several results went against my first design, and the design
changed to match. Full tables, sample sizes and caveats:
**[bench/RESULTS.md](bench/RESULTS.md)**.

<p align="center"><img src="docs/results_card.png" alt="results summary" width="100%"></p>

- **Retrieval should pick tools; Jev should judge them.** Jev alone picking a
  server and then a tool got the right tool first 18% of the time on
  MCPToolBench++. BM25 + MiniLM got 62%.
- **Knowing when nothing fits depends on the kind of miss.** Local Jev wins on
  near-misses (When2Call AUROC 0.88 vs 0.74 for embeddings). Embeddings win
  when the request is out of domain (0.91 vs 0.72). Hosted Jev scored 0.91
  there on an 80-query pilot.
- **Jev's confidence is honest enough to gate on.** Its most confident 29% of
  400 tickets were 98.3% right. Claude Haiku doing every ticket itself: 99.3%.
- **Code Mode pays off at scale, not before.** At 612 tools it matched Claude
  Code's built-in tool search on success, with 61% fewer input tokens. At 87
  tools, plain direct tool calling was better.

<p align="center">
<img src="docs/gating.png" alt="Jev confidence vs accuracy" width="49%">
<img src="docs/abstention.png" alt="abstention AUROC by benchmark" width="49%">
</p>

## Run it

```bash
git clone https://github.com/VishiATChoudhary/toolJev && cd toolJev
uv venv && uv pip install -e ".[local,retrieval,dev]"   # local Jev (nanojev) + embedding recall
uv run python examples/showcase.py                        # the GIF above, on your machine
uv run python examples/demo.py                            # the real MCP gateway over stdio, 3 toy servers
```

Add it to Claude Code:

```bash
claude mcp add tooljev -- uv --directory /path/to/toolJev run tooljev --config examples/config.toml
```

`--transport http --port 8765` serves streamable HTTP instead of stdio. List
your own upstream servers in a TOML file like
[`examples/config.toml`](examples/config.toml).

## Backends

| `[decider] backend` | What | Needs |
|---|---|---|
| `hosted` | TypeSafe Jev via `typesafe-sdk` | `TYPESAFE_API_KEY` |
| `nanojev` | [nanojev](https://github.com/VishiATChoudhary/nanojev), local, `kind = "encoder"` or `"decoder"` | nothing (downloads a model once) |

Both implement one method, `decide(state, questions) -> answers`, in TypeSafe's
wire format, so everything above the backend is shared.

## How `search` decides

1. **Recall:** BM25 and MiniLM embeddings over every tool's description, fused
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

## Sandbox limits

Each `execute` runs in a Monty worker with no filesystem, network or
environment access. It gets a fresh worker unless it names a `session`; up to 8
sessions are kept, least recently used first out. Limits, all set under
`[sandbox]`, are sized for bulk jobs of hundreds of items:

- 120 s wall clock and 20 s of sandbox CPU
- 256 MB of memory
- 2,000 host calls; a whole `jev.map` counts as one call, up to 5,000 items
- 16 concurrent calls
- stdout truncated at 4k characters

A session that hits a time or memory limit is discarded, as Monty advises.
Tool arguments are checked against the tool's JSON Schema before the call goes
upstream. When an upstream tool fails, the error surfaces in the sandbox as a
catchable `RuntimeError`. Each search and execute, and every host call inside it, is
logged to `~/.tooljev/traces/YYYY-MM-DD.jsonl`.

The sandbox is not authorization: anything a listed upstream tool can do,
sandbox code can do. Use per-server `allow = [...]` to expose less.

## Not yet

- Filling free-text arguments. Jev cannot write them; the agent writes them in
  code.
- `llm_query` fallback inside the sandbox.
- Hosted Jev beyond an 80-query pilot: API credits ran out mid-benchmark.
- Combining embedding similarity and Jev fit into one abstention score. A
  simple fitted combination did not beat embeddings alone.

## Tests and benchmarks

```bash
uv run pytest -m "not slow"     # offline, fake decider, in-process upstreams
uv run pytest -m slow           # real nanojev
TYPESAFE_API_KEY=... uv run pytest -m hosted

uv pip install -e ".[bench]"
uv run python -m bench.run && uv run python -m bench.report      # routing + abstention
uv run python -m bench.agent.run_agent --task triage --n 40      # agent in the loop (uses `claude -p`)
```

Benchmark data is not committed: fetch it into `bench/data/` as described in
[bench/RESULTS.md](bench/RESULTS.md).

## Prior art and credit

- **Code Mode:** [Cloudflare](https://blog.cloudflare.com/code-mode-mcp/),
  [Anthropic](https://www.anthropic.com/engineering/code-execution-with-mcp),
  and [FastMCP](https://gofastmcp.com/servers/transforms/code-mode).
- **Recursive Language Models:** [Zhang, Kraska and Khattab](https://arxiv.org/abs/2512.24601).
- **Jev:** [TypeSafe](https://typesafe.ai).
  [nanojev](https://github.com/VishiATChoudhary/nanojev) is a small local
  stand-in with the same interface.
- **Sandbox:** [Monty](https://github.com/pydantic/monty), by Pydantic.
- **Tool retrieval:** [RAG-MCP](https://arxiv.org/abs/2505.03275),
  [MCP-Zero](https://arxiv.org/abs/2506.01056), and "Selection Is Retrieval,
  Abstention Is Not" ([arXiv 2609.18672](https://arxiv.org/abs/2609.18672)).
  This repo reached the same conclusion the hard way.

The full survey is in [RESEARCH.md](RESEARCH.md).
