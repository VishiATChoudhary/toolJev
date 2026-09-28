# toolJev: prior-art research

Compiled 2026-09-28. Goal: an MCP gateway that aggregates many MCP servers and
uses Jev (TypeSafe's non-generative decision model) to decide which tool to
call, whether to call one at all, and when to escalate.

Status key: **V** = verified against the primary page, **U** = unverified
(search summaries, vendor claims, repos found but not opened).

---

## 1. What TypeSafe itself ships for tool calling

| Fact | Status | Source |
|---|---|---|
| Official function-calling cookbook: one Choice `__tool__` over all functions, args as enum questions, call confidence = least certain judgement | V | https://docs.typesafe.ai/cookbooks/function_calling.md |
| Smart Home demo: Noul detects compound requests, LLM splits them, LLM fallback for general chat | V | https://docs.typesafe.ai/demos/smart-home.md |
| Intent routing pattern: code vs specialist LLM vs human | V | https://docs.typesafe.ai/patterns/intent-routing.md |
| Skill suggestion over 182 skills: rank all, then verify top 3 with full descriptions. Wrong loads 16.8% to 7.3% | V | https://docs.typesafe.ai/cookbooks/skill_suggestion.md |
| Pydantic AI native support (`typesafe:jev-latest`), route question over tools + output types, hard error at 256 options, `str` field escalates via `FallbackModel` | V | https://pydantic.dev/docs/ai/models/typesafe/ |
| SDKs: `pip install typesafe-sdk`, `npm install @typesafe-ai/sdk` | V | https://docs.typesafe.ai/api.md |
| `system-one-adapter-python`: drop-in client backed by LLM APIs | V | https://github.com/typesafe-ai |
| **No official MCP server or MCP guidance** | V (absence) | docs search |

### Arguments

Jev has exactly three question types (choice, score, noul). No extraction type.
Official stance: closed-set args (`Literal`, `bool`, `list[Literal]`) become
questions; **free text, numbers, dates get no question and keep the function
default**. For values in text: code proposes candidate spans, a Choice picks one
("the model chooses, code owns the string").

### API shape (V, https://docs.typesafe.ai/api.md)

```json
// request
{"state": "...", "model": "jev-latest",
 "questions": {"team": {"type": "choice", "instructions": "...",
   "criteria": {"billing": "...", "technical": "...", "sales": null}}}}
// response
{"model": "jev-1.13.0",
 "answers": {"team": {"type": "choice", "choice": "billing",
   "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
   "confidence": 0.81}},
 "usage": {"input_tokens": 318, "output_tokens": 34}}
```

Limits: 255 options per Choice (256th rejected, no automatic two-stage), 64k
tokens per request with 32k for `state`. Errors 401/422/429/529. Latency per
TypeSafe: 70-500 ms, typically ~100-150 ms. Local clone `../nanojev` exposes the
identical `decide(state, questions)` contract.

## 2. Third-party Jev + MCP projects (all U)

Many appeared within two weeks of launch:

- MCP servers wrapping Jev: `rahulrajaram/jev-mcp`, `tphakala/jev-mcp` (Go),
  `jkudish/jev-mcp`, `itsmostafa/typesafe-mcp`, others.
- Tool routers: `abhishekashokvkumar/jev-mcp-dispatcher` ("MCP tool calls with
  no LLM at all"), `jackbarunz/jev-tool-router`, `BillionsBobby/JevRouter`,
  `shimo4228/jev-skill-router`.
- Tool-call guards: `leepokai/jev-guard`, `seb4ez/jevguard-mcp`.
- Curated list: https://github.com/cobanov/awesome-jev

Implication: "Jev as an MCP server" is taken. "Jev picks a tool" is taken.
The open ground is below.

## 3. MCP gateway landscape

Every gateway filters tools one of three ways:

| Approach | Examples | Weakness |
|---|---|---|
| Static rules / RBAC / namespaces | agentgateway, MetaMCP, Kong, IBM ContextForge, 1mcp | Blind to the request |
| Embedding or BM25 top-k | LiteLLM semantic filter, ToolHive vMCP Optimizer, Portkey mcp-tool-filter | Uncalibrated scores, fixed k, no "none of these" |
| LLM searches itself via meta-tools | Anthropic/OpenAI tool search, Docker Dynamic MCP, Composio, Cloudflare/Bifrost Code Mode | Costs a model turn per lookup |

MCP spec **2026-07-28**: stateless (no initialize handshake, no sessions),
`tools/list` must carry `ttlMs` + `cacheScope` and not vary per connection,
list-changed via `subscriptions/listen`, sampling deprecated, elicitation via
`input_required`. No tool namespacing/filtering in spec yet (SEP-2084 open).
Consequence: per-session filtered tool lists are no longer clean; per-request
meta-tools (`find_tool` / `call_tool`) are the natural gateway shape.

FastMCP 3 (PrefectHQ/fastmcp): `create_proxy()`, providers-based mounting,
transforms to rename/namespace/hide tools.

## 4. Evidence that shapes the design

| Finding | Source |
|---|---|
| Full catalog in prompt: top-1 falls 0.85 to 0.12 from 10 to 7,278 tools | arXiv 2608.22695 |
| Server-then-tool hierarchy: 95.2% vs 69.2% flat at 1,648 tools | MCP-Zero, arXiv 2506.01056 |
| RAG-MCP retrieval: selection 43.1% vs 13.6%, >50% fewer tokens | arXiv 2505.03275 |
| Selection is retrieval, abstention is not: BM25 98.8% on lexical match, 51.2% on paraphrase; "not in catalog" AUC 0.697 from retrieval scores vs 0.806 encoder | arXiv 2609.18672 |
| Raw tool-call confidence ECE 0.18-0.31, learned calibrator 0.025-0.065 with ~300 labels | MICE, arXiv 2504.20168 |
| Models over-call tools; need call / ask / can't answer / answer-directly | When2Call, arXiv 2504.18851 |
| Tool-description poisoning: 36.5% avg attack success | MCPTox, arXiv 2508.14925 |
| Tool2Vec: embed tools by example queries, +27 Recall@K | arXiv 2409.02141 |

Arg filling alternatives if needed: GLiNER2 (205M, CPU, schema extraction with
per-field confidence, spans only), small FC models under constrained decoding
(Qwen3-1.7B/4B Apache-2.0; xLAM-2 is CC-BY-NC), llguidance/XGrammar.

## 5. Benchmarks for evaluating a router

| Use | Benchmark |
|---|---|
| Single-step selection accuracy | MCPToolBench++, ToolRet |
| Abstention ("should not call") | BFCL v4 irrelevance split, When2Call |
| Context-bloat savings | MCPVerse (552 tools, 140k schema tokens) |
| End-to-end with real servers | LiveMCPBench (527 tools), MCP-Bench |
| Poisoning robustness | MCPTox |

None report router latency.

## 6. The gap toolJev can fill

1. **Calibrated abstain + escalate.** Call directly when confident, expose a
   small shortlist when unsure, hand to the LLM when out of catalog. No gateway
   does this; Jev's Noul + Choice confidence is built for it.
2. **Hierarchical server-then-tool Choice** to cross the 255 cap cleanly
   (MCP-Zero shape, TypeSafe skill-suggestion shape).
3. **Pluggable backend**: hosted Jev or local nanojev behind one contract.
4. **Measured latency vs accuracy per catalog size**, which nobody publishes.
5. **Arguments stay honest**: enums via Jev, everything else explicitly
   delegated rather than defaulted silently.

---

## 7. Direction chosen: Code Mode + RLM, with Jev as the in-code primitive

### RLM (Recursive Language Models)

- Paper arXiv 2512.24601 (Zhang, Kraska, Khattab; v3 2026-05-11). Repo
  `alexzhang13/rlm` (5.6k stars), `rlm-minimal`, and `dspy.RLM`. (V)
- Mechanics: context lives in a REPL variable; root model sees metadata and
  truncated stdout; code calls `llm_query` / `llm_query_batched` (sub-LM) or
  `rlm_query` (recursive); ends with `FINAL(x)` / `FINAL_VAR(name)`. (V)
- Results: OOLONG 132k, RLM(GPT-5-mini) beats GPT-5 by 34+ points at similar
  cost; +130% over CodeAct with sub-calls (v3). (V)
- Limits: blocking sub-calls, no prefix caching, high cost/runtime variance. (V)
- Mismatch: `llm_query` returns a string. Jev returns typed probabilities, so it
  needs a new typed primitive, not a drop-in swap.

### MCP Code Mode

| Work | Key facts | Status |
|---|---|---|
| Cloudflare Code Mode (2025-09-26) | TS API generated from MCP schemas, V8 isolates, bindings keep keys out of sandbox | V |
| Cloudflare Code Mode MCP (2026-02-20) | Only `search()` + `execute()`; 2,500+ endpoints in ~1k tokens | V |
| Anthropic "Code execution with MCP" (Nov 2025) | Tools as file tree, progressive disclosure, 150k to 2k tokens | V |
| FastMCP 3.1 CodeMode transform | **Python on Monty**, 1/2/3-stage discovery, limits 30 s / 100 MB / 50 calls, pluggable `SandboxProvider` | V |
| pydantic-ai-harness Code Mode | Monty, 30 s / 256 MiB / 100 calls | V (snippets) |
| Bifrost | Token savings 58% at 96 tools to 92.8% at 508 tools | vendor claim |
| UTCP code-mode, pctx, lootbox | TS/Deno implementations | V |

Local sandbox without dialogs: **pydantic/monty** (Rust Python interpreter, no
fs/env/network, host reached only through injected functions). Alternatives:
Deno/Pyodide, Docker. RLM's default `local` mode is plain `exec`, unsafe.

### Combinations found

- Code Mode + RLM + Jev: **nothing public** (X/Twitter not verifiable).
- pydantic-ai #8472 (2026-09-18, open): proposes Jev as a tool in code mode and
  Jev judgments in `before_tool_execute` hooks. Closest precedent. (V)
- `SmayanKulkarni/RLM-research`: RLM for MCP tool discovery only. (V)
- `TypeSafeAI/jev-harness`: "LLM proposes, Jev answers, code decides"; no
  execution, no MCP. (V)
- `ProCleiton/rlm-mcp`: RLM over MCP, upstream tools not in REPL. (V)

### Failure modes to design for

- Sandbox is not authorization; enforce per-tool permissions at the host.
- Sandbox errors are vaguer than typed tool errors; return structured errors.
- Code mode not worth it under ~10 tools; always log trajectories.

### Gap toolJev fills

1. One sandbox holding both upstream MCP tools and large tool results as
   variables (RLM "context as variable" applied to tool output).
2. Jev as a typed, batched, calibrated in-code decision primitive in place of
   `llm_query`.
3. Jev-backed `search` with calibrated confidence and abstention.
