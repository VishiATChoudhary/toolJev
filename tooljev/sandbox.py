"""`execute(code)`: run model-written Python in Monty with tools and Jev injected.

This is Code Mode crossed with a Recursive Language Model. The code sees
upstream MCP tools as `mcp.<server>.<tool>(...)`, and where an RLM would make a
sub-LLM call it makes a Jev call instead: typed, calibrated, ~100 ms, and
unable to produce anything outside the options it was given. Tool results stay
in sandbox variables; only what the code prints or returns reaches the client.

Monty has no filesystem, network or environment access. The only way out is
through the host objects built here, and every one of those calls is counted,
concurrency-limited and traced.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from pydantic_monty import (
    AsyncMonty,
    ClassInstance,
    CollectString,
    MontyCrashedError,
    MontyRuntimeError,
    MontySyntaxError,
)

from .catalog import Catalog, ToolInfo, py_name
from .config import SandboxConfig
from .decider import Decider

PRELUDE_DOC = '''\
Run Python in a sandbox (no files, network or env; stdlib subset such as
asyncio, json, re, math, datetime, collections, itertools, random) with these
objects in scope. All tool and jev calls are async:
`await` them, and use `asyncio.gather(*coros)` to run many at once (keyword
arguments such as return_exceptions are not supported; use try/except).

mcp.<server>.<tool>(**kwargs)   call an upstream MCP tool (keyword args only).
                                Find tools and signatures with `search` first.
                                Failures raise RuntimeError: catch or let fail.

jev: a fast decision model. It never writes text; it returns calibrated
probabilities over options you define. Use it instead of reasoning over each
item yourself, e.g. to triage, route, filter or rank tool results in bulk.
    await jev.choice(state, instructions, options)  -> {"choice", "probabilities", "confidence"}
        options: list of names, or {name: description}
    await jev.noul(state, instructions)             -> float probability the statement is true
    await jev.score(state, instructions, levels)    -> {"score" (fractional level index), "probabilities", "confidence"}
    await jev.ask(state, {name: question})          -> {name: answer}   several questions, one call
    await jev.map(states, {name: question})         -> [{name: answer}]  same questions over many states
        min_confidence=0.9 adds "confident": bool per item. Jev's confidence tracks its
        accuracy, so act on confident items in code and look at the rest yourself.
        build questions with jev.Choice(instructions, options),
        jev.Noul(instructions), jev.Score(instructions, levels)
    state may be a string or any JSON value (a tool result works as-is).

FINAL(value)   the value to return. Otherwise the last expression is returned.
session="name" keep variables between execute calls, like a REPL: fetch once,
               then inspect and act in later calls. Omit for a clean slate.
print(...)     also returned, truncated. Keep output small: summarise in code.
'''


@dataclass
class _Run:
    """Per-execute bookkeeping shared by every host call."""

    cfg: SandboxConfig
    calls: list[dict[str, Any]] = field(default_factory=list)
    final: Any = None
    has_final: bool = False
    sem: asyncio.Semaphore = field(init=False)

    def __post_init__(self) -> None:
        self.sem = asyncio.Semaphore(self.cfg.max_concurrency)

    async def track(self, kind: str, name: str, coro_fn, *args: Any, items: int | None = None) -> Any:
        if len(self.calls) >= self.cfg.max_calls:
            raise RuntimeError(f"call limit reached ({self.cfg.max_calls} calls per execute)")
        rec: dict[str, Any] = {"kind": kind, "name": name}
        if items is not None:
            rec["items"] = items
        self.calls.append(rec)
        t0 = time.perf_counter()
        async with self.sem:
            try:
                out = await coro_fn(*args)
                rec["ok"] = True
                return out
            except Exception as e:
                rec["ok"] = False
                rec["error"] = str(e)[:300]
                raise
            finally:
                rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)


class _Binding:
    """Points host objects at the current execute's bookkeeping.

    A named session keeps the same host objects across executes (Monty ties each
    wrapper to one object per session), so they read the live _Run through this.
    """

    def __init__(self, run: _Run):
        self.run = run


class JevAPI:
    """What sandbox code sees as `jev`."""

    def __init__(self, decider: Decider, binding: _Binding):
        self._d = decider
        self._b = binding

    @property
    def _run(self) -> _Run:
        return self._b.run

    async def _decide(self, name: str, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return await self._run.track("jev", name, self._d.decide, state, questions)

    # question builders, so code need not remember the wire format
    def Choice(self, instructions: str, options: Any) -> dict[str, Any]:
        criteria = dict(options) if isinstance(options, dict) else {str(o): None for o in options}
        return {"type": "choice", "instructions": instructions, "criteria": criteria}

    def Score(self, instructions: str, levels: list[Any]) -> dict[str, Any]:
        return {"type": "score", "instructions": instructions, "criteria": list(levels)}

    def Noul(self, instructions: str) -> dict[str, Any]:
        return {"type": "noul", "instructions": instructions}

    async def choice(self, state: Any, instructions: str, options: Any) -> dict[str, Any]:
        return (await self._decide("choice", state, {"q": self.Choice(instructions, options)}))["q"]

    async def score(self, state: Any, instructions: str, levels: list[Any]) -> dict[str, Any]:
        return (await self._decide("score", state, {"q": self.Score(instructions, levels)}))["q"]

    async def noul(self, state: Any, instructions: str) -> float:
        return (await self._decide("noul", state, {"q": self.Noul(instructions)}))["q"]["noul"]

    async def ask(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        return await self._decide("ask", state, questions)

    async def map(self, states: list[Any], questions: dict[str, Any],
                  min_confidence: float | None = None) -> list[dict[str, Any]]:
        # One deliberate bulk operation, so one call against the limit; its size has its own cap.
        # Counting items instead made a 400-ticket map fail and pushed agents into hand-batching.
        states = list(states)
        if len(states) > self._run.cfg.max_map_items:
            raise RuntimeError(f"jev.map takes at most {self._run.cfg.max_map_items} states")
        results = await self._run.track("jev", "map", self._map_all, states, questions, items=len(states))
        if min_confidence is not None:
            for r in results:
                r["confident"] = all(_confidence(a) >= min_confidence for a in r.values())
        return results

    async def _map_all(self, states: list[Any], questions: dict[str, Any]) -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(self._run.cfg.max_concurrency)

        async def one(state: Any) -> dict[str, Any]:
            async with sem:
                return await self._d.decide(state, questions)

        return list(await asyncio.gather(*(one(s) for s in states)))


def _confidence(answer: dict[str, Any]) -> float:
    """How sure an answer is: top probability for Choice/Score, max(p, 1-p) for a Noul."""
    if answer.get("type") == "noul":
        return max(answer["noul"], 1 - answer["noul"])
    return max(answer["probabilities"].values())


_JEV_METHODS = {"Choice", "Score", "Noul", "choice", "score", "noul", "ask", "map"}


def _server_object(catalog: Catalog, binding: _Binding, server: str, tools: list[ToolInfo]) -> ClassInstance:
    """A host object whose async methods are the server's tools."""

    def bind(tool: ToolInfo):
        async def method(self, *args: Any, **kwargs: Any) -> Any:
            if args:
                raise TypeError(f"mcp.{tool.path}() takes keyword arguments only")
            return await binding.run.track("mcp", tool.path, catalog.call, tool, kwargs)
        return method

    methods = {py_name(t.name): bind(t) for t in tools}
    cls = type(f"server_{py_name(server)}", (), methods)
    return ClassInstance(cls(), allowed_methods=set(methods))


class _Namespace:
    pass


class Sandbox:
    def __init__(self, catalog: Catalog, decider: Decider, cfg: SandboxConfig):
        self.catalog = catalog
        self.decider = decider
        self.cfg = cfg
        self._pool: AsyncMonty | None = None
        # name -> (checkout context, session, lock, binding); least recently used first
        self._sessions: OrderedDict[str, tuple[Any, Any, asyncio.Lock, _Binding]] = OrderedDict()

    async def __aenter__(self) -> Sandbox:
        self._pool = await AsyncMonty().__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for name in list(self._sessions):
            await self._drop(name)
        if self._pool is not None:
            await self._pool.__aexit__(*exc)

    def _limits(self) -> dict[str, Any]:
        return {"max_feed_duration_secs": self.cfg.cpu_secs,
                "max_memory": self.cfg.max_memory_mb * 1024 * 1024,
                # Monty counts every await as a suspension (default cap 1000), so a loop of a
                # few hundred awaited tool calls would hit it long before max_calls.
                "max_suspensions": max(1000, self.cfg.max_calls * 8)}

    async def _session(self, name: str) -> tuple[Any, asyncio.Lock, _Binding, bool]:
        """A named session that keeps its variables across execute calls. Returns
        (session, lock, binding, is_new); host objects are passed in only when new."""
        is_new = name not in self._sessions
        if is_new:
            while len(self._sessions) >= self.cfg.max_sessions:
                await self._drop(next(iter(self._sessions)))
            cm = self._pool.checkout(limits=self._limits())
            self._sessions[name] = (cm, await cm.__aenter__(), asyncio.Lock(), _Binding(_Run(self.cfg)))
        self._sessions.move_to_end(name)
        _, session, lock, binding = self._sessions[name]
        return session, lock, binding, is_new

    async def _drop(self, name: str) -> None:
        cm, _, _, _ = self._sessions.pop(name)
        try:
            await cm.__aexit__(None, None, None)
        except Exception:
            pass

    def _inputs(self, binding: _Binding) -> dict[str, Any]:
        ns = _Namespace()
        for server in self.catalog.server_names():
            setattr(ns, py_name(server),
                    _server_object(self.catalog, binding, server, self.catalog.tools(server)))
        return {
            "mcp": ClassInstance(ns, eager_attrs="all"),
            "jev": ClassInstance(JevAPI(self.decider, binding), allowed_methods=_JEV_METHODS),
        }

    async def _feed(self, sess: Any, code: str, inputs: dict[str, Any] | None, final, out) -> Any:
        return await asyncio.wait_for(
            sess.feed_run(code, inputs=inputs, external_lookup={"FINAL": final}, print_callback=out),
            timeout=self.cfg.timeout_secs,
        )

    async def execute(self, code: str, session: str | None = None) -> dict[str, Any]:
        """Run `code`. With `session`, variables persist across calls under that name."""
        assert self._pool is not None, "use `async with Sandbox(...)`"
        await self.catalog.refresh()
        run = _Run(self.cfg)

        def final(value: Any) -> None:
            run.final, run.has_final = value, True

        out = CollectString()
        t0 = time.perf_counter()
        result: dict[str, Any] = {"ok": True}
        poisoned = False
        try:
            if session is None:
                async with self._pool.checkout(limits=self._limits()) as sess:
                    value = await self._feed(sess, code, self._inputs(_Binding(run)), final, out)
            else:
                sess, lock, binding, is_new = await self._session(session)
                async with lock:
                    binding.run = run
                    value = await self._feed(sess, code, self._inputs(binding) if is_new else None,
                                             final, out)
            result["result"] = _jsonable(run.final if run.has_final else value)
        except MontySyntaxError as e:
            result = {"ok": False, "error": {"type": "SyntaxError", "message": str(e)}}
        except MontyRuntimeError as e:
            result = {"ok": False, "error": {"type": "RuntimeError", "message": str(e),
                                             "traceback": _tail(e.display(format="traceback"))}}
            # Monty: after a time or memory limit the heap has no guarantees; don't feed it again.
            poisoned = any(k in str(e) for k in ("TimeoutError", "MemoryError", "time limit"))
        except asyncio.TimeoutError:
            result = {"ok": False, "error": {"type": "Timeout",
                                             "message": f"execute exceeded {self.cfg.timeout_secs}s"}}
            poisoned = True
        except MontyCrashedError as e:
            result = {"ok": False, "error": {"type": "SandboxCrashed", "message": str(e)}}
            poisoned = True
        if session is not None:
            if poisoned and session in self._sessions:
                await self._drop(session)
                result["session_reset"] = True
            result["session"] = session

        result["stdout"] = _truncate(out.output, self.cfg.max_stdout_chars)
        result["calls"] = _summarise_calls(run.calls)
        result["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        result["_trace"] = run.calls  # full per-call log; stripped before returning to the client
        return result


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [{len(text) - limit} more chars truncated]"


def _tail(text: str, lines: int = 12) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _summarise_calls(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts per call name, so a 500-item jev.map doesn't flood the client."""
    summary: dict[str, dict[str, Any]] = {}
    for c in calls:
        key = f"{c['kind']}.{c['name']}"
        s = summary.setdefault(key, {"n": 0, "errors": 0, "ms_total": 0.0})
        s["n"] += 1
        s["errors"] += 0 if c.get("ok") else 1
        s["ms_total"] = round(s["ms_total"] + c.get("ms", 0.0), 1)
        if "items" in c:
            s["items"] = s.get("items", 0) + c["items"]
    return summary
