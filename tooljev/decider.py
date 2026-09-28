"""Decision backends: the one place toolJev talks to a Jev-shaped model.

Everything above this module speaks a single wire format, the one TypeSafe's
API uses, so a backend is only a translation layer:

    question  {"type": "choice", "instructions": str, "criteria": {name: desc | None}}
              {"type": "score",  "instructions": str, "criteria": [level, ...]}
              {"type": "noul",   "instructions": str}

    answer    {"type": "choice", "choice": str, "probabilities": {name: p}, "confidence": c}
              {"type": "score",  "score": float, "probabilities": {level: p}, "confidence": c}
              {"type": "noul",   "noul": p}
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol

Question = dict[str, Any]
Answer = dict[str, Any]

# TypeSafe rejects a 256th option outright; the search layer splits around it.
HOSTED_MAX_OPTIONS = 255


class DecisionError(RuntimeError):
    """The backend could not answer. Raised into sandbox code as-is."""


class Decider(Protocol):
    max_options: int

    async def decide(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]: ...


class HostedJev:
    """TypeSafe's hosted Jev. Needs TYPESAFE_API_KEY (or api_key=)."""

    max_options = HOSTED_MAX_OPTIONS
    reranks_well = True  # measured: beats retrieval order on MCPToolBench++, LiveMCPBench, When2Call

    def __init__(self, model: str = "jev-latest", api_key: str | None = None, max_retries: int = 3,
                 timeout: float = 30.0):
        from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy

        # The SDK already backs off on 429/529 and honours Retry-After. Its 10 s default
        # timeout tripped on the API's occasional multi-second tail, hence 30 s.
        self._client = AsyncTypeSafeClient(
            api_key=api_key, model=model, retry=RetryPolicy(max_retries=max_retries), timeout=timeout
        )

    async def decide(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        from typesafe_sdk import (TypeSafeAPITimeoutError, TypeSafeError, TypeSafeInternalServerError)

        # The SDK retries 429/529 itself; gateway 503s and timeouts were observed too, and
        # it does not retry those, so they get a short backoff here.
        for attempt in range(4):
            try:
                resp = await self._client.system_one(state=state, questions=questions)
                break
            except (TypeSafeInternalServerError, TypeSafeAPITimeoutError) as e:
                if attempt == 3:
                    raise DecisionError(f"jev: {e}") from e
                await asyncio.sleep(2 ** attempt)
            except TypeSafeError as e:
                raise DecisionError(f"jev: {e}") from e
        return {name: ans.model_dump(mode="json") for name, ans in resp.answers.items()}

    async def aclose(self) -> None:
        await self._client.aclose()


class NanoJev:
    """Local nanojev: no key, no network, slower and less accurate.

    `kind="encoder"` uses the NLI encoder backend, `"decoder"` the small LM.
    The model is loaded lazily and calls are serialised, since one torch model
    is not safe to drive from several threads at once.
    """

    def __init__(self, kind: str = "decoder", **kwargs: Any):
        self.kind = kind
        self._kwargs = kwargs
        self._engine = None
        self._lock = asyncio.Lock()

    @property
    def max_options(self) -> int:
        return self._load().max_options

    async def warmup(self) -> None:
        """Load the model now rather than on the first request (~seconds)."""
        await asyncio.to_thread(self._load)

    def _load(self):
        if self._engine is None:
            import nanojev

            cls = nanojev.EncoderSystemOne if self.kind == "encoder" else nanojev.SystemOne
            kwargs = dict(self._kwargs)
            if isinstance(kwargs.get("dtype"), str):  # from TOML: dtype = "float16"
                import torch

                kwargs["dtype"] = getattr(torch, kwargs["dtype"])
            self._engine = cls(**kwargs)
        return self._engine

    async def decide(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        import nanojev

        text = state if isinstance(state, str) else _as_text(state)
        native = {name: _to_nanojev(nanojev, q) for name, q in questions.items()}
        async with self._lock:
            engine = await asyncio.to_thread(self._load)
            resp = await asyncio.to_thread(engine.decide, text, native)
        return {name: _from_nanojev(questions[name], a) for name, a in resp.answers.items()}


def _as_text(state: Any) -> str:
    import json

    return json.dumps(state, ensure_ascii=False, default=str)


def _to_nanojev(nanojev: Any, q: Question) -> Any:
    kind = q.get("type")
    instructions = str(q.get("instructions") or "")
    if kind == "choice":
        criteria = {k: (v if v is None else str(v)) for k, v in q["criteria"].items()}
        if all(v is None for v in criteria.values()):
            return nanojev.Choice(instructions=instructions, criteria=list(criteria))
        return nanojev.Choice(
            instructions=instructions, criteria={k: v or k for k, v in criteria.items()}
        )
    if kind == "score":
        return nanojev.Score(instructions=instructions, criteria=[str(c) for c in q["criteria"]])
    if kind == "noul":
        return nanojev.Noul(instructions=instructions)
    raise DecisionError(f"unknown question type {kind!r}")


def _from_nanojev(q: Question, a: Any) -> Answer:
    kind = q["type"]
    if kind == "choice":
        return {"type": kind, "choice": a.choice, "probabilities": dict(a.probabilities),
                "confidence": a.confidence}
    if kind == "score":
        # nanojev keys levels by their text; the wire format keys them by index.
        probs = {i: a.probabilities[str(c)] for i, c in enumerate(q["criteria"])}
        return {"type": kind, "score": a.score, "probabilities": probs, "confidence": a.confidence}
    return {"type": kind, "noul": a.noul}


def make_decider(backend: str, **kwargs: Any) -> Decider:
    if backend == "hosted":
        return HostedJev(**kwargs)
    if backend in ("nanojev", "local"):
        return NanoJev(**kwargs)
    raise ValueError(f"unknown decider backend {backend!r}; use 'hosted' or 'nanojev'")
