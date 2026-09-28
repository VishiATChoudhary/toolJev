"""Shared fixtures: a deterministic stand-in for Jev and two tiny upstream servers."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
from fastmcp import FastMCP

from tooljev.catalog import Catalog
from tooljev.config import Config, SandboxConfig, SearchConfig, ServerConfig


def _words(x: Any) -> set[str]:
    text = x if isinstance(x, str) else json.dumps(x)
    return {w for w in re.findall(r"[a-z]+", text.lower().replace("_", " ")) if len(w) > 2}


class FakeDecider:
    """Scores options by word overlap with the state. Deterministic, instant.

    Good enough to exercise routing logic; says nothing about Jev's accuracy.
    """

    def __init__(self, max_options: int = 255):
        self.max_options = max_options
        self.calls: list[dict[str, Any]] = []

    async def decide(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"state": state, "questions": questions})
        assert questions, "empty question set"
        sw = _words(state)
        out = {}
        for name, q in questions.items():
            if q["type"] == "noul":
                hit = sw & _words(q.get("instructions"))
                out[name] = {"type": "noul", "noul": 0.9 if hit else 0.05}
                continue
            crit = q["criteria"]
            keys = list(crit) if isinstance(crit, dict) else list(range(len(crit)))
            assert len(keys) <= self.max_options, f"{len(keys)} options > {self.max_options}"
            texts = [f"{k} {crit[k] or ''}" for k in keys] if isinstance(crit, dict) \
                else [str(c) for c in crit]
            logits = [3.0 * len(sw & _words(t)) for t in texts]
            z = sum(math.exp(v) for v in logits)
            probs = [math.exp(v) / z for v in logits]
            best = max(range(len(keys)), key=probs.__getitem__)
            if q["type"] == "choice":
                out[name] = {"type": "choice", "choice": keys[best],
                             "probabilities": dict(zip(keys, probs)), "confidence": probs[best]}
            else:
                out[name] = {"type": "score", "score": sum(i * p for i, p in enumerate(probs)),
                             "probabilities": dict(enumerate(probs)), "confidence": probs[best]}
        return out


def make_math() -> FastMCP:
    s = FastMCP("math")

    @s.tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @s.tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers together."""
        return a * b

    @s.tool(name="divide-safely")
    def divide(numerator: float, denominator: float) -> float:
        """Divide numerator by denominator."""
        if denominator == 0:
            raise ValueError("division by zero")
        return numerator / denominator

    return s


def make_tickets() -> FastMCP:
    s = FastMCP("tickets")
    tickets = [
        {"id": 1, "body": "Production outage, checkout returns 500 for every customer"},
        {"id": 2, "body": "Typo on the pricing page footer"},
        {"id": 3, "body": "Outage in EU region, dashboards down"},
        {"id": 4, "body": "Feature request: dark mode"},
    ]
    closed: list[int] = []

    @s.tool
    def list_tickets(limit: int = 50) -> list[dict]:
        """List open support tickets."""
        return [t for t in tickets if t["id"] not in closed][:limit]

    @s.tool
    def close_ticket(ticket_id: int, reason: str = "") -> dict:
        """Close a support ticket by id."""
        closed.append(ticket_id)
        return {"closed": ticket_id, "reason": reason}

    return s


def servers() -> list[ServerConfig]:
    return [
        ServerConfig("math", make_math(), description="Arithmetic: add, multiply, divide numbers."),
        ServerConfig("tickets", make_tickets(), description="Customer support tickets: list, close."),
    ]


@pytest.fixture
def decider() -> FakeDecider:
    return FakeDecider()


@pytest.fixture
async def catalog():
    async with Catalog(servers()) as c:
        yield c


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(servers=servers(), search=SearchConfig(dense_model=""), sandbox=SandboxConfig(),
                  trace_dir=tmp_path / "traces")
