"""Backend translation. Real backends are behind markers; these run offline."""

import os
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tooljev.decider import _from_nanojev, _to_nanojev, make_decider


@dataclass
class _Q:
    instructions: str
    criteria: object = None


fake_nanojev = SimpleNamespace(Choice=_Q, Score=_Q, Noul=lambda instructions: _Q(instructions))


def test_choice_without_descriptions_becomes_list():
    q = _to_nanojev(fake_nanojev, {"type": "choice", "instructions": "i", "criteria": {"a": None, "b": None}})
    assert q.criteria == ["a", "b"]


def test_choice_partial_descriptions_fall_back_to_name():
    q = _to_nanojev(fake_nanojev, {"type": "choice", "instructions": "i", "criteria": {"a": "alpha", "b": None}})
    assert q.criteria == {"a": "alpha", "b": "b"}


def test_score_answer_rekeyed_by_level_index():
    a = SimpleNamespace(score=1.2, probabilities={"low": 0.2, "high": 0.8}, confidence=0.5)
    out = _from_nanojev({"type": "score", "criteria": ["low", "high"]}, a)
    assert out["probabilities"] == {0: 0.2, 1: 0.8}


def test_unknown_backend():
    with pytest.raises(ValueError):
        make_decider("gpt")


@pytest.mark.hosted
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="needs TYPESAFE_API_KEY")
async def test_hosted_jev_roundtrip():
    d = make_decider("hosted")
    out = await d.decide("checkout is down for everyone", {
        "team": {"type": "choice", "instructions": "Which team", "criteria": {"infra": None, "design": None}},
        "urgent": {"type": "noul", "instructions": "This is urgent"},
    })
    await d.aclose()
    assert out["team"]["choice"] in {"infra", "design"}
    assert 0 <= out["urgent"]["noul"] <= 1


@pytest.mark.slow
async def test_nanojev_roundtrip():
    pytest.importorskip("nanojev")
    d = make_decider("nanojev", kind="encoder")
    out = await d.decide("checkout is down for everyone", {
        "team": {"type": "choice", "instructions": "Which team", "criteria": {"infra": "outages", "design": "fonts"}},
    })
    assert out["team"]["choice"] in {"infra", "design"}
