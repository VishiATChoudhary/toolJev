import pytest
from fastmcp import FastMCP

from tooljev.catalog import Catalog
from tooljev.config import SearchConfig, ServerConfig
from tooljev.search import Searcher, take_by_mass

from .conftest import FakeDecider


def test_take_by_mass_confident_returns_one():
    assert take_by_mass({"a": 0.95, "b": 0.05}, 0.9, 5) == [("a", 0.95)]


def test_take_by_mass_torn_returns_several_capped():
    probs = {"a": 0.3, "b": 0.3, "c": 0.2, "d": 0.2}
    assert [k for k, _ in take_by_mass(probs, 0.9, 3)] == ["a", "b", "c"]


def test_take_by_mass_never_empty():
    assert take_by_mass({"a": 0.1}, 0.9, 0) == [("a", 0.1)]


FAST = dict(dense_model="")  # BM25-only recall: no embedding model load in unit tests
HIER = SearchConfig(mode="hierarchical", abstain="hard")


async def test_retrieve_mode_routes_and_reranks(catalog, decider):
    r = await Searcher(catalog, decider, SearchConfig(rerank=True, **FAST)).search("multiply two numbers 6 and 7")
    assert r["tools"][0]["path"] == "math.multiply"
    assert r["in_catalog"] >= 0.5
    # one Jev call: a choice over the retrieved tools plus a noul per tool
    assert len(decider.calls) == 1
    qs = decider.calls[0]["questions"]
    assert "choice" in qs and "tool:math.multiply" in qs


async def test_retrieve_mode_without_rerank_uses_jev_only_for_fit(catalog, decider):
    r = await Searcher(catalog, decider, SearchConfig(rerank=False, **FAST)).search("multiply 6 and 7")
    assert r["tools"][0]["path"] == "math.multiply"
    assert "choice" not in decider.calls[0]["questions"]


async def test_retrieve_mode_abstains(catalog, decider):
    r = await Searcher(catalog, decider, SearchConfig(abstain="hard", **FAST)).search("book a flight to Paris")
    assert r["tools"] == []


async def test_soft_abstain_warns_but_returns_tools(catalog, decider):
    r = await Searcher(catalog, decider, SearchConfig(**FAST)).search("book a flight to Paris")
    assert r["tools"] and "warning" in r
    assert all("fit" in t and "probability" not in t for t in r["tools"])


async def test_recall_k_caps_candidates(catalog, decider):
    await Searcher(catalog, decider, SearchConfig(recall_k=2, **FAST)).search("add numbers")
    assert len([q for q in decider.calls[0]["questions"] if q.startswith("tool:")]) == 2


async def test_hierarchical_routes_to_server_then_tool(catalog, decider):
    r = await Searcher(catalog, decider, HIER).search("multiply two numbers 6 and 7")
    assert r["tools"][0]["path"] == "math.multiply"
    assert "async def mcp.math.multiply(*, a: float, b: float)" in r["tools"][0]["signature"]
    assert list(r["servers"]) == ["math"]
    # pass 1: server choice + a noul per server; pass 2: tool choice + a noul per tool
    assert set(decider.calls[0]["questions"]) == {"choice", "srv:math", "srv:tickets"}
    assert set(decider.calls[1]["questions"]) == {"choice", "tool:math.add", "tool:math.multiply",
                                                  "tool:math.divide_safely"}
    assert r["in_catalog"] >= 0.5


async def test_hierarchical_abstains_when_nothing_fits(catalog, decider):
    r = await Searcher(catalog, decider, HIER).search("book a flight to Paris")
    assert r["tools"] == []
    assert "reason" in r
    assert r["in_catalog"] < 0.5


async def test_hyphenated_tool_gets_python_path(catalog, decider):
    r = await Searcher(catalog, decider, SearchConfig(**FAST)).search("divide numerator by denominator")
    assert r["tools"][0]["path"] == "math.divide_safely"


def _tiny(name):
    s = FastMCP(name)
    s.tool(lambda: None, name=name, description=f"does {name}")
    return s


async def test_knockout_rounds_respect_option_cap():
    """7 servers and 31 tools against a 4-option cap: both passes need knockouts."""
    big = FastMCP("big")
    for i in range(30):
        big.tool(lambda: None, name=f"tool_{i}", description=f"generic tool number {i}")
    big.tool(lambda: None, name="weather_forecast", description="Weather forecast for a city")
    decider = FakeDecider(max_options=4)  # FakeDecider asserts the cap on every call
    others = [ServerConfig(f"misc{i}", _tiny(f"thing{i}"), description=f"misc thing {i}") for i in range(6)]
    async with Catalog([ServerConfig("big", big, description="weather and misc"), *others]) as cat:
        r = await Searcher(cat, decider, HIER).search("weather forecast for Berlin")
    assert r["tools"][0]["path"] == "big.weather_forecast"


async def test_allow_list_hides_tools():
    from .conftest import make_math

    async with Catalog([ServerConfig("math", make_math(), allow=["add"])]) as cat:
        assert [t.name for t in cat.tools()] == ["add"]


@pytest.mark.parametrize("schema,expected", [
    ({"type": "string", "enum": ["a", "b"]}, 'Literal["a", "b"]'),
    ({"anyOf": [{"type": "integer"}, {"type": "null"}]}, "int | None"),
    ({"type": "array", "items": {"type": "string"}}, "list[str]"),
])
def test_stub_types(schema, expected):
    from tooljev.catalog import _py_type

    assert _py_type(schema) == expected
