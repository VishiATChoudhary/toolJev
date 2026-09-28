import time

import pytest

from tooljev.config import SandboxConfig
from tooljev.sandbox import Sandbox


@pytest.fixture
async def sandbox(catalog, decider):
    async with Sandbox(catalog, decider, SandboxConfig(timeout_secs=5, cpu_secs=2)) as sb:
        yield sb


async def run(sb, code, **cfg):
    for k, v in cfg.items():
        setattr(sb.cfg, k, v)
    return await sb.execute(code)


async def test_calls_upstream_tool(sandbox):
    r = await run(sandbox, "await mcp.math.multiply(a=6, b=7)")
    assert r["ok"], r
    assert r["result"] == 42
    assert r["calls"]["mcp.math.multiply"]["n"] == 1


async def test_hyphenated_tool_and_error_is_catchable(sandbox):
    code = """
ok = await mcp.math.divide_safely(numerator=1, denominator=4)
try:
    await mcp.math.divide_safely(numerator=1, denominator=0)
    err = None
except Exception as e:
    err = str(e)
[ok, err]
"""
    r = await run(sandbox, code)
    assert r["ok"], r
    assert r["result"][0] == 0.25
    assert "division by zero" in r["result"][1]


async def test_uncaught_tool_error_is_reported(sandbox):
    r = await run(sandbox, "await mcp.math.divide_safely(numerator=1, denominator=0)")
    assert not r["ok"]
    assert "division by zero" in r["error"]["message"]


async def test_positional_args_rejected(sandbox):
    r = await run(sandbox, "await mcp.math.add(1, 2)")
    assert not r["ok"]
    assert "keyword arguments only" in r["error"]["message"]


async def test_rlm_style_triage_in_one_execute(sandbox, decider):
    """Fetch, triage every item with jev, act on some: one execute, no LLM per item."""
    code = """
tickets = await mcp.tickets.list_tickets()
answers = await jev.map([t["body"] for t in tickets],
                        {"outage": jev.Noul("This reports a production outage")})
urgent = [t["id"] for t, a in zip(tickets, answers) if a["outage"]["noul"] > 0.5]
for tid in urgent:
    await mcp.tickets.close_ticket(ticket_id=tid, reason="escalated to on-call")
print(f"{len(tickets)} tickets, {len(urgent)} escalated")
FINAL({"escalated": urgent})
"""
    r = await run(sandbox, code)
    assert r["ok"], r
    assert r["result"] == {"escalated": [1, 3]}
    assert r["stdout"].strip() == "4 tickets, 2 escalated"
    assert r["calls"]["jev.map"]["n"] == 1  # one bulk call over 4 states
    assert r["calls"]["mcp.tickets.close_ticket"]["n"] == 2


async def test_jev_primitives(sandbox):
    code = """
c = await jev.choice("the checkout page is down", "Which team", {"infra": "outages, down, errors", "design": "fonts, colours"})
n = await jev.noul("the checkout page is down", "The page is down")
s = await jev.score("minor typo", "Severity", ["minor typo", "major bug", "outage"])
a = await jev.ask("checkout down", {"t": jev.Choice("team", ["infra", "design"]), "u": jev.Noul("down")})
[c["choice"], n > 0.5, s["score"] < 1, sorted(a)]
"""
    r = await run(sandbox, code)
    assert r["ok"], r
    assert r["result"] == ["infra", True, True, ["t", "u"]]


async def test_gather_runs_host_calls_concurrently(sandbox, decider):
    import asyncio

    orig = decider.decide

    async def slow(state, questions):
        await asyncio.sleep(0.2)
        return await orig(state, questions)

    decider.decide = slow
    t = time.perf_counter()
    r = await run(sandbox, "await jev.map(['a b', 'c d', 'e f', 'g h', 'i j'], {'x': jev.Noul('a')})")
    assert r["ok"], r
    assert time.perf_counter() - t < 0.8  # 5 x 0.2s serial would be 1.0s


async def test_call_limit(sandbox):
    r = await run(sandbox, "[await mcp.math.add(a=i, b=1) for i in range(10)]", max_calls=3)
    assert not r["ok"]
    assert "call limit" in r["error"]["message"]


async def test_cpu_limit(sandbox):
    r = await run(sandbox, "while True:\n    pass", cpu_secs=0.3)
    assert not r["ok"]
    assert "time limit" in r["error"]["message"]


async def test_no_filesystem(sandbox):
    r = await run(sandbox, "import os\nos.listdir('/')")
    assert not r["ok"]
    assert "Permission" in r["error"]["message"]


async def test_syntax_error(sandbox):
    r = await run(sandbox, "x = (")
    assert r["error"]["type"] == "SyntaxError"


async def test_stdout_truncated(sandbox):
    r = await run(sandbox, "print('x' * 10000)", max_stdout_chars=100)
    assert r["ok"]
    assert "more chars truncated" in r["stdout"]
    assert len(r["stdout"]) < 200


async def test_last_expression_is_result_without_final(sandbox):
    r = await run(sandbox, "x = 2\nx * 21")
    assert r["result"] == 42


async def test_map_counts_as_one_call(sandbox):
    r = await run(sandbox, "len(await jev.map([str(i) for i in range(50)], {'x': jev.Noul('a')}))",
                  max_calls=3)
    assert r["ok"], r
    assert r["result"] == 50
    assert r["calls"]["jev.map"]["n"] == 1


async def test_map_item_cap(sandbox):
    r = await run(sandbox, "await jev.map(['a'] * 20, {'x': jev.Noul('a')})", max_map_items=10)
    assert not r["ok"]
    assert "at most 10" in r["error"]["message"]


async def test_session_keeps_variables(sandbox):
    a = await sandbox.execute("tickets = await mcp.tickets.list_tickets()\nlen(tickets)", session="s1")
    assert a["ok"] and a["session"] == "s1"
    b = await sandbox.execute("[t['id'] for t in tickets][:2]", session="s1")
    assert b["ok"], b
    assert b["result"] == [1, 2]
    # other sessions and sessionless calls start clean
    c = await sandbox.execute("tickets", session="s2")
    assert not c["ok"] and "NameError" in c["error"]["message"]
    d = await sandbox.execute("tickets")
    assert not d["ok"]


async def test_session_survives_ordinary_errors_but_not_timeouts(sandbox):
    await sandbox.execute("x = 1", session="s")
    e = await sandbox.execute("1/0", session="s")
    assert not e["ok"] and not e.get("session_reset")
    assert (await sandbox.execute("x", session="s"))["result"] == 1
    sandbox.cfg.cpu_secs = 0.3
    t = await sandbox.execute("while True:\n    pass", session="s")
    assert t["session_reset"]
    assert not (await sandbox.execute("x", session="s"))["ok"]  # fresh session


async def test_sessions_evicted_lru(sandbox):
    sandbox.cfg.max_sessions = 2
    for name in ("a", "b", "c"):
        await sandbox.execute(f"v = '{name}'", session=name)
    assert not (await sandbox.execute("v", session="a"))["ok"]  # evicted
    assert (await sandbox.execute("v", session="c"))["result"] == "c"


async def test_map_min_confidence_flags_items(sandbox):
    code = """
rs = await jev.map(["outage outage outage", "zzz"],
                   {"t": jev.Choice("team", {"infra": "outage", "design": "fonts"})}, min_confidence=0.9)
[r["confident"] for r in rs]
"""
    r = await run(sandbox, code)
    assert r["ok"], r
    assert r["result"] == [True, False]


async def test_arguments_validated_against_schema(sandbox):
    r = await run(sandbox, "await mcp.math.add(a={'choice': 1}, b=2)")
    assert not r["ok"]
    assert "invalid arguments" in r["error"]["message"] and "a:" in r["error"]["message"]
