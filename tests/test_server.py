import json

from fastmcp import Client

from tooljev.server import build_app


async def test_gateway_end_to_end(config, decider):
    app = build_app(config, decider=decider)
    async with Client(app) as c:
        tools = {t.name for t in await c.list_tools()}
        assert tools == {"search", "execute"}

        found = (await c.call_tool("search", {"query": "multiply 6 by 7"})).structured_content
        assert found["tools"][0]["path"] == "math.multiply"

        ran = (await c.call_tool("execute", {"code": "await mcp.math.multiply(a=6, b=7)"})).structured_content
        assert ran["ok"] and ran["result"] == 42
        assert "_trace" not in ran

    lines = [json.loads(l) for f in config.trace_dir.glob("*.jsonl") for l in f.read_text().splitlines()]
    assert [l["op"] for l in lines] == ["search", "execute"]
    assert lines[1]["calls"][0]["name"] == "math.multiply"


def test_config_resolves_paths_relative_to_file(tmp_path):
    import sys

    from tooljev.config import load_config

    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[servers.x]\ncommand = "python"\nargs = ["up.py"]\n')
    s = load_config(cfg_file).servers[0]
    assert s.transport["cwd"] == str(tmp_path)
    assert s.transport["command"] == sys.executable


async def test_broken_upstream_is_skipped(decider):
    from tooljev.catalog import Catalog
    from tooljev.config import ServerConfig

    from .conftest import make_math

    bad = ServerConfig("bad", {"command": "/nonexistent/binary", "args": []})
    async with Catalog([ServerConfig("math", make_math()), bad]) as cat:
        assert cat.server_names() == ["math"]
        assert "bad" in cat.failed
