"""The gateway: one MCP server exposing `search` and `execute` over many upstreams."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import Any

from fastmcp import FastMCP

from .catalog import Catalog
from .config import Config, load_config
from .decider import Decider, make_decider
from .sandbox import PRELUDE_DOC, Sandbox
from .search import Searcher
from .trace import write_trace

INSTRUCTIONS = """\
toolJev fronts many MCP servers through two tools.
1. `search(query)` returns the few tools that fit a request, with Python
   signatures and calibrated probabilities. Empty `tools` means nothing fits.
2. `execute(code)` runs Python that calls those tools as `mcp.<server>.<tool>()`
   and can use `jev` for fast typed decisions over data. Do bulk work in one
   execute rather than one tool call per turn.
"""


def build_app(cfg: Config, decider: Decider | None = None) -> FastMCP:
    decider = decider or make_decider(**cfg.decider)
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_app: FastMCP):
        async with Catalog(cfg.servers, ttl_secs=cfg.catalog_ttl_secs) as catalog, \
                Sandbox(catalog, decider, cfg.sandbox) as sandbox:
            if hasattr(decider, "warmup"):
                await decider.warmup()
            state["searcher"] = Searcher(catalog, decider, cfg.search)
            state["sandbox"] = sandbox
            try:
                yield
            finally:
                if hasattr(decider, "aclose"):
                    await decider.aclose()

    app = FastMCP("toolJev", instructions=INSTRUCTIONS, lifespan=lifespan)

    @app.tool
    async def search(query: str) -> dict:
        """Find the upstream tools that fit a request, described in plain language.

        Returns up to a handful of tools, each with a Python signature to call
        from `execute` and the probability it is the right one. When
        `tools` is empty, no connected server can serve the request.
        """
        result = await state["searcher"].search(query)
        write_trace(cfg.trace_dir, {"op": "search", "query": query, "result": result})
        return result

    @app.tool(description=PRELUDE_DOC)
    async def execute(code: str, session: str | None = None) -> dict:
        result = await state["sandbox"].execute(code, session=session)
        trace = result.pop("_trace")
        write_trace(cfg.trace_dir, {"op": "execute", "code": code, "calls": trace,
                                    **{k: v for k, v in result.items() if k != "calls"}})
        return result

    return app


def main() -> None:
    p = argparse.ArgumentParser(prog="tooljev", description=__doc__)
    p.add_argument("--config", required=True, help="path to config.toml")
    p.add_argument("--transport", default="stdio", choices=["stdio", "http"])
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    app = build_app(load_config(a.config))
    if a.transport == "http":
        app.run(transport="http", host=a.host, port=a.port, show_banner=False)
    else:
        app.run(show_banner=False)


if __name__ == "__main__":
    main()
