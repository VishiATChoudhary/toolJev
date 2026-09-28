"""TOML config. See examples/config.toml for every key with its default."""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ServerConfig:
    name: str
    # Anything fastmcp.Client accepts for one server: {"command", "args", "env"}
    # for stdio, or {"url", "headers"} for HTTP. Tests pass a FastMCP instance.
    transport: Any
    description: str = ""
    # Tool names to expose. Empty means every tool the server lists.
    allow: list[str] = field(default_factory=list)


@dataclass
class SearchConfig:
    # "retrieve": BM25 + dense recall, then Jev reranks and judges fit (default).
    # "hierarchical": Jev alone picks a server, then a tool. Slower and, measured
    # on MCPToolBench++, much less accurate at selection; kept for comparison.
    mode: str = "retrieve"
    recall_k: int = 15  # tools handed from retrieval to Jev
    # Jev reranks the retrieved tools. Off by default: with the local nanojev
    # encoder, reranking cut MCPToolBench++ top-1 from 0.625 to 0.233 (120-query
    # pilot). Hosted Jev has not been measured. When off, retrieval order stands
    # and Jev answers only "does any of these fit?".
    rerank: bool = False
    dense_model: str = "BAAI/bge-base-en-v1.5"  # "" for BM25 only
    server_mass: float = 0.9  # keep servers until this much probability is covered
    max_servers: int = 3
    tool_mass: float = 0.9
    max_tools: int = 5
    abstain_below: float = 0.5  # best in-catalog Noul under this counts as "nothing fits"
    abstain: str = "soft"  # "soft": warn but still return tools; "hard": return no tools


@dataclass
class SandboxConfig:
    # Sized for bulk work (hundreds of items per execute): at 30 s / 200 calls, agents
    # routing 400 tickets spent most of their turns hand-batching around the limits.
    timeout_secs: float = 120.0  # wall clock, including time waiting on tools
    cpu_secs: float = 20.0  # time spent running sandbox code itself
    max_memory_mb: int = 256
    max_calls: int = 2000  # mcp.* plus jev.* calls per execute
    max_stdout_chars: int = 4000
    max_concurrency: int = 16  # parallel host calls from one execute
    max_map_items: int = 5000  # states per jev.map; a whole map counts as one call
    max_sessions: int = 8  # named REPL sessions kept alive, least recently used evicted


@dataclass
class Config:
    servers: list[ServerConfig]
    decider: dict[str, Any] = field(default_factory=lambda: {"backend": "hosted"})
    search: SearchConfig = field(default_factory=SearchConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    catalog_ttl_secs: float = 300.0
    trace_dir: Path | None = Path.home() / ".tooljev" / "traces"


_SERVER_META = {"description", "allow"}


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text())
    servers = []
    for name, entry in raw.get("servers", {}).items():
        transport = {k: v for k, v in entry.items() if k not in _SERVER_META}
        if "command" in transport:
            # Relative args resolve against the config file, not wherever the
            # gateway happens to be launched from (MCP clients pick the cwd).
            transport.setdefault("cwd", str(path.parent))
            if transport["command"] in ("python", "python3"):
                transport["command"] = sys.executable
        servers.append(ServerConfig(name=name, transport=transport,
                                    description=entry.get("description", ""),
                                    allow=list(entry.get("allow", []))))
    trace_dir = raw.get("trace_dir", str(Config.trace_dir))
    return Config(
        servers=servers,
        decider=raw.get("decider", {"backend": "hosted"}),
        search=SearchConfig(**raw.get("search", {})),
        sandbox=SandboxConfig(**raw.get("sandbox", {})),
        catalog_ttl_secs=raw.get("catalog_ttl_secs", 300.0),
        trace_dir=Path(trace_dir).expanduser() if trace_dir else None,
    )
