"""Upstream MCP servers: connect, list, cache, render as Python stubs, call."""

from __future__ import annotations

import asyncio
import json
import keyword
import re
import sys
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from fastmcp import Client

from .config import ServerConfig


class ToolCallError(RuntimeError):
    """An upstream tool failed. Surfaces in sandbox code as a catchable error."""


def py_name(name: str) -> str:
    """Make an MCP name usable as a Python attribute: `get-issue` -> `get_issue`."""
    s = re.sub(r"\W", "_", name)
    if not s or s[0].isdigit() or keyword.iskeyword(s):
        s = "_" + s
    return s


@dataclass(frozen=True)
class ToolInfo:
    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def path(self) -> str:
        """How sandbox code reaches it: `server.tool`."""
        return f"{py_name(self.server)}.{py_name(self.name)}"

    def summary(self, limit: int = 200) -> str:
        """First paragraph, cut to roughly `limit` English characters' worth of tokens.

        Non-ASCII characters count 4x: CJK runs about one token per character
        against about four characters per token for English, and one long
        Chinese description otherwise pads every row of an encoder batch.
        """
        first = (self.description or "").strip().split("\n\n")[0].replace("\n", " ")
        used = 0
        for i, ch in enumerate(first):
            used += 1 if ord(ch) < 128 else 4
            if used > limit:
                return first[:i]
        return first

    def stub(self) -> str:
        """A Python signature for the tool, keyword-only, with the description as docstring."""
        props = self.input_schema.get("properties", {}) or {}
        required = set(self.input_schema.get("required", []) or [])
        params = []
        for pname in sorted(props, key=lambda p: p not in required):  # required first
            ann = _py_type(props[pname])
            if pname in required:
                params.append(f"{py_name(pname)}: {ann}")
            else:
                params.append(f"{py_name(pname)}: {ann} = {_default(props[pname])}")
        sig = ", ".join(["*", *params]) if params else ""
        doc = (self.description or "").strip()
        arg_docs = [f"    {py_name(p)}: {s['description']}" for p, s in props.items()
                    if isinstance(s, dict) and s.get("description")]
        if arg_docs:
            doc += "\n\nArgs:\n" + "\n".join(arg_docs)
        doc = doc.replace('"""', "'''")
        return f'async def mcp.{self.path}({sig}):\n    """{doc}"""'


_JSON_TO_PY = {"string": "str", "integer": "int", "number": "float", "boolean": "bool",
               "array": "list", "object": "dict", "null": "None"}


def _py_type(schema: Any) -> str:
    if not isinstance(schema, dict):
        return "Any"
    if "enum" in schema:
        return "Literal[" + ", ".join(json.dumps(v) for v in schema["enum"]) + "]"
    for combo in ("anyOf", "oneOf"):
        if combo in schema:
            return " | ".join(dict.fromkeys(_py_type(s) for s in schema[combo]))
    t = schema.get("type")
    if isinstance(t, list):
        return " | ".join(_JSON_TO_PY.get(x, "Any") for x in t)
    if t == "array" and isinstance(schema.get("items"), dict):
        return f"list[{_py_type(schema['items'])}]"
    return _JSON_TO_PY.get(t, "Any")


def _default(schema: Any) -> str:
    if isinstance(schema, dict) and "default" in schema:
        return repr(schema["default"])
    return "None"


def validate_args(schema: dict[str, Any], args: dict[str, Any]) -> str | None:
    """First JSON Schema violation as a short message, or None. Unusable schemas pass."""
    import jsonschema

    if not schema:
        return None
    try:
        err = jsonschema.exceptions.best_match(jsonschema.Draft202012Validator(schema).iter_errors(args))
    except (jsonschema.exceptions.SchemaError, TypeError):
        return None
    if err is None:
        return None
    where = ".".join(str(p) for p in err.absolute_path) or "arguments"
    return f"{where}: {err.message[:300]}"


def unwrap_result(result: Any) -> Any:
    """Turn a CallToolResult into plain JSON-ish data for sandbox code."""
    sc = result.structured_content
    if sc is not None:
        # FastMCP wraps non-object returns as {"result": x}.
        if isinstance(sc, dict) and set(sc) == {"result"}:
            return sc["result"]
        return sc
    texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
    if len(texts) == 1:
        try:
            return json.loads(texts[0])
        except ValueError:
            return texts[0]
    return texts


class Catalog:
    """Holds one open client per upstream server for the gateway's lifetime."""

    def __init__(self, servers: list[ServerConfig], ttl_secs: float = 300.0):
        self.servers = {s.name: s for s in servers}
        self.ttl_secs = ttl_secs
        self._clients: dict[str, Client] = {}
        self._tools: dict[str, list[ToolInfo]] = {}
        self._fetched_at = 0.0
        self._stack = AsyncExitStack()
        self._refresh_lock = asyncio.Lock()
        self.failed: dict[str, str] = {}

    async def __aenter__(self) -> Catalog:
        for s in self.servers.values():
            transport = s.transport
            if isinstance(transport, dict) and "mcpServers" not in transport:
                transport = {"mcpServers": {s.name: transport}}
            try:
                self._clients[s.name] = await self._stack.enter_async_context(Client(transport))
            except Exception as e:
                # One broken upstream should not take the whole gateway down.
                self.failed[s.name] = str(e)
                print(f"tooljev: skipping server {s.name!r}: {e}", file=sys.stderr)
        await self.refresh(force=True)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self._stack.aclose()

    async def refresh(self, force: bool = False) -> None:
        async with self._refresh_lock:
            if not force and time.monotonic() - self._fetched_at < self.ttl_secs:
                return
            results = await asyncio.gather(
                *(self._list(name) for name in self._clients), return_exceptions=True
            )
            for name, res in zip(self._clients, results):
                # A server that fails to list keeps its last good tools.
                if not isinstance(res, BaseException):
                    self._tools[name] = res
            self._fetched_at = time.monotonic()

    async def _list(self, name: str) -> list[ToolInfo]:
        allow = set(self.servers[name].allow)
        tools = await self._clients[name].list_tools()
        return [ToolInfo(name, t.name, t.description or "", dict(t.input_schema or {}))
                for t in tools if not allow or t.name in allow]

    def tools(self, server: str | None = None) -> list[ToolInfo]:
        if server is not None:
            return list(self._tools.get(server, []))
        return [t for ts in self._tools.values() for t in ts]

    def server_names(self) -> list[str]:
        return [n for n in self.servers if self._tools.get(n)]

    def find(self, path: str) -> ToolInfo | None:
        return next((t for t in self.tools() if t.path == path), None)

    async def call(self, tool: ToolInfo, arguments: dict[str, Any]) -> Any:
        client = self._clients[tool.server]
        # Sandbox code uses python-safe argument names; map back to the schema's.
        names = {py_name(p): p for p in (tool.input_schema.get("properties") or {})}
        args = {names.get(k, k): v for k, v in arguments.items()}
        problem = validate_args(tool.input_schema, args)
        if problem:
            # Fail in the sandbox, where the code can fix itself, rather than hope the
            # upstream validates. A mock that didn't once took whole answer dicts as enums.
            raise ToolCallError(f"{tool.path}: invalid arguments: {problem}")
        try:
            result = await client.call_tool(tool.name, args, raise_on_error=False)
        except Exception as e:  # transport failure, validation, timeouts
            raise ToolCallError(f"{tool.path}: {e}") from e
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content).strip()
            raise ToolCallError(f"{tool.path}: {text or 'tool returned an error'}")
        return unwrap_result(result)
