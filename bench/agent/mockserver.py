"""A stand-in MCP server built from a JSON spec. Logs every call for scoring.

    python mockserver.py spec.json log.jsonl

spec: {"name": str, "tools": [{"name", "description", "input_schema", "returns"?}]}
Tools answer with their `returns` value if given, else a generic success.
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult


def clean_schema(schema: Any) -> dict:
    """Benchmarks ship loose schemas ("dict", missing type); MCP clients want JSON Schema."""
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    s = json.loads(json.dumps(schema))

    def fix(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "dict":
                node["type"] = "object"
            if node.get("type") == "float":
                node["type"] = "number"
            if node.get("type") in ("int", "integer"):
                node["type"] = "integer"
            if node.get("type") == "tuple":
                node["type"] = "array"
            for v in node.values():
                fix(v)
        elif isinstance(node, list):
            for v in node:
                fix(v)

    fix(s)
    s["type"] = "object"
    s.setdefault("properties", {})
    if not isinstance(s.get("required", []), list):
        s.pop("required")
    return s


class MockTool(Tool):
    returns: Any = None
    server: str = ""
    log_path: str = ""

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        from tooljev.catalog import validate_args

        problem = validate_args(self.parameters, arguments)
        if problem:  # reject like a real server would; identical in every condition
            raise ValueError(f"invalid arguments: {problem}")
        with open(self.log_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), "server": self.server, "tool": self.name,
                                "args": arguments}) + "\n")
        # Echo the request back with an unambiguous success, so agents don't retry a
        # call that "did nothing". Identical in every benchmark condition.
        out = self.returns if self.returns is not None else {
            "status": "success", "message": f"{self.name} completed successfully.", "request": arguments}
        return ToolResult(structured_content=out if isinstance(out, dict) else {"result": out})


def build(spec: dict, log_path: str) -> FastMCP:
    app = FastMCP(spec["name"])
    for t in spec["tools"]:
        app.add_tool(MockTool(name=t["name"], description=t.get("description") or t["name"],
                              parameters=clean_schema(t.get("input_schema")),
                              returns=t.get("returns"), server=spec["name"], log_path=log_path))
    return app


if __name__ == "__main__":
    build(json.load(open(sys.argv[1])), sys.argv[2]).run(show_banner=False)
