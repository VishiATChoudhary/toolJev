"""Append-only JSONL trace of every search and execute, one file per day."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def write_trace(trace_dir: Path | None, record: dict[str, Any]) -> None:
    if trace_dir is None:
        return
    trace_dir.mkdir(parents=True, exist_ok=True)
    record = {"ts": time.time(), **record}
    path = trace_dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")
