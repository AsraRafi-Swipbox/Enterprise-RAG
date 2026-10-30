"""Append-only JSONL audit trail. Raw query text is NOT stored by default (hash only)."""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


class AuditLog:
    def __init__(self, path: str | Path | None, store_query_text: bool = False):
        self.path = Path(path) if path else None
        self.store_query_text = store_query_text
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **event) -> None:
        if not self.path:
            return
        q = event.pop("query", None)
        if q is not None:
            event["query_sha256"] = hashlib.sha256(q.encode()).hexdigest()
            if self.store_query_text:
                event["query"] = q
        event["ts"] = time.time()
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")
