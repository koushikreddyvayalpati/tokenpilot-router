from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class JsonAnswerCache:
    def __init__(self, path: Path, ttl_seconds: int = 86_400) -> None:
        self.path = path
        self.ttl_seconds = ttl_seconds
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except json.JSONDecodeError:
            return {}

    def _write(self, data: dict[str, Any]) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, sort_keys=True, indent=2))
        tmp.replace(self.path)

    @staticmethod
    def key(prompt: str, tier: str) -> str:
        material = json.dumps({"prompt": prompt.strip(), "tier": tier}, sort_keys=True)
        return hashlib.sha256(material.encode()).hexdigest()

    def get(self, prompt: str, tier: str) -> dict[str, Any] | None:
        data = self._read()
        key = self.key(prompt, tier)
        entry = data.get(key)
        if not entry:
            return None
        if time.time() - entry.get("created_at", 0) > self.ttl_seconds:
            data.pop(key, None)
            self._write(data)
            return None
        return entry["value"]

    def set(self, prompt: str, tier: str, value: dict[str, Any]) -> None:
        data = self._read()
        data[self.key(prompt, tier)] = {"created_at": time.time(), "value": value}
        self._write(data)

    def clear(self) -> None:
        self._write({})

