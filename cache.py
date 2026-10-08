"""Caches that keep repeat work off the slow paths.

- `ocr_cache`: OCR + extraction results on disk, keyed by the file's SHA256,
  so re-uploading the same invoice/certificate returns instantly and survives
  a restart. Documents never change under the same hash, so no expiry.
- `llm_cache`: LLM answers in memory, keyed by model + full prompt. The prompt
  carries the backend context, so changed data is a different key — a stale
  answer can't be served for data that has moved on. Short TTL anyway.
"""
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

CACHE_DIR = os.getenv("OCR_CACHE_DIR") or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
LLM_CACHE_TTL = int(os.getenv("LLM_CACHE_TTL", "900"))  # seconds
LLM_CACHE_SIZE = int(os.getenv("LLM_CACHE_SIZE", "500"))


def make_key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class DiskCache:
    def __init__(self, directory: str) -> None:
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, namespace: str, key: str) -> str:
        return os.path.join(self.directory, f"{namespace}-{key}.json")

    def get(self, namespace: str, key: str) -> Optional[Any]:
        try:
            with open(self._path(namespace, key), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        path = self._path(namespace, key)
        tmp = f"{path}.{threading.get_ident()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(value, f, default=str)
            os.replace(tmp, path)  # atomic, so a concurrent reader never sees half a file
        except OSError:
            pass  # a cache write failing must never fail the request


class TTLCache:
    def __init__(self, ttl: int, max_size: int) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._data: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._data.get(key)
            if not entry:
                return None
            expires, value = entry
            if expires < time.monotonic():
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = (time.monotonic() + self.ttl, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)


ocr_cache = DiskCache(CACHE_DIR)
llm_cache = TTLCache(LLM_CACHE_TTL, LLM_CACHE_SIZE)
