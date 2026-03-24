# backend/utils/cache.py

from typing import Any, Dict
from threading import Lock
import time


class SimpleCache:
    """
    A tiny in-memory cache with optional TTL support.
    Safe for development. Replace with Redis in production.
    """

    def __init__(self):
        self._store: Dict[str, Any] = {}
        self._expiry: Dict[str, float] = {}
        self._lock = Lock()

    def get(self, key: str) -> Any:
        with self._lock:
            if key in self._store:
                # Check expiration
                exp = self._expiry.get(key)
                if exp and exp < time.time():
                    # expired → delete
                    del self._store[key]
                    del self._expiry[key]
                    return None
                return self._store[key]
            return None

    def set(self, key: str, value: Any, ttl: float = 300):
        """
        ttl = 300 seconds by default
        """
        with self._lock:
            self._store[key] = value
            self._expiry[key] = time.time() + ttl


# ------------------------------------------------------
# Global cache instance
# ------------------------------------------------------
_cache = SimpleCache()


def cache_get(key: str) -> Any:
    return _cache.get(key)


def cache_set(key: str, value: Any, ttl: float = 300):
    _cache.set(key, value, ttl)
