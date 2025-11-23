"""Shared cache infrastructure for both single-process and multi-process runs.

When running sequentially, caches are kept in-process using lightweight
OrderedDict-based LRU buffers. For multiprocessing, the training harness can
provide a SyncManager-backed context via ``create_shared_context`` and
``install_context`` so workers all reuse the same cache storage without
ballooning per-process memory usage.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from multiprocessing.managers import SyncManager
from threading import RLock
from typing import Any, Callable, Dict, Hashable, Mapping, MutableMapping, Optional
import uuid

CacheKey = Hashable
CacheValueFactory = Callable[[], Any]

_MISSING = object()

_DEFAULT_MAX_SIZES: dict[str, int] = {
    "state_hash": 200_000,
    "winning_move": 200_000,
    "valid_moves": 120_000,
}

_CONTEXT_ID_KEY = "__context_id__"


@dataclass
class SharedCacheConfig:
    data: MutableMapping[CacheKey, Any]
    lock: Any  # proxied Lock
    max_size: int


class CacheInterface:
    def get_or_set(self, key: CacheKey, factory: CacheValueFactory) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def clear(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class LocalLRUCache(CacheInterface):
    def __init__(self, max_size: int) -> None:
        self._max_size = max_size
        self._data: "OrderedDict[CacheKey, Any]" = OrderedDict()
        self._lock = RLock()

    def peek(self, key: CacheKey) -> Any:
        with self._lock:
            if key in self._data:
                value = self._data.pop(key)
                self._data[key] = value
                return value
        return _MISSING

    def store(self, key: CacheKey, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            if len(self._data) > self._max_size:
                self._data.popitem(last=False)

    def get_or_set(self, key: CacheKey, factory: CacheValueFactory) -> Any:
        cached = self.peek(key)
        if cached is not _MISSING:
            return cached

        # compute outside lock to avoid blocking other readers
        value = factory()

        with self._lock:
            existing = self._data.pop(key, None)
            if existing is not None:
                self._data[key] = existing
                return existing

            self._data[key] = value
            if len(self._data) > self._max_size:
                self._data.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class SharedProxyCache(CacheInterface):
    def __init__(self, config: SharedCacheConfig) -> None:
        self._data = config.data
        self._lock = config.lock
        self._max_size = config.max_size
        local_hint = max(1024, self._max_size // 8)
        self._local = LocalLRUCache(min(self._max_size, local_hint))

    def get_or_set(self, key: CacheKey, factory: CacheValueFactory) -> Any:
        local_value = self._local.peek(key)
        if local_value is not _MISSING:
            return local_value

        try:
            shared_value = self._data[key]
        except KeyError:
            shared_value = _MISSING

        if shared_value is not _MISSING:
            self._local.store(key, shared_value)
            return shared_value

        computed = factory()

        with self._lock:
            try:
                existing = self._data[key]
            except KeyError:
                existing = _MISSING

            if existing is not _MISSING:
                self._local.store(key, existing)
                return existing

            self._data[key] = computed
            self._evict_locked()

        self._local.store(key, computed)
        return computed

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
        self._local.clear()

    def _evict_locked(self) -> None:
        while len(self._data) > self._max_size:
            victim = next(iter(self._data.keys()), None)
            if victim is None:
                break
            self._data.pop(victim, None)


_registry: dict[str, CacheInterface] = {}
_registry_lock = RLock()
_current_context: Optional[Mapping[str, Any]] = None
_current_context_id: Optional[str] = None


def _context_identifier(context: Optional[Mapping[str, Any]]) -> Optional[str]:
    if context is None:
        return None
    value = context.get(_CONTEXT_ID_KEY)
    if isinstance(value, str):
        return value
    return None


def install_context(context: Optional[Mapping[str, Any]]) -> None:
    """Install a new cache context for this process."""
    global _registry, _current_context, _current_context_id
    with _registry_lock:
        _current_context = context
        _current_context_id = _context_identifier(context)
        _registry = {}


def ensure_context(context: Optional[Mapping[str, Any]]) -> None:
    """Ensure the provided context is active in this process."""
    desired = _context_identifier(context)
    if desired == _current_context_id:
        return
    install_context(context)


def _resolve_max_size(name: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    return _DEFAULT_MAX_SIZES.get(name, 50_000)


def _make_local_cache(name: str, max_size: Optional[int]) -> CacheInterface:
    return LocalLRUCache(_resolve_max_size(name, max_size))


def _make_shared_cache(name: str, max_size: Optional[int]) -> CacheInterface:
    assert _current_context is not None
    config_dict = _current_context.get(name)
    if config_dict is None:
        raise KeyError(f"Shared context missing cache '{name}'")
    limit = max_size if max_size is not None else config_dict["max_size"]
    config = SharedCacheConfig(
        data=config_dict["data"],
        lock=config_dict["lock"],
        max_size=limit,
    )
    return SharedProxyCache(config)


def create_shared_context(
    manager: SyncManager,
    *,
    max_sizes: Optional[Mapping[str, int]] = None,
) -> Dict[str, Any]:
    context: Dict[str, Any] = {}
    for name, default_max in _DEFAULT_MAX_SIZES.items():
        limit = max_sizes[name] if max_sizes and name in max_sizes else default_max
        context[name] = {
            "data": manager.dict(),
            "lock": manager.Lock(),
            "max_size": limit,
        }
    context[_CONTEXT_ID_KEY] = uuid.uuid4().hex
    return context


def get_cache(
    name: str,
    *,
    max_size: Optional[int] = None,
    allow_shared: bool = True,
) -> CacheInterface:
    with _registry_lock:
        cache = _registry.get(name)
        if cache is not None:
            return cache
        if _current_context is None or not allow_shared:
            cache = _make_local_cache(name, max_size)
        else:
            cache = _make_shared_cache(name, max_size)
        _registry[name] = cache
        return cache


def clear_all() -> None:
    """Clear all caches in the current context."""
    with _registry_lock:
        for cache in _registry.values():
            cache.clear()


_DIGEST_SIZE = 16


def digest_bytes(data: bytes) -> bytes:
    """Return a stable digest for byte data suitable for cache keys."""
    return hashlib.blake2b(data, digest_size=_DIGEST_SIZE).digest()