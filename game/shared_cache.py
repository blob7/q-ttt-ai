"""Deprecated shared cache module.

The project now computes all board transforms and lookups directly without any
cross-process caching. This stub remains only to keep older imports from
breaking while the new pipeline settles in.
"""

from __future__ import annotations


def create_shared_context(*args, **kwargs):  # pragma: no cover - legacy shim
    raise RuntimeError("Shared cache has been removed; direct compute is now used.")


def install_context(*args, **kwargs):  # pragma: no cover - legacy shim
    return None


def ensure_context(*args, **kwargs):  # pragma: no cover - legacy shim
    return None


def get_cache(*args, **kwargs):  # pragma: no cover - legacy shim
    raise RuntimeError("Caches are no longer supported in this build.")


def clear_all():  # pragma: no cover - legacy shim
    return None


def digest_bytes(data: bytes) -> bytes:  # pragma: no cover - legacy shim
    import hashlib

    return hashlib.blake2b(data, digest_size=16).digest()
