from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod


class NonceStore(ABC):
    @abstractmethod
    async def exists(self, nonce: str) -> bool:
        ...

    @abstractmethod
    async def add(self, nonce: str, ttl: int = 120) -> None:
        ...


class InMemoryNonceStore(NonceStore):
    def __init__(self) -> None:
        self._store: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def _cleanup(self) -> None:
        now = time.time()
        expired = [k for k, exp in self._store.items() if exp < now]
        for k in expired:
            del self._store[k]

    async def exists(self, nonce: str) -> bool:
        async with self._lock:
            await self._cleanup()
            return nonce in self._store

    async def add(self, nonce: str, ttl: int = 120) -> None:
        async with self._lock:
            self._store[nonce] = time.time() + ttl


class RedisNonceStore(NonceStore):
    def __init__(self, redis_url: str) -> None:
        self._redis_url = redis_url
        self._client = None

    async def _get_client(self):
        if self._client is None:
            import redis.asyncio as aioredis  # type: ignore[import-untyped]
            self._client = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._client

    async def exists(self, nonce: str) -> bool:
        client = await self._get_client()
        return bool(await client.exists(f"nonce:{nonce}"))

    async def add(self, nonce: str, ttl: int = 120) -> None:
        client = await self._get_client()
        await client.setex(f"nonce:{nonce}", ttl, "1")


_nonce_store: NonceStore | None = None


def get_nonce_store() -> NonceStore:
    global _nonce_store
    if _nonce_store is None:
        from app.config import get_settings
        settings = get_settings()
        if settings.redis_url:
            _nonce_store = RedisNonceStore(settings.redis_url)
        else:
            _nonce_store = InMemoryNonceStore()
    return _nonce_store


def set_nonce_store(store: NonceStore) -> None:
    global _nonce_store
    _nonce_store = store
