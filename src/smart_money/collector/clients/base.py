"""Base class for blockchain API clients with shared retry/rate-limit logic."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import aiohttp

from ...shared.models import Transaction, TokenTransfer

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0


class BaseBlockchainClient(ABC):
    """Async blockchain client with built-in retry and session management."""

    def __init__(self, base_url: str, api_key: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                trust_env=True,
                timeout=aiohttp.ClientTimeout(total=15),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(self, params: dict) -> dict:
        session = await self._get_session()
        for attempt in range(_MAX_RETRIES):
            try:
                async with session.get(self._base_url, params=params) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                wait = _BACKOFF_BASE ** attempt
                logger.warning(
                    "Request failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(wait)
        raise RuntimeError(f"All {_MAX_RETRIES} retries exhausted for {self._base_url}")

    @abstractmethod
    async def get_transactions(
        self, address: str, start_block: int = 0
    ) -> list[Transaction]: ...

    @abstractmethod
    async def get_token_transfers(
        self, address: str, start_block: int = 0
    ) -> list[TokenTransfer]: ...
