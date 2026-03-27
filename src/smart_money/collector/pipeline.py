"""Async ingestion pipeline — polls blockchain clients and publishes events."""

from __future__ import annotations

import asyncio
import logging

from ..shared.events import Event, EventBus, EventType
from ..shared.models import Transaction
from .cache import TxCache
from .clients.base import BaseBlockchainClient

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Continuously fetches new transactions for tracked wallets.

    Flow: tracked_wallets → fan-out to BlockchainClient calls → dedup → publish.
    """

    def __init__(
        self,
        clients: list[BaseBlockchainClient],
        event_bus: EventBus,
        cache: TxCache | None = None,
        poll_interval_sec: float = 15.0,
    ) -> None:
        self._clients = clients
        self._event_bus = event_bus
        self._cache = cache or TxCache()
        self._poll_interval = poll_interval_sec
        self._tracked_wallets: set[str] = set()
        self._running = False

    def add_wallet(self, address: str) -> None:
        self._tracked_wallets.add(address.lower())

    def remove_wallet(self, address: str) -> None:
        self._tracked_wallets.discard(address.lower())

    async def run(self) -> None:
        """Long-running polling loop."""
        self._running = True
        logger.info(
            "IngestionPipeline started — polling every %.1fs", self._poll_interval
        )
        try:
            while self._running:
                if self._tracked_wallets:
                    await self._poll_all()
                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            logger.info("IngestionPipeline stopped")

    async def _poll_all(self) -> None:
        """Fan-out fetch for all tracked wallets across all clients."""
        tasks = [
            self._fetch_wallet(client, addr)
            for client in self._clients
            for addr in list(self._tracked_wallets)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        new_txs: list[Transaction] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error("Fetch error: %s", result)
                continue
            for tx in result:
                if self._cache.is_new(tx.tx_hash):
                    new_txs.append(tx)

        if new_txs:
            logger.info("Publishing %d new transactions", len(new_txs))
            await self._event_bus.publish(
                Event(event_type=EventType.NEW_TRANSACTIONS, payload=new_txs)
            )

    async def _fetch_wallet(
        self, client: BaseBlockchainClient, address: str
    ) -> list[Transaction]:
        try:
            return await client.get_transactions(address)
        except Exception:
            logger.exception("Failed to fetch %s", address)
            return []

    def stop(self) -> None:
        self._running = False
