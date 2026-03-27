"""Collector service — orchestrates the ingestion pipeline."""

from __future__ import annotations

import logging

from ..shared.events import EventBus
from .cache import TxCache
from .clients.base import BaseBlockchainClient
from .pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


class CollectorServiceImpl:
    """Manages lifecycle of the ingestion pipeline and tracked wallets."""

    def __init__(
        self,
        clients: list[BaseBlockchainClient],
        event_bus: EventBus,
        poll_interval_sec: float = 15.0,
    ) -> None:
        self._cache = TxCache()
        self._pipeline = IngestionPipeline(
            clients=clients,
            event_bus=event_bus,
            cache=self._cache,
            poll_interval_sec=poll_interval_sec,
        )

    async def start(self) -> None:
        logger.info("CollectorService starting")
        await self._pipeline.run()

    async def stop(self) -> None:
        self._pipeline.stop()

    async def add_wallet(self, address: str) -> None:
        self._pipeline.add_wallet(address)
        logger.info("Now tracking wallet %s", address)

    async def remove_wallet(self, address: str) -> None:
        self._pipeline.remove_wallet(address)
        logger.info("Stopped tracking wallet %s", address)

    @property
    def tracked_wallets(self) -> set[str]:
        return self._pipeline._tracked_wallets
