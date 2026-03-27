"""Agent coordinator — manages lifecycle and communication between agents.

The coordinator ensures both agents start/stop cleanly and that the
event bus is the sole communication channel between them.
Neither agent imports or calls the other directly.
"""

from __future__ import annotations

import asyncio
import logging

from ..shared.events import EventBus
from .crypto_analysis_agent import CryptoAnalysisAgent
from .smart_money_agent import SmartMoneyAgent

logger = logging.getLogger(__name__)


class AgentCoordinator:
    """Wires the SmartMoneyAgent and CryptoAnalysisAgent together.

    Data flow through the EventBus:

        Collector ──NEW_TRANSACTIONS──▶ SmartMoneyAgent
                                              │
                                  SIGNAL_GENERATED
                                              │
                                              ▼
                                     CryptoAnalysisAgent
                                              │
                                MARKET_CONTEXT_UPDATED
                                              │
                                              ▼
                                     SmartMoneyAgent
                                              │
                                 RECOMMENDATION_READY
                                              │
                                              ▼
                                      API / External
    """

    def __init__(
        self,
        event_bus: EventBus,
        smart_money_agent: SmartMoneyAgent,
        crypto_agent: CryptoAnalysisAgent,
    ) -> None:
        self._event_bus = event_bus
        self.smart_money_agent = smart_money_agent
        self.crypto_agent = crypto_agent

    async def start_all(self) -> None:
        """Start both agents. Order matters: crypto agent first so it's ready
        to receive signals when the smart-money agent starts processing."""
        logger.info("AgentCoordinator: starting all agents")
        await self.crypto_agent.start()
        await self.smart_money_agent.start()
        logger.info("AgentCoordinator: all agents running")

    async def stop_all(self) -> None:
        logger.info("AgentCoordinator: stopping all agents")
        await self.smart_money_agent.stop()
        await self.crypto_agent.stop()
        self._event_bus.stop()
        logger.info("AgentCoordinator: all agents stopped")

    async def run(self) -> None:
        """Run the coordinator as a long-lived task (for asyncio.gather).

        Starts agents, then launches the crypto agent's market monitor
        as a background task.
        """
        await self.start_all()
        try:
            # Run market monitoring loop
            await self.crypto_agent.monitor_markets(interval_sec=60.0)
        except asyncio.CancelledError:
            await self.stop_all()
