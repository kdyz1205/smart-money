"""Agent coordinator — manages lifecycle and communication between agents.

The coordinator ensures both agents start/stop cleanly and that the
event bus is the sole communication channel between them.
Neither agent imports or calls the other directly.
"""

from __future__ import annotations

import asyncio
import logging

from ..shared.events import EventBus
from ..validator.service import ValidatorService
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
                           ┌──────────────────┼──────────────────┐
                           ▼                  ▼                  ▼
                  CryptoAnalysisAgent   ValidatorService   (recorded)
                           │                  │
              MARKET_CONTEXT_UPDATED   FILL_SPEED_ALERT
                           │           VOLUME_SURGE_DETECTED
                           ▼           BREAKOUT_PRESIGNAL
                     SmartMoneyAgent          │
                           │                  ▼
                  RECOMMENDATION_READY   API / Dashboard
                           │
                           ▼
                    API / External
    """

    def __init__(
        self,
        event_bus: EventBus,
        smart_money_agent: SmartMoneyAgent,
        crypto_agent: CryptoAnalysisAgent,
        validator: ValidatorService | None = None,
    ) -> None:
        self._event_bus = event_bus
        self.smart_money_agent = smart_money_agent
        self.crypto_agent = crypto_agent
        self.validator = validator

    async def start_all(self) -> None:
        """Start all agents and services. Order matters: crypto agent first so it's ready
        to receive signals when the smart-money agent starts processing."""
        logger.info("AgentCoordinator: starting all agents")
        await self.crypto_agent.start()
        await self.smart_money_agent.start()
        if self.validator:
            await self.validator.start()
        logger.info("AgentCoordinator: all agents running")

    async def stop_all(self) -> None:
        logger.info("AgentCoordinator: stopping all agents")
        if self.validator:
            await self.validator.stop()
        await self.smart_money_agent.stop()
        await self.crypto_agent.stop()
        self._event_bus.stop()
        logger.info("AgentCoordinator: all agents stopped")

    async def run(self) -> None:
        """Run the coordinator as a long-lived task (for asyncio.gather).

        Starts agents, then launches the crypto agent's market monitor
        and validator's nightly backtest loop as background tasks.
        """
        await self.start_all()
        try:
            tasks = [self.crypto_agent.monitor_markets(interval_sec=60.0)]
            if self.validator:
                tasks.append(self.validator.run_backtest_loop(interval_hours=24.0))
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            await self.stop_all()
