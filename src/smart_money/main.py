"""Entry point — wires all modules together and runs the system.

Architecture:
    EventBus (async message backbone)
        ├── CollectorService (polls blockchain APIs)
        ├── AnalyzerService (features → clustering → anomaly)
        ├── PredictorService (time-series → signal generation)
        ├── SmartMoneyAgent (orchestrates collector→analyzer→predictor)
        ├── CryptoAnalysisAgent (market data enrichment)
        ├── AgentCoordinator (manages agent lifecycle)
        └── FastAPI (HTTP API for visualization & control)
"""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from .api.app import create_app
from .analyzer.service import AnalyzerServiceImpl
from .collector.clients.etherscan import EtherscanClient
from .collector.clients.okx_dex import OkxDexClient
from .collector.service import CollectorServiceImpl
from .integration.agent_coordinator import AgentCoordinator
from .integration.crypto_analysis_agent import CryptoAnalysisAgent
from .integration.smart_money_agent import SmartMoneyAgent
from .predictor.service import PredictorServiceImpl
from .shared.config import Settings
from .shared.events import EventBus


async def main() -> None:
    config = Settings()

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting Smart Money Chain Agent")

    # Core event bus
    event_bus = EventBus()

    # Blockchain clients
    clients = []
    if config.etherscan_api_key:
        clients.append(EtherscanClient(api_key=config.etherscan_api_key))
    if config.okx_api_key:
        clients.append(OkxDexClient(api_key=config.okx_api_key))

    if not clients:
        logger.warning("No API keys configured — running in demo mode with no data sources")
        clients.append(EtherscanClient(api_key=""))

    # Services
    collector = CollectorServiceImpl(
        clients=clients,
        event_bus=event_bus,
        poll_interval_sec=config.poll_interval_sec,
    )
    analyzer = AnalyzerServiceImpl(event_bus=event_bus)
    predictor = PredictorServiceImpl(event_bus=event_bus)

    # Pre-load tracked wallets
    for addr in config.tracked_wallets:
        await collector.add_wallet(addr)

    # Agents
    smart_money = SmartMoneyAgent(
        event_bus=event_bus,
        collector=collector,
        analyzer=analyzer,
        predictor=predictor,
    )
    crypto_agent = CryptoAnalysisAgent(event_bus=event_bus, config=config)

    coordinator = AgentCoordinator(
        event_bus=event_bus,
        smart_money_agent=smart_money,
        crypto_agent=crypto_agent,
    )

    # API
    app = create_app(
        analyzer=analyzer,
        collector=collector,
        predictor=predictor,
        smart_money_agent=smart_money,
    )
    server_config = uvicorn.Config(
        app,
        host=config.api_host,
        port=config.api_port,
        log_level=config.log_level.lower(),
    )
    server = uvicorn.Server(server_config)

    # Run everything concurrently
    logger.info("All systems go — launching event bus, agents, collector, and API")
    await asyncio.gather(
        event_bus.run(),
        coordinator.run(),
        collector.start(),
        server.serve(),
    )


def run() -> None:
    """CLI entry point (registered in pyproject.toml)."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
