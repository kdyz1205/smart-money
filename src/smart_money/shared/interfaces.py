"""Protocols (structural interfaces) that enforce module boundaries.

Modules depend on these protocols, never on concrete implementations.
This allows independent testing and prevents circular imports.
"""

from __future__ import annotations

from typing import Protocol

from .events import Event
from .models import (
    MarketContext,
    Signal,
    Transaction,
    TokenTransfer,
    WalletFeatures,
    WalletProfile,
)


class BlockchainClient(Protocol):
    """Abstract interface for any blockchain data provider."""

    async def get_transactions(
        self, address: str, start_block: int = 0
    ) -> list[Transaction]: ...

    async def get_token_transfers(
        self, address: str, start_block: int = 0
    ) -> list[TokenTransfer]: ...


class CollectorService(Protocol):
    """Collects on-chain data and publishes to the event bus."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def add_wallet(self, address: str) -> None: ...
    async def remove_wallet(self, address: str) -> None: ...


class AnalyzerProtocol(Protocol):
    """Analyzes wallets and identifies smart money."""

    async def analyze_wallets(self, txs: list[Transaction]) -> list[WalletProfile]: ...
    def extract_features(self, txs: list[Transaction]) -> list[WalletFeatures]: ...
    def identify_smart_money(self, profiles: list[WalletProfile]) -> list[WalletProfile]: ...


class PredictorProtocol(Protocol):
    """Predicts buy-in intent and generates signals."""

    async def predict(
        self, profiles: list[WalletProfile], features: list[WalletFeatures]
    ) -> list[Signal]: ...


class MarketDataProvider(Protocol):
    """Provides market context data (implemented by CryptoAnalysisAgent)."""

    async def get_market_context(self, token_symbol: str) -> MarketContext | None: ...


class Agent(Protocol):
    """Common lifecycle protocol for both agents."""

    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def handle_event(self, event: Event) -> None: ...
