"""Pydantic data models shared across all modules and both agents.

These models form the contract between the Smart Money Agent and the
Crypto Analysis Agent. Every module imports from here — never define
domain objects locally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .constants import Chain, RiskLevel, SignalType, Trend


# ── On-chain primitives ──────────────────────────────────────────────


class Transaction(BaseModel):
    """A single on-chain transaction."""

    tx_hash: str
    chain: Chain
    from_addr: str
    to_addr: str
    value_wei: int
    token_symbol: str | None = None
    token_address: str | None = None
    block_number: int
    timestamp: datetime
    gas_used: int
    method_id: str | None = None


class TokenTransfer(BaseModel):
    """An ERC-20 / BEP-20 token transfer event."""

    tx_hash: str
    chain: Chain
    from_addr: str
    to_addr: str
    token_address: str
    token_symbol: str
    amount: float
    value_usd: float | None = None
    timestamp: datetime


# ── Wallet models ─────────────────────────────────────────────────────


class WalletProfile(BaseModel):
    """High-level profile of a wallet, enriched by the analyzer."""

    address: str
    chain: Chain
    cluster_id: int | None = None
    labels: list[str] = Field(default_factory=list)
    is_smart_money: bool = False
    smart_money_score: float = Field(default=0.0, ge=0.0, le=1.0)
    total_tx_count: int = 0
    win_rate: float | None = None
    avg_hold_duration_hours: float | None = None
    pnl_usd: float | None = None
    last_active: datetime | None = None


class WalletFeatures(BaseModel):
    """Numerical feature vector for ML models (clustering, anomaly, prediction)."""

    address: str
    tx_frequency_24h: float = 0.0
    tx_frequency_7d: float = 0.0
    avg_tx_value_usd: float = 0.0
    total_volume_usd: float = 0.0
    unique_tokens_traded: int = 0
    dex_to_cex_ratio: float = 0.0
    gas_spend_ratio: float = 0.0
    avg_hold_duration_hours: float = 0.0
    win_rate: float = 0.0
    max_single_trade_usd: float = 0.0
    inflow_outflow_ratio: float = 0.0

    def to_vector(self) -> list[float]:
        """Return feature values as a flat list for sklearn models."""
        return [
            self.tx_frequency_24h,
            self.tx_frequency_7d,
            self.avg_tx_value_usd,
            self.total_volume_usd,
            float(self.unique_tokens_traded),
            self.dex_to_cex_ratio,
            self.gas_spend_ratio,
            self.avg_hold_duration_hours,
            self.win_rate,
            self.max_single_trade_usd,
            self.inflow_outflow_ratio,
        ]


# ── Signals & predictions ────────────────────────────────────────────


class Signal(BaseModel):
    """A buy-in intent signal produced by the predictor."""

    signal_id: str
    timestamp: datetime
    token_address: str
    token_symbol: str
    chain: Chain
    signal_type: SignalType
    confidence: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel = RiskLevel.MEDIUM
    risk_score: float = Field(default=0.5, ge=0.0, le=1.0)
    contributing_wallets: list[str] = Field(default_factory=list)
    predicted_buy_volume_usd: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Recommendation(BaseModel):
    """Final output combining smart-money signal with market context."""

    signal: Signal
    market_context: MarketContext | None = None
    action: str  # "buy", "sell", "watch", "avoid"
    reasoning: str
    timestamp: datetime


# ── Market context (from Crypto Analysis Agent) ──────────────────────


class MarketContext(BaseModel):
    """Produced by the CryptoAnalysisAgent, consumed by the SmartMoneyAgent.

    This is the primary data exchange model between the two agents.
    The crypto analysis agent enriches signals with macro/market data
    so the smart money agent can make better-informed recommendations.
    """

    token_symbol: str
    token_address: str | None = None
    chain: Chain | None = None
    price_usd: float
    price_change_24h_pct: float | None = None
    volume_24h_usd: float
    market_cap_usd: float | None = None
    liquidity_usd: float | None = None
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    trend: Trend = Trend.NEUTRAL
    volatility_24h: float | None = None
    timestamp: datetime


# ── Parameter tuning ─────────────────────────────────────────────────


class AnalysisParams(BaseModel):
    """Tunable parameters exposed via the control panel (API or UI).

    Both AI auto-tuning and manual adjustment write to this model.
    """

    anomaly_contamination: float = Field(default=0.05, ge=0.01, le=0.5)
    cluster_min_samples: int = Field(default=5, ge=2)
    signal_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    prediction_window_hours: int = Field(default=24, ge=1)
    max_tracked_wallets: int = Field(default=500, ge=10)
    risk_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    volume_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    recency_weight: float = Field(default=0.2, ge=0.0, le=1.0)
