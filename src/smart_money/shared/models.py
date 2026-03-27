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

    # Validation / fill-speed params
    fill_speed_percentile: float = Field(default=95.0, ge=50.0, le=99.9)
    fill_speed_stealth_threshold: float = Field(default=3.0, ge=1.0)
    fill_speed_interval_sec: float = Field(default=45.0, ge=5.0)
    volume_surge_sm_ratio: float = Field(default=0.35, ge=0.05, le=1.0)
    volume_surge_multiplier: float = Field(default=5.0, ge=2.0)
    breakout_concentration_pct: float = Field(default=3.0, ge=1.0)
    breakout_buy_sell_ratio: float = Field(default=8.0, ge=2.0)
    wallet_high_confidence_win_rate: float = Field(default=0.65, ge=0.3, le=1.0)


# ── Validation & fill-speed models ───────────────────────────────────


class FillSpeedMetrics(BaseModel):
    """Per-wallet fill-speed analysis for a specific token over a time window."""

    wallet_address: str
    token_address: str
    token_symbol: str
    window_start: datetime
    window_end: datetime
    num_trades: int
    total_volume_usd: float
    fill_speed_usd_per_sec: float
    stealth_score: float  # fill_speed / market_volume_share
    avg_trade_interval_sec: float
    is_alert: bool = False


class FillSpeedAlert(BaseModel):
    """High-speed accumulation alert — one of the hardest real-time signals."""

    alert_id: str
    timestamp: datetime
    wallet_address: str
    token_address: str
    token_symbol: str
    chain: Chain
    fill_speed_usd_per_sec: float
    stealth_score: float
    historical_percentile: float  # what percentile this speed is at
    total_volume_usd: float
    num_rapid_trades: int
    avg_interval_sec: float
    liquidity_pct: float  # volume as % of token liquidity
    metadata: dict[str, Any] = Field(default_factory=dict)


class VolumeSurge(BaseModel):
    """Short-term volume spike from smart-money wallets."""

    surge_id: str
    timestamp: datetime
    token_address: str
    token_symbol: str
    chain: Chain
    window_minutes: int  # 5 or 15
    smart_money_volume_usd: float
    total_market_volume_usd: float
    sm_volume_ratio: float  # smart-money / total
    vs_24h_avg_multiplier: float  # how many times 24h average
    net_buy_volume_usd: float
    price_change_pct: float  # price change during this window
    is_stealth_accumulation: bool  # volume spike + price flat
    contributing_wallets: list[str] = Field(default_factory=list)


class BreakoutPresignal(BaseModel):
    """Pre-breakout signal detected from smart-money behavior."""

    presignal_id: str
    timestamp: datetime
    token_address: str
    token_symbol: str
    chain: Chain
    signal_type: str  # "concentration_surge", "buy_sell_asymmetry", "coordinated_buy", "stealth_then_aggressive"
    confidence: float = Field(ge=0.0, le=1.0)
    detail: dict[str, Any] = Field(default_factory=dict)
    contributing_wallets: list[str] = Field(default_factory=list)


class BacktestResult(BaseModel):
    """Result of historical backtesting of signal quality."""

    run_id: str
    timestamp: datetime
    lookback_days: int
    total_breakouts_found: int  # tokens that had 15%+ rise in 30 min
    signals_before_breakout: int  # our signals that preceded a breakout
    signals_total: int  # total signals in the period
    precision: float  # signals that correctly preceded breakout / total signals
    recall: float  # breakouts we caught / total breakouts
    f1_score: float
    avg_lead_time_minutes: float  # how early our signal was before breakout
    per_signal_type: dict[str, dict[str, float]] = Field(default_factory=dict)


class WalletPerformanceRecord(BaseModel):
    """Rolling performance record for a smart-money wallet."""

    address: str
    chain: Chain
    total_trades_30d: int = 0
    winning_trades_30d: int = 0
    win_rate_30d: float = 0.0
    avg_pnl_pct: float = 0.0
    best_trade_pnl_pct: float = 0.0
    worst_trade_pnl_pct: float = 0.0
    avg_hold_hours: float = 0.0
    is_high_confidence: bool = False  # win_rate >= 65%
    last_updated: datetime | None = None


class SlippageMetrics(BaseModel):
    """Slippage analysis for a wallet's trades on a specific token."""

    wallet_address: str
    token_address: str
    avg_slippage_pct: float
    max_slippage_pct: float
    min_slippage_pct: float
    num_trades: int
    has_better_routing: bool = False  # low slippage + high speed = likely OTC/better router
