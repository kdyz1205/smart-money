"""Global constants and enums."""

from enum import Enum


class Chain(str, Enum):
    ETH = "ethereum"
    BSC = "bsc"
    ARB = "arbitrum"
    BASE = "base"
    POLYGON = "polygon"
    SOL = "solana"


class SignalType(str, Enum):
    ACCUMULATION = "accumulation"
    COORDINATED_BUY = "coordinated_buy"
    EARLY_ENTRY = "early_entry"
    WHALE_MOVE = "whale_move"
    SMART_EXIT = "smart_exit"
    HIGH_SPEED_ACCUMULATION = "high_speed_accumulation"
    BREAKOUT_PRESIGNAL = "breakout_presignal"
    STEALTH_THEN_AGGRESSIVE = "stealth_then_aggressive"
    VOLUME_SURGE = "volume_surge"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Trend(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


# Thresholds
MIN_TX_COUNT_FOR_ANALYSIS = 10
DEFAULT_ANOMALY_CONTAMINATION = 0.05
DEFAULT_CLUSTER_MIN_SAMPLES = 5
DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_POLL_INTERVAL_SEC = 15.0

# Validation thresholds
FILL_SPEED_PERCENTILE = 95  # trigger when above 95th percentile
FILL_SPEED_STEALTH_THRESHOLD = 3.0
FILL_SPEED_INTERVAL_SEC = 45  # consecutive tx interval threshold
FILL_SPEED_LIQUIDITY_PCT = 1.5  # % of total liquidity

VOLUME_SURGE_SM_RATIO = 0.35  # smart-money volume / total volume
VOLUME_SURGE_MULTIPLIER = 5.0  # vs 24h average
VOLUME_SURGE_NET_BUY_PCT = 0.08  # net buy / 24h avg volume

BREAKOUT_CONCENTRATION_INCREASE_PCT = 3.0  # 300% in 30 min
BREAKOUT_BUY_SELL_RATIO = 8.0
BREAKOUT_COORDINATED_MIN_WALLETS = 3
BREAKOUT_COORDINATED_WINDOW_SEC = 600  # 10 minutes

WALLET_HIGH_CONFIDENCE_WIN_RATE = 0.65  # 65%+ win rate for HC list
BACKTEST_BREAKOUT_THRESHOLD_PCT = 15.0  # 15% rise in 30 min
BACKTEST_LOOKBACK_DAYS = 90
