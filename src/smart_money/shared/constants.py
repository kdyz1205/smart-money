"""Global constants and enums."""

from enum import Enum


class Chain(str, Enum):
    ETH = "ethereum"
    BSC = "bsc"
    ARB = "arbitrum"
    BASE = "base"
    POLYGON = "polygon"


class SignalType(str, Enum):
    ACCUMULATION = "accumulation"
    COORDINATED_BUY = "coordinated_buy"
    EARLY_ENTRY = "early_entry"
    WHALE_MOVE = "whale_move"
    SMART_EXIT = "smart_exit"


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
