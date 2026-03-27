"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field

from .constants import (
    DEFAULT_ANOMALY_CONTAMINATION,
    DEFAULT_CLUSTER_MIN_SAMPLES,
    DEFAULT_POLL_INTERVAL_SEC,
    DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD,
)


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # API keys
    etherscan_api_key: str = ""
    okx_api_key: str = ""
    okx_secret_key: str = ""
    okx_passphrase: str = ""

    # App
    log_level: str = "INFO"
    poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC
    api_host: str = "0.0.0.0"
    api_port: int = 8001

    # Analysis parameters
    anomaly_contamination: float = DEFAULT_ANOMALY_CONTAMINATION
    cluster_min_samples: int = DEFAULT_CLUSTER_MIN_SAMPLES
    signal_confidence_threshold: float = DEFAULT_SIGNAL_CONFIDENCE_THRESHOLD

    # Tracked wallets (comma-separated addresses to monitor)
    tracked_wallets: list[str] = Field(default_factory=list)
