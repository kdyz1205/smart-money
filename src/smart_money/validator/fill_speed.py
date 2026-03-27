"""Order Fill Speed analysis — detect high-speed stealth accumulation.

MEV bots and top smart-money wallets often enter "high-speed fill" mode
5-30 minutes before a major breakout. This module detects that pattern.

Key metrics:
  - Fill Speed = cumulative buy volume / time (USD/sec)
  - Stealth Score = Fill Speed / market volume share (higher = more hidden)

Trigger conditions (configurable):
  - Fill Speed > wallet's historical 95th percentile AND Stealth Score > 3.0
  - OR 3 consecutive trades with interval < 45s, total > 1.5% of liquidity
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from ..shared.models import (
    FillSpeedAlert,
    FillSpeedMetrics,
    Transaction,
)
from ..shared.constants import Chain

logger = logging.getLogger(__name__)


def analyze_fill_speed(
    wallet_address: str,
    txs: list[Transaction],
    window_sec: float = 900.0,  # 15-minute window
    market_volume_usd: float = 1.0,
) -> FillSpeedMetrics | None:
    """Compute fill-speed metrics for a wallet's recent trades on a token.

    Args:
        wallet_address: The wallet to analyze.
        txs: Transactions from this wallet, pre-filtered to one token.
        window_sec: Time window to analyze (default 15 min).
        market_volume_usd: Total market volume in this window for stealth calc.
    """
    if len(txs) < 2:
        return None

    # Sort by timestamp
    sorted_txs = sorted(txs, key=lambda t: t.timestamp)
    window_start = sorted_txs[0].timestamp
    window_end = sorted_txs[-1].timestamp
    elapsed = (window_end - window_start).total_seconds()
    if elapsed <= 0:
        return None

    total_volume = sum(float(tx.value_wei) for tx in sorted_txs)
    fill_speed = total_volume / elapsed

    # Stealth score: how much volume relative to market
    wallet_share = total_volume / max(market_volume_usd, 1.0)
    stealth_score = fill_speed / max(wallet_share, 0.001)

    # Average interval between trades
    intervals = []
    for i in range(1, len(sorted_txs)):
        dt = (sorted_txs[i].timestamp - sorted_txs[i - 1].timestamp).total_seconds()
        intervals.append(dt)
    avg_interval = float(np.mean(intervals)) if intervals else 0.0

    token_sym = sorted_txs[0].token_symbol or "UNKNOWN"
    token_addr = sorted_txs[0].token_address or sorted_txs[0].to_addr

    return FillSpeedMetrics(
        wallet_address=wallet_address,
        token_address=token_addr,
        token_symbol=token_sym,
        window_start=window_start,
        window_end=window_end,
        num_trades=len(sorted_txs),
        total_volume_usd=total_volume,
        fill_speed_usd_per_sec=fill_speed,
        stealth_score=stealth_score,
        avg_trade_interval_sec=avg_interval,
    )


def detect_fill_speed_alerts(
    wallet_txs_by_token: dict[str, list[Transaction]],
    wallet_address: str,
    historical_speeds: list[float],
    market_volumes: dict[str, float],
    percentile_threshold: float = 95.0,
    stealth_threshold: float = 3.0,
    rapid_interval_sec: float = 45.0,
    liquidity_pct_threshold: float = 1.5,
    token_liquidity: dict[str, float] | None = None,
) -> list[FillSpeedAlert]:
    """Detect high-speed accumulation alerts across all tokens for a wallet.

    Two trigger conditions (OR):
    1. Fill Speed > historical 95th percentile AND Stealth Score > 3.0
    2. 3+ consecutive trades with interval < 45s AND total > 1.5% liquidity
    """
    alerts: list[FillSpeedAlert] = []
    token_liq = token_liquidity or {}

    # Historical percentile threshold
    if historical_speeds:
        speed_threshold = float(np.percentile(historical_speeds, percentile_threshold))
    else:
        speed_threshold = float("inf")

    for token_addr, txs in wallet_txs_by_token.items():
        market_vol = market_volumes.get(token_addr, 1.0)
        metrics = analyze_fill_speed(
            wallet_address, txs, market_volume_usd=market_vol
        )
        if not metrics:
            continue

        is_speed_alert = (
            metrics.fill_speed_usd_per_sec > speed_threshold
            and metrics.stealth_score > stealth_threshold
        )

        # Check rapid consecutive trades
        sorted_txs = sorted(txs, key=lambda t: t.timestamp)
        max_rapid_count = 0
        max_rapid_volume = 0.0
        current_count = 1
        current_volume = float(sorted_txs[0].value_wei) if sorted_txs else 0.0
        for i in range(1, len(sorted_txs)):
            interval = (
                sorted_txs[i].timestamp - sorted_txs[i - 1].timestamp
            ).total_seconds()
            if interval < rapid_interval_sec:
                current_count += 1
                current_volume += float(sorted_txs[i].value_wei)
            else:
                current_count = 1
                current_volume = float(sorted_txs[i].value_wei)
            if current_count > max_rapid_count:
                max_rapid_count = current_count
                max_rapid_volume = current_volume

        rapid_count = max_rapid_count
        rapid_volume = max_rapid_volume
        liquidity = token_liq.get(token_addr, float("inf"))
        liq_pct = (rapid_volume / max(liquidity, 1.0)) * 100
        is_rapid_alert = rapid_count >= 3 and liq_pct > liquidity_pct_threshold

        if is_speed_alert or is_rapid_alert:
            # Compute percentile of current speed
            if historical_speeds:
                pctile = float(
                    np.searchsorted(
                        np.sort(historical_speeds), metrics.fill_speed_usd_per_sec
                    )
                    / max(len(historical_speeds), 1)
                    * 100
                )
            else:
                pctile = 99.0

            alerts.append(
                FillSpeedAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc),
                    wallet_address=wallet_address,
                    token_address=token_addr,
                    token_symbol=metrics.token_symbol,
                    chain=Chain.ETH,
                    fill_speed_usd_per_sec=metrics.fill_speed_usd_per_sec,
                    stealth_score=metrics.stealth_score,
                    historical_percentile=pctile,
                    total_volume_usd=metrics.total_volume_usd,
                    num_rapid_trades=rapid_count,
                    avg_interval_sec=metrics.avg_trade_interval_sec,
                    liquidity_pct=liq_pct,
                    metadata={
                        "trigger": "speed+stealth" if is_speed_alert else "rapid_consecutive",
                    },
                )
            )

    return alerts
