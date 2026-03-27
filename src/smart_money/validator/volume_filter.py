"""Short-term Volume Spike Filter — detect smart-money volume surges.

Monitors 5-minute and 15-minute rolling windows for:
  1. Smart-money volume / total market volume > 35%
  2. Smart-money volume > 5x 24h average
  3. Net buy volume > 8% of 24h average

Combined with price behavior:
  - Volume spike + price flat (< 3%) → stealth accumulation
  - Volume spike + price breaking resistance → strong buy signal
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from ..shared.constants import Chain
from ..shared.models import Transaction, VolumeSurge

logger = logging.getLogger(__name__)


def detect_volume_surges(
    txs: list[Transaction],
    smart_money_addresses: set[str],
    total_market_volume_usd: float,
    avg_volume_24h_usd: float,
    current_price: float,
    price_at_window_start: float,
    window_minutes: int = 5,
    sm_ratio_threshold: float = 0.35,
    multiplier_threshold: float = 5.0,
    net_buy_pct_threshold: float = 0.08,
) -> VolumeSurge | None:
    """Analyze a batch of transactions for smart-money volume surge.

    Args:
        txs: Transactions in the time window.
        smart_money_addresses: Set of known smart-money wallet addresses.
        total_market_volume_usd: Total market volume in this window.
        avg_volume_24h_usd: 24h average volume for this token.
        current_price: Current token price.
        price_at_window_start: Price at window start for change calculation.
        window_minutes: Window size (5 or 15).
    """
    if not txs:
        return None

    # Separate smart-money vs non-smart-money
    sm_buy_volume = 0.0
    sm_sell_volume = 0.0
    sm_wallets: set[str] = set()

    for tx in txs:
        vol = float(tx.value_wei)
        from_addr = tx.from_addr.lower()
        to_addr = tx.to_addr.lower()

        if from_addr in smart_money_addresses:
            sm_buy_volume += vol
            sm_wallets.add(from_addr)
        if to_addr in smart_money_addresses:
            sm_sell_volume += vol
            sm_wallets.add(to_addr)

    sm_total = sm_buy_volume + sm_sell_volume
    net_buy = sm_buy_volume - sm_sell_volume

    if sm_total == 0:
        return None

    # Compute ratios
    sm_ratio = sm_total / max(total_market_volume_usd, 1.0)
    vs_24h = sm_total / max(avg_volume_24h_usd, 1.0)
    net_buy_pct = abs(net_buy) / max(avg_volume_24h_usd, 1.0)

    # Price change in window
    if price_at_window_start > 0:
        price_change_pct = (current_price - price_at_window_start) / price_at_window_start * 100
    else:
        price_change_pct = 0.0

    # Check if any threshold is exceeded
    is_surge = (
        sm_ratio > sm_ratio_threshold
        or vs_24h > multiplier_threshold
        or net_buy_pct > net_buy_pct_threshold
    )

    if not is_surge:
        return None

    # Stealth accumulation: volume spike but price hasn't moved much
    is_stealth = sm_ratio > sm_ratio_threshold and abs(price_change_pct) < 3.0

    token_addr = txs[0].token_address or txs[0].to_addr
    token_sym = txs[0].token_symbol or "UNKNOWN"

    return VolumeSurge(
        surge_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        token_address=token_addr,
        token_symbol=token_sym,
        chain=txs[0].chain,
        window_minutes=window_minutes,
        smart_money_volume_usd=sm_total,
        total_market_volume_usd=total_market_volume_usd,
        sm_volume_ratio=round(sm_ratio, 4),
        vs_24h_avg_multiplier=round(vs_24h, 2),
        net_buy_volume_usd=net_buy,
        price_change_pct=round(price_change_pct, 2),
        is_stealth_accumulation=is_stealth,
        contributing_wallets=sorted(sm_wallets),
    )
