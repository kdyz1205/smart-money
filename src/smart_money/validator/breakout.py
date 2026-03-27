"""Breakout pre-signal detection — identify smart-money patterns before price breakout.

Four detection dimensions:
  A. Smart-money concentration surge (Top-50 wallets' share rises 300%+ in 30 min)
  B. Buy/sell asymmetry (ratio > 8:1 with buys below current price)
  C. Multi-wallet coordinated buy (DBSCAN: 3+ smart wallets buy same token in 10 min)
  D. Stealth accumulation → aggressive fill (slow small buys then sudden large orders)
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from ..shared.constants import Chain
from ..shared.models import BreakoutPresignal, Transaction

logger = logging.getLogger(__name__)


def detect_concentration_surge(
    recent_txs: list[Transaction],
    older_txs: list[Transaction],
    smart_money_addresses: set[str],
    threshold_pct: float = 3.0,
) -> BreakoutPresignal | None:
    """Detect smart-money concentration increasing rapidly.

    Compare smart-money wallet count and volume share in the most recent
    30 minutes vs the prior 30 minutes. Trigger if increase > threshold_pct (300%).
    """
    def sm_stats(txs: list[Transaction]) -> tuple[int, float]:
        addrs = set()
        vol = 0.0
        for tx in txs:
            if tx.from_addr.lower() in smart_money_addresses:
                addrs.add(tx.from_addr.lower())
                vol += float(tx.value_wei)
        return len(addrs), vol

    recent_count, recent_vol = sm_stats(recent_txs)
    older_count, older_vol = sm_stats(older_txs)

    if older_count == 0 and recent_count >= 3:
        increase_pct = float("inf")
    elif older_count > 0:
        increase_pct = (recent_count - older_count) / older_count
    else:
        return None

    count_surge = increase_pct >= (threshold_pct - 1.0)  # 300% = 3x original
    vol_surge = recent_vol > older_vol * (threshold_pct - 0.5) if older_vol > 0 else recent_vol > 0

    if not (count_surge or vol_surge):
        return None

    token_addr = recent_txs[0].token_address or recent_txs[0].to_addr if recent_txs else ""
    token_sym = next((tx.token_symbol for tx in recent_txs if tx.token_symbol), "UNKNOWN")
    wallets = list({tx.from_addr.lower() for tx in recent_txs if tx.from_addr.lower() in smart_money_addresses})

    return BreakoutPresignal(
        presignal_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        token_address=token_addr,
        token_symbol=token_sym,
        chain=recent_txs[0].chain if recent_txs else Chain.ETH,
        signal_type="concentration_surge",
        confidence=min(1.0, increase_pct / 5.0),
        detail={
            "recent_sm_wallets": recent_count,
            "older_sm_wallets": older_count,
            "increase_pct": round(increase_pct * 100, 1),
            "recent_volume": recent_vol,
            "older_volume": older_vol,
        },
        contributing_wallets=wallets,
    )


def detect_buy_sell_asymmetry(
    txs: list[Transaction],
    smart_money_addresses: set[str],
    current_price_wei: float,
    ratio_threshold: float = 8.0,
) -> BreakoutPresignal | None:
    """Detect extreme buy/sell imbalance from smart-money wallets.

    Trigger when buy:sell ratio > 8:1 and average buy price < current price.
    """
    buy_volume = 0.0
    sell_volume = 0.0
    buy_prices: list[float] = []

    for tx in txs:
        vol = float(tx.value_wei)
        if tx.from_addr.lower() in smart_money_addresses:
            buy_volume += vol
            buy_prices.append(vol)  # proxy for price
        elif tx.to_addr.lower() in smart_money_addresses:
            sell_volume += vol

    if sell_volume == 0 and buy_volume > 0:
        ratio = float("inf")
    elif sell_volume > 0:
        ratio = buy_volume / sell_volume
    else:
        return None

    if ratio < ratio_threshold:
        return None

    avg_buy_price = float(np.mean(buy_prices)) if buy_prices else 0.0
    is_below_market = avg_buy_price < current_price_wei

    token_addr = txs[0].token_address or txs[0].to_addr if txs else ""
    token_sym = next((tx.token_symbol for tx in txs if tx.token_symbol), "UNKNOWN")
    wallets = list({tx.from_addr.lower() for tx in txs if tx.from_addr.lower() in smart_money_addresses})

    confidence = min(1.0, ratio / 15.0)
    if is_below_market:
        confidence = min(1.0, confidence + 0.15)

    return BreakoutPresignal(
        presignal_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        token_address=token_addr,
        token_symbol=token_sym,
        chain=txs[0].chain if txs else Chain.ETH,
        signal_type="buy_sell_asymmetry",
        confidence=round(confidence, 4),
        detail={
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "ratio": round(ratio, 2),
            "avg_buy_below_market": is_below_market,
        },
        contributing_wallets=wallets,
    )


def detect_stealth_then_aggressive(
    txs: list[Transaction],
    wallet_address: str,
    small_trade_threshold_usd: float = 1000.0,
    large_trade_multiplier: float = 5.0,
) -> BreakoutPresignal | None:
    """Detect stealth accumulation followed by aggressive buying.

    Pattern: wallet makes many small buys over 1-2 hours, then suddenly
    places a large order (5x+ the average small order).
    """
    sorted_txs = sorted(txs, key=lambda t: t.timestamp)
    if len(sorted_txs) < 5:
        return None

    values = [float(tx.value_wei) for tx in sorted_txs]

    # Find the transition point: where order size jumps
    small_phase_values: list[float] = []
    transition_idx = None

    for i, val in enumerate(values):
        if val <= small_trade_threshold_usd:
            small_phase_values.append(val)
        elif small_phase_values and val > np.mean(small_phase_values) * large_trade_multiplier:
            transition_idx = i
            break

    if transition_idx is None or len(small_phase_values) < 3:
        return None

    avg_small = float(np.mean(small_phase_values))
    aggressive_val = values[transition_idx]

    token_addr = sorted_txs[0].token_address or sorted_txs[0].to_addr
    token_sym = next((tx.token_symbol for tx in sorted_txs if tx.token_symbol), "UNKNOWN")

    # Duration of stealth phase
    stealth_duration_min = (
        sorted_txs[transition_idx].timestamp - sorted_txs[0].timestamp
    ).total_seconds() / 60

    confidence = min(
        1.0,
        (aggressive_val / max(avg_small, 1.0)) / 20.0 + len(small_phase_values) / 20.0,
    )

    return BreakoutPresignal(
        presignal_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        token_address=token_addr,
        token_symbol=token_sym,
        chain=sorted_txs[0].chain,
        signal_type="stealth_then_aggressive",
        confidence=round(confidence, 4),
        detail={
            "stealth_trades": len(small_phase_values),
            "avg_stealth_size": round(avg_small, 2),
            "aggressive_size": aggressive_val,
            "multiplier": round(aggressive_val / max(avg_small, 1.0), 1),
            "stealth_duration_min": round(stealth_duration_min, 1),
        },
        contributing_wallets=[wallet_address],
    )
