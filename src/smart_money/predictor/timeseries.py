"""Time series analysis for predicting wallet buy-in behavior.

Uses rolling statistics and momentum indicators to detect
accumulation patterns before they become visible on-chain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AccumulationSignal:
    """Intermediate signal from time-series analysis."""

    token_address: str
    token_symbol: str
    buy_probability: float  # 0 to 1
    predicted_volume_usd: float
    momentum_score: float  # positive = accumulating
    wallet_addresses: list[str]


def detect_accumulation(
    token_address: str,
    token_symbol: str,
    buy_volumes: list[float],
    sell_volumes: list[float],
    timestamps: list[float],
    wallet_addresses: list[str],
    window: int = 12,
) -> AccumulationSignal | None:
    """Detect accumulation pattern using volume momentum.

    Compares recent buy volume trend against sell volume trend.
    If buy momentum is increasing while sell momentum is flat/decreasing,
    this indicates smart accumulation.
    """
    if len(buy_volumes) < window:
        return None

    buys = np.array(buy_volumes[-window:])
    sells = np.array(sell_volumes[-window:])

    # Exponential moving average of buy/sell volumes
    weights = np.exp(np.linspace(0, 1, window))
    weights /= weights.sum()

    buy_ema = float(np.dot(buys, weights))
    sell_ema = float(np.dot(sells, weights))

    # Buy momentum: ratio of recent EMA to older average
    if len(buy_volumes) >= window * 2:
        older_buys = np.array(buy_volumes[-window * 2 : -window])
        older_avg = float(np.mean(older_buys)) if len(older_buys) > 0 else 1.0
        momentum = (buy_ema - older_avg) / max(older_avg, 1.0)
    else:
        momentum = 0.0

    # Buy probability based on volume imbalance and momentum
    if buy_ema + sell_ema > 0:
        imbalance = buy_ema / (buy_ema + sell_ema)
    else:
        imbalance = 0.5

    buy_probability = float(np.clip(imbalance * 0.6 + max(momentum, 0) * 0.4, 0, 1))

    if buy_probability < 0.3:
        return None

    return AccumulationSignal(
        token_address=token_address,
        token_symbol=token_symbol,
        buy_probability=buy_probability,
        predicted_volume_usd=buy_ema,
        momentum_score=momentum,
        wallet_addresses=wallet_addresses,
    )


def detect_coordinated_buying(
    wallet_buy_times: dict[str, list[float]],
    time_window_sec: float = 300.0,
    min_wallets: int = 3,
) -> list[str] | None:
    """Detect if multiple wallets are buying the same token within a short window.

    Returns the list of coordinated wallet addresses, or None.
    """
    if len(wallet_buy_times) < min_wallets:
        return None

    # Find overlapping buy windows
    all_times: list[tuple[float, str]] = []
    for addr, times in wallet_buy_times.items():
        for t in times:
            all_times.append((t, addr))
    all_times.sort()

    # Sliding window
    coordinated: set[str] = set()
    for i, (t_i, addr_i) in enumerate(all_times):
        window_addrs = {addr_i}
        for j in range(i + 1, len(all_times)):
            t_j, addr_j = all_times[j]
            if t_j - t_i > time_window_sec:
                break
            window_addrs.add(addr_j)
        if len(window_addrs) >= min_wallets:
            coordinated.update(window_addrs)

    return list(coordinated) if len(coordinated) >= min_wallets else None
