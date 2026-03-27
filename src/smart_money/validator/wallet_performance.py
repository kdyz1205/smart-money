"""Wallet performance scoring — track historical win rates.

Maintains a rolling 30-day performance record for each smart-money wallet.
Only wallets with win_rate >= 65% enter the "High-Confidence" list,
which gets higher prediction weight.

Also includes:
  - CEX cross-validation (large buy on DEX + CEX inflow = strong signal)
  - Slippage sensitivity (low slippage + high speed = better router / OTC)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from ..shared.constants import Chain
from ..shared.models import (
    SlippageMetrics,
    Transaction,
    WalletPerformanceRecord,
    WalletProfile,
)

logger = logging.getLogger(__name__)

# Known CEX deposit addresses (simplified — in production, use a label DB)
KNOWN_CEX_ADDRESSES: set[str] = {
    # Binance hot wallets (examples)
    "0x28c6c06298d514db089934071355e5743bf21d60",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549",
    # OKX hot wallets
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b",
    "0x98ec059dc3adfbdd63429227115656b07c44a305",
}


def compute_wallet_performance(
    address: str,
    trades: list[dict],
    lookback_days: int = 30,
    chain: Chain = Chain.ETH,
) -> WalletPerformanceRecord:
    """Compute rolling performance for a wallet.

    Args:
        address: Wallet address.
        trades: List of trade dicts with keys:
            token, entry_time, exit_time, entry_price, exit_price, hold_hours
    """
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=lookback_days)

    recent_trades = [
        t for t in trades if t.get("exit_time", now) >= cutoff
    ]

    if not recent_trades:
        return WalletPerformanceRecord(address=address, chain=chain)

    pnl_pcts: list[float] = []
    hold_hours: list[float] = []

    for t in recent_trades:
        entry = t.get("entry_price", 0)
        exit_ = t.get("exit_price", 0)
        if entry > 0:
            pnl_pct = (exit_ - entry) / entry * 100
        else:
            pnl_pct = 0.0
        pnl_pcts.append(pnl_pct)
        hold_hours.append(t.get("hold_hours", 0))

    wins = sum(1 for p in pnl_pcts if p > 0)
    total = len(pnl_pcts)

    record = WalletPerformanceRecord(
        address=address,
        chain=chain,
        total_trades_30d=total,
        winning_trades_30d=wins,
        win_rate_30d=round(wins / max(total, 1), 4),
        avg_pnl_pct=round(float(np.mean(pnl_pcts)), 2),
        best_trade_pnl_pct=round(float(max(pnl_pcts)), 2) if pnl_pcts else 0.0,
        worst_trade_pnl_pct=round(float(min(pnl_pcts)), 2) if pnl_pcts else 0.0,
        avg_hold_hours=round(float(np.mean(hold_hours)), 1) if hold_hours else 0.0,
        is_high_confidence=wins / max(total, 1) >= 0.65,
        last_updated=now,
    )

    return record


def detect_cex_cross_flow(
    txs: list[Transaction],
    smart_money_addresses: set[str],
    time_window_sec: float = 600.0,
) -> list[dict]:
    """Detect CEX-to-DEX flow patterns that indicate imminent buying.

    Pattern: CEX withdrawal to new wallet → fast DEX buy = high probability pump.
    Or: Smart money buys on DEX + simultaneous large CEX inflow = confirmation.
    """
    alerts: list[dict] = []

    cex_withdrawals: list[Transaction] = []
    dex_buys: list[Transaction] = []

    for tx in txs:
        from_lower = tx.from_addr.lower()
        to_lower = tx.to_addr.lower()

        if from_lower in KNOWN_CEX_ADDRESSES:
            cex_withdrawals.append(tx)
        if from_lower in smart_money_addresses and tx.method_id and tx.method_id != "0x":
            dex_buys.append(tx)

    # Match CEX withdrawal → DEX buy within time window
    for withdrawal in cex_withdrawals:
        for buy in dex_buys:
            time_diff = abs(
                (buy.timestamp - withdrawal.timestamp).total_seconds()
            )
            if (
                time_diff <= time_window_sec
                and withdrawal.to_addr.lower() == buy.from_addr.lower()
            ):
                alerts.append(
                    {
                        "type": "cex_to_dex_flow",
                        "cex_tx": withdrawal.tx_hash,
                        "dex_tx": buy.tx_hash,
                        "wallet": buy.from_addr,
                        "token": buy.token_symbol,
                        "delay_sec": time_diff,
                        "volume_wei": buy.value_wei,
                    }
                )

    if alerts:
        logger.info("Detected %d CEX-to-DEX flow events", len(alerts))
    return alerts


def compute_slippage(
    txs: list[Transaction],
    wallet_address: str,
    expected_prices: dict[str, float],
) -> SlippageMetrics | None:
    """Compute slippage metrics for a wallet's trades on a token.

    Low slippage + high fill speed → wallet likely uses better routing
    (private mempool, OTC, or aggregator).
    """
    wallet_txs = [
        tx for tx in txs if tx.from_addr.lower() == wallet_address.lower()
    ]
    if not wallet_txs:
        return None

    token_addr = wallet_txs[0].token_address or wallet_txs[0].to_addr
    slippages: list[float] = []

    for tx in wallet_txs:
        expected = expected_prices.get(tx.tx_hash, float(tx.value_wei))
        actual = float(tx.value_wei)
        if expected > 0:
            slip = abs(actual - expected) / expected * 100
            slippages.append(slip)

    if not slippages:
        return None

    avg_slip = float(np.mean(slippages))
    max_slip = float(np.max(slippages))
    min_slip = float(np.min(slippages))

    return SlippageMetrics(
        wallet_address=wallet_address,
        token_address=token_addr,
        avg_slippage_pct=round(avg_slip, 4),
        max_slippage_pct=round(max_slip, 4),
        min_slippage_pct=round(min_slip, 4),
        num_trades=len(slippages),
        has_better_routing=avg_slip < 0.5 and len(slippages) >= 3,
    )
