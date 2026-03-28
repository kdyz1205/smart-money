"""Feature engineering — transform raw transactions into wallet feature vectors.

Pure pandas transforms, no side-effects. Easy to unit test.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

from ..shared.models import Transaction, WalletFeatures


def extract_features(txs: list[Transaction]) -> list[WalletFeatures]:
    """Build a WalletFeatures vector for each unique address seen in *txs*."""
    if not txs:
        return []

    # Group transactions by participating addresses
    # Only include from_addr (senders). Including to_addr contaminates
    # analysis with contract/pool addresses that receive all trades.
    by_wallet: dict[str, list[Transaction]] = defaultdict(list)
    for tx in txs:
        from_lower = tx.from_addr.lower()
        # Skip obvious non-wallet addresses
        if from_lower in ("unknown", "") or from_lower.startswith(("pool_", "anon_")):
            continue
        by_wallet[from_lower].append(tx)

    now = datetime.now(timezone.utc)
    results: list[WalletFeatures] = []

    for addr, wallet_txs in by_wallet.items():
        if len(wallet_txs) < 2:
            continue

        timestamps = sorted(tx.timestamp for tx in wallet_txs)
        values_wei = [tx.value_wei for tx in wallet_txs]
        gas_values = [tx.gas_used for tx in wallet_txs]

        # Time-based features
        hours_24 = sum(
            1 for ts in timestamps if (now - ts).total_seconds() < 86400
        )
        hours_168 = sum(
            1 for ts in timestamps if (now - ts).total_seconds() < 604800
        )

        # Value features (approximate USD using wei — real impl would use price feeds)
        avg_val = float(np.mean(values_wei)) if values_wei else 0.0
        total_vol = float(np.sum(values_wei)) if values_wei else 0.0
        max_val = float(np.max(values_wei)) if values_wei else 0.0

        # Token diversity
        unique_tokens = len(
            {tx.token_symbol for tx in wallet_txs if tx.token_symbol}
        )

        # Gas ratio (how much of total value goes to gas)
        total_gas = sum(gas_values)
        gas_ratio = total_gas / max(total_vol, 1.0)

        # Hold duration approximation (time between first and last tx)
        if len(timestamps) >= 2:
            span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
        else:
            span_hours = 0.0

        # Inflow / outflow
        outflow = sum(tx.value_wei for tx in wallet_txs if tx.from_addr.lower() == addr)
        inflow = sum(tx.value_wei for tx in wallet_txs if tx.to_addr.lower() == addr)
        io_ratio = inflow / max(outflow, 1.0)

        # DEX vs CEX heuristic: if method_id present → likely contract/DEX interaction
        dex_count = sum(1 for tx in wallet_txs if tx.method_id and tx.method_id != "0x")
        cex_count = len(wallet_txs) - dex_count
        dex_cex_ratio = dex_count / max(cex_count, 1)

        results.append(
            WalletFeatures(
                address=addr,
                tx_frequency_24h=float(hours_24),
                tx_frequency_7d=float(hours_168),
                avg_tx_value_usd=avg_val,
                total_volume_usd=total_vol,
                unique_tokens_traded=unique_tokens,
                dex_to_cex_ratio=dex_cex_ratio,
                gas_spend_ratio=gas_ratio,
                avg_hold_duration_hours=span_hours,
                win_rate=0.0,  # requires PnL calculation — filled later
                max_single_trade_usd=max_val,
                inflow_outflow_ratio=io_ratio,
            )
        )

    return results
