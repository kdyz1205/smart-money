"""Shared test fixtures."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from smart_money.shared.constants import Chain
from smart_money.shared.events import EventBus
from smart_money.shared.models import Transaction, WalletFeatures, WalletProfile


@pytest.fixture
def event_bus() -> EventBus:
    return EventBus()


@pytest.fixture
def sample_transactions() -> list[Transaction]:
    """Generate a set of realistic test transactions."""
    base_time = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    txs = []
    for i in range(30):
        txs.append(
            Transaction(
                tx_hash=f"0x{'a' * 63}{i:01x}",
                chain=Chain.ETH,
                from_addr=f"0x{'1' * 38}{i % 5:02d}",
                to_addr=f"0x{'2' * 38}{i % 3:02d}",
                value_wei=(i + 1) * 10**17,
                token_symbol="USDC" if i % 2 == 0 else "WETH",
                token_address="0xtoken_usdc" if i % 2 == 0 else "0xtoken_weth",
                block_number=18000000 + i,
                timestamp=datetime(
                    2024, 6, 1, 12, i, 0, tzinfo=timezone.utc
                ),
                gas_used=21000 + i * 100,
                method_id="0xa9059cbb" if i % 3 == 0 else None,
            )
        )
    return txs


@pytest.fixture
def sample_features() -> list[WalletFeatures]:
    """Generate test wallet features for clustering/anomaly tests."""
    features = []
    for i in range(20):
        features.append(
            WalletFeatures(
                address=f"0x{'1' * 38}{i:02d}",
                tx_frequency_24h=float(i * 2),
                tx_frequency_7d=float(i * 10),
                avg_tx_value_usd=float(i * 1000),
                total_volume_usd=float(i * 50000),
                unique_tokens_traded=i + 1,
                dex_to_cex_ratio=0.5 + i * 0.05,
                gas_spend_ratio=0.01 + i * 0.001,
                avg_hold_duration_hours=float(i * 5),
                win_rate=min(1.0, 0.3 + i * 0.04),
                max_single_trade_usd=float(i * 5000),
                inflow_outflow_ratio=0.8 + i * 0.02,
            )
        )
    return features


@pytest.fixture
def sample_profiles() -> list[WalletProfile]:
    """Generate test wallet profiles."""
    return [
        WalletProfile(
            address=f"0x{'1' * 38}{i:02d}",
            chain=Chain.ETH,
            cluster_id=i % 3,
            is_smart_money=i > 15,
            smart_money_score=min(1.0, 0.3 + i * 0.04),
            total_tx_count=i * 10,
        )
        for i in range(20)
    ]
