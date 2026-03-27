"""Tests for volume surge detection."""

from datetime import datetime, timezone

from smart_money.shared.constants import Chain
from smart_money.shared.models import Transaction
from smart_money.validator.volume_filter import detect_volume_surges


def _make_txs(sm_addr: str, n: int = 10) -> list[Transaction]:
    return [
        Transaction(
            tx_hash=f"0x{'b' * 63}{i:01x}",
            chain=Chain.ETH,
            from_addr=sm_addr,
            to_addr="0xtoken_xyz",
            value_wei=10**18,
            token_symbol="XYZ",
            token_address="0xtoken_xyz",
            block_number=18000000 + i,
            timestamp=datetime(2024, 6, 1, 12, i, 0, tzinfo=timezone.utc),
            gas_used=21000,
        )
        for i in range(n)
    ]


def test_detect_volume_surge_with_high_sm_ratio() -> None:
    sm_addr = "0xsmart"
    txs = _make_txs(sm_addr, n=10)
    total_market = sum(float(tx.value_wei) for tx in txs) * 1.5  # SM is 66% of market

    surge = detect_volume_surges(
        txs=txs,
        smart_money_addresses={sm_addr},
        total_market_volume_usd=total_market,
        avg_volume_24h_usd=total_market * 0.1,  # way below current
        current_price=1.0,
        price_at_window_start=1.0,
        window_minutes=5,
        sm_ratio_threshold=0.35,
    )
    assert surge is not None
    assert surge.sm_volume_ratio > 0.35
    assert sm_addr in surge.contributing_wallets


def test_detect_volume_surge_stealth() -> None:
    sm_addr = "0xsmart"
    txs = _make_txs(sm_addr, n=5)
    total = sum(float(tx.value_wei) for tx in txs) * 1.2

    surge = detect_volume_surges(
        txs=txs,
        smart_money_addresses={sm_addr},
        total_market_volume_usd=total,
        avg_volume_24h_usd=total * 0.1,
        current_price=100.0,
        price_at_window_start=99.0,  # < 3% change
        window_minutes=5,
    )
    assert surge is not None
    assert surge.is_stealth_accumulation


def test_detect_volume_surge_no_smart_money() -> None:
    txs = _make_txs("0xrandom", n=5)
    surge = detect_volume_surges(
        txs=txs,
        smart_money_addresses={"0xother"},
        total_market_volume_usd=1e18,
        avg_volume_24h_usd=1e18,
        current_price=1.0,
        price_at_window_start=1.0,
    )
    assert surge is None
