"""Tests for breakout pre-signal detection."""

from datetime import datetime, timezone

from smart_money.shared.constants import Chain
from smart_money.shared.models import Transaction
from smart_money.validator.breakout import (
    detect_buy_sell_asymmetry,
    detect_concentration_surge,
    detect_stealth_then_aggressive,
)


def _make_tx(from_addr: str, value: int, minute: int) -> Transaction:
    return Transaction(
        tx_hash=f"0x{from_addr[:4]}{minute:04x}",
        chain=Chain.ETH,
        from_addr=from_addr,
        to_addr="0xtoken",
        value_wei=value,
        token_symbol="TEST",
        token_address="0xtoken",
        block_number=18000000 + minute,
        timestamp=datetime(2024, 6, 1, 12, minute, 0, tzinfo=timezone.utc),
        gas_used=21000,
    )


def test_concentration_surge_detected() -> None:
    sm_addrs = {"0xsm01", "0xsm02", "0xsm03", "0xsm04"}
    older = [_make_tx("0xrandom", 1000, i) for i in range(5)]
    recent = [_make_tx(addr, 5000, 10 + i) for i, addr in enumerate(sm_addrs)]

    result = detect_concentration_surge(recent, older, sm_addrs, threshold_pct=2.0)
    assert result is not None
    assert result.signal_type == "concentration_surge"
    assert len(result.contributing_wallets) >= 3


def test_buy_sell_asymmetry() -> None:
    sm_addrs = {"0xsm01", "0xsm02"}
    # Many buys from smart money, no sells
    txs = [_make_tx("0xsm01", 10000, i) for i in range(10)]
    result = detect_buy_sell_asymmetry(txs, sm_addrs, current_price_wei=5000.0, ratio_threshold=5.0)
    assert result is not None
    assert result.signal_type == "buy_sell_asymmetry"


def test_stealth_then_aggressive() -> None:
    # 6 small trades then 1 big trade
    txs = [_make_tx("0xwallet", 100, i) for i in range(6)]
    txs.append(_make_tx("0xwallet", 5000, 7))  # 50x the average

    result = detect_stealth_then_aggressive(txs, "0xwallet", small_trade_threshold_usd=200)
    assert result is not None
    assert result.signal_type == "stealth_then_aggressive"
    assert result.detail["stealth_trades"] >= 3
