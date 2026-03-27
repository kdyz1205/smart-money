"""Tests for wallet performance scoring."""

from datetime import datetime, timezone

from smart_money.validator.wallet_performance import compute_wallet_performance


def test_compute_wallet_performance_basic() -> None:
    trades = [
        {"token": "ETH", "entry_price": 100, "exit_price": 120, "hold_hours": 5,
         "exit_time": datetime(2026, 3, 20, tzinfo=timezone.utc)},
        {"token": "BTC", "entry_price": 50000, "exit_price": 48000, "hold_hours": 10,
         "exit_time": datetime(2026, 3, 21, tzinfo=timezone.utc)},
        {"token": "SOL", "entry_price": 100, "exit_price": 150, "hold_hours": 2,
         "exit_time": datetime(2026, 3, 22, tzinfo=timezone.utc)},
    ]
    record = compute_wallet_performance("0xwallet", trades)
    assert record.total_trades_30d == 3
    assert record.winning_trades_30d == 2
    assert record.win_rate_30d > 0.6
    assert record.is_high_confidence  # 2/3 = 66%


def test_compute_wallet_performance_empty() -> None:
    record = compute_wallet_performance("0xwallet", [])
    assert record.total_trades_30d == 0
    assert record.win_rate_30d == 0.0
    assert not record.is_high_confidence
