"""Tests for the backtest engine."""

import uuid
from datetime import datetime, timedelta, timezone

from smart_money.shared.constants import Chain, SignalType
from smart_money.shared.models import Signal
from smart_money.validator.backtest import PriceEvent, find_breakouts, run_backtest


def _make_price_events(
    token: str, base_price: float, n: int, interval_min: int = 5, increment: float = 0.5,
) -> list[PriceEvent]:
    start = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return [
        PriceEvent(
            token_address=token,
            token_symbol="TEST",
            timestamp=start + timedelta(minutes=i * interval_min),
            price_usd=base_price + i * increment,
        )
        for i in range(n)
    ]


def test_find_breakouts_basic() -> None:
    # Price goes from 100 to 118 in 6 intervals of 5 min = 30 min → 18% rise
    events = _make_price_events("0xtoken", base_price=100.0, n=7, increment=3.0)
    breakouts = find_breakouts(events, rise_threshold_pct=15.0, window_minutes=30)
    assert len(breakouts) >= 1
    assert breakouts[0].rise_pct >= 15.0


def test_find_breakouts_no_breakout() -> None:
    # Price goes from 100 to 103 → only 3%
    events = _make_price_events("0xtoken", base_price=100.0, n=7)
    # Override: flat prices
    for e in events:
        e.price_usd = 100.0 + 0.3 * events.index(e)
    breakouts = find_breakouts(events, rise_threshold_pct=15.0)
    assert len(breakouts) == 0


def test_run_backtest_precision_recall() -> None:
    start = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)

    # Create a breakout: price 100→120 in 30 min
    price_history = [
        PriceEvent("0xtoken", "TEST", start + timedelta(minutes=i * 5), 100.0 + i * 4)
        for i in range(7)
    ]

    # Signal 10 minutes before breakout
    signal = Signal(
        signal_id=str(uuid.uuid4()),
        timestamp=start - timedelta(minutes=10),
        token_address="0xtoken",
        token_symbol="TEST",
        chain=Chain.ETH,
        signal_type=SignalType.ACCUMULATION,
        confidence=0.8,
    )

    result = run_backtest(
        signals=[signal],
        price_history=price_history,
        lookback_days=90,
        max_lead_time_minutes=30.0,
    )
    assert result.total_breakouts_found >= 1
    assert result.signals_before_breakout >= 1
    assert result.precision > 0
    assert result.recall > 0
