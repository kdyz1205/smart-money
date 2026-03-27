"""Tests for fill-speed analysis."""

from datetime import datetime, timezone

from smart_money.shared.constants import Chain
from smart_money.shared.models import Transaction
from smart_money.validator.fill_speed import analyze_fill_speed, detect_fill_speed_alerts


def _make_tx(i: int, interval_sec: int = 30) -> Transaction:
    return Transaction(
        tx_hash=f"0x{'f' * 63}{i:01x}",
        chain=Chain.ETH,
        from_addr="0xwallet1",
        to_addr="0xtoken_abc",
        value_wei=(i + 1) * 10**16,
        token_symbol="ABC",
        token_address="0xtoken_abc",
        block_number=18000000 + i,
        timestamp=datetime(2024, 6, 1, 12, 0, i * interval_sec, tzinfo=timezone.utc),
        gas_used=21000,
    )


def test_analyze_fill_speed_basic() -> None:
    txs = [_make_tx(i, interval_sec=10) for i in range(5)]
    metrics = analyze_fill_speed("0xwallet1", txs, market_volume_usd=1e18)
    assert metrics is not None
    assert metrics.num_trades == 5
    assert metrics.fill_speed_usd_per_sec > 0
    assert metrics.avg_trade_interval_sec > 0


def test_analyze_fill_speed_too_few_txs() -> None:
    txs = [_make_tx(0)]
    assert analyze_fill_speed("0xwallet1", txs) is None


def test_detect_fill_speed_alerts_rapid() -> None:
    # 5 trades, 10s apart — very rapid
    txs = [_make_tx(i, interval_sec=10) for i in range(5)]
    alerts = detect_fill_speed_alerts(
        wallet_txs_by_token={"0xtoken_abc": txs},
        wallet_address="0xwallet1",
        historical_speeds=[],
        market_volumes={},
        rapid_interval_sec=15.0,
        liquidity_pct_threshold=0.0,  # low threshold to trigger
        token_liquidity={"0xtoken_abc": 1.0},  # small liquidity so liq_pct is high
    )
    assert len(alerts) >= 1
    assert alerts[0].wallet_address == "0xwallet1"
