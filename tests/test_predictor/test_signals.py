"""Tests for signal generation."""

from smart_money.predictor.signals import (
    create_coordinated_buy_signal,
    create_signal_from_accumulation,
)
from smart_money.predictor.timeseries import AccumulationSignal
from smart_money.shared.constants import SignalType
from smart_money.shared.models import WalletProfile


def test_create_signal_from_accumulation(sample_profiles: list[WalletProfile]) -> None:
    profile_map = {p.address: p for p in sample_profiles}
    acc = AccumulationSignal(
        token_address="0xtoken",
        token_symbol="TEST",
        buy_probability=0.8,
        predicted_volume_usd=100000.0,
        momentum_score=0.5,
        wallet_addresses=list(profile_map.keys())[:3],
    )
    signal = create_signal_from_accumulation(acc, profile_map)
    assert signal.signal_type == SignalType.ACCUMULATION
    assert 0 <= signal.confidence <= 1
    assert 0 <= signal.risk_score <= 1
    assert signal.token_symbol == "TEST"


def test_create_coordinated_buy_signal(sample_profiles: list[WalletProfile]) -> None:
    profile_map = {p.address: p for p in sample_profiles}
    addrs = list(profile_map.keys())[:5]
    signal = create_coordinated_buy_signal(
        token_address="0xtoken",
        token_symbol="ETH",
        wallet_addresses=addrs,
        profiles=profile_map,
    )
    assert signal.signal_type == SignalType.COORDINATED_BUY
    assert len(signal.contributing_wallets) == 5
