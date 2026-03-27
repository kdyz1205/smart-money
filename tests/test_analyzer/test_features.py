"""Tests for wallet feature extraction."""

from smart_money.analyzer.features import extract_features
from smart_money.shared.models import Transaction


def test_extract_features_from_transactions(
    sample_transactions: list[Transaction],
) -> None:
    features = extract_features(sample_transactions)
    assert len(features) > 0

    for feat in features:
        assert feat.address
        assert feat.tx_frequency_7d >= 0
        vec = feat.to_vector()
        assert len(vec) == 11
        assert all(isinstance(v, float) for v in vec)


def test_extract_features_empty() -> None:
    features = extract_features([])
    assert features == []


def test_feature_vector_consistency(sample_transactions: list[Transaction]) -> None:
    features = extract_features(sample_transactions)
    for feat in features:
        vec = feat.to_vector()
        assert vec[0] == feat.tx_frequency_24h
        assert vec[4] == float(feat.unique_tokens_traded)
