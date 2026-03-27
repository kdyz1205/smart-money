"""Tests for wallet clustering."""

from smart_money.analyzer.clustering import cluster_wallets
from smart_money.shared.models import WalletFeatures


def test_cluster_wallets_dbscan(sample_features: list[WalletFeatures]) -> None:
    profiles = cluster_wallets(sample_features, method="dbscan", min_samples=3)
    assert len(profiles) == len(sample_features)
    for p in profiles:
        assert p.cluster_id is not None


def test_cluster_wallets_kmeans(sample_features: list[WalletFeatures]) -> None:
    profiles = cluster_wallets(sample_features, method="kmeans", n_clusters=4)
    assert len(profiles) == len(sample_features)
    cluster_ids = {p.cluster_id for p in profiles}
    assert len(cluster_ids) <= 4


def test_cluster_wallets_too_few() -> None:
    features = [
        WalletFeatures(address="0x001"),
        WalletFeatures(address="0x002"),
    ]
    profiles = cluster_wallets(features, min_samples=5)
    assert len(profiles) == 2
    assert all(p.cluster_id == -1 for p in profiles)
