"""Wallet clustering — group wallets by behavioral similarity.

Uses KMeans for initial grouping and DBSCAN for density-based discovery
of coordinated wallet groups.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

from ..shared.models import WalletFeatures, WalletProfile
from ..shared.constants import Chain

logger = logging.getLogger(__name__)


def cluster_wallets(
    features: list[WalletFeatures],
    method: str = "dbscan",
    min_samples: int = 5,
    n_clusters: int = 8,
    chain: Chain = Chain.ETH,
) -> list[WalletProfile]:
    """Assign cluster IDs to wallets based on their feature vectors.

    Returns a WalletProfile for each wallet with cluster_id set.
    """
    if len(features) < min_samples:
        logger.warning(
            "Not enough wallets (%d) for clustering (min=%d)",
            len(features),
            min_samples,
        )
        return [
            WalletProfile(address=f.address, chain=chain, cluster_id=-1)
            for f in features
        ]

    X = np.array([f.to_vector() for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=min_samples)
    else:
        effective_k = min(n_clusters, max(2, len(features) // 3))
        model = KMeans(n_clusters=effective_k, n_init="auto")

    labels = model.fit_predict(X_scaled)

    profiles: list[WalletProfile] = []
    for feat, label in zip(features, labels):
        profiles.append(
            WalletProfile(
                address=feat.address,
                chain=chain,
                cluster_id=int(label),
                total_tx_count=int(feat.tx_frequency_7d),
                avg_hold_duration_hours=feat.avg_hold_duration_hours,
                win_rate=feat.win_rate,
            )
        )

    n_clusters_found = len(set(labels) - {-1})
    n_noise = int(np.sum(labels == -1))
    logger.info(
        "Clustering complete: %d clusters, %d noise points", n_clusters_found, n_noise
    )
    return profiles
