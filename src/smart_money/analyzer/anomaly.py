"""Anomaly detection — identify unusually active or prescient wallets.

Uses Isolation Forest to flag wallets whose behavior deviates significantly
from the norm, which often correlates with "smart money" activity.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..shared.models import WalletFeatures

logger = logging.getLogger(__name__)


def detect_anomalies(
    features: list[WalletFeatures],
    contamination: float = 0.05,
) -> list[tuple[WalletFeatures, float]]:
    """Score each wallet's anomaly level.

    Returns list of (features, anomaly_score) pairs.
    Score < 0 means anomalous (the more negative, the more anomalous).
    """
    if len(features) < 10:
        logger.warning("Too few wallets (%d) for anomaly detection", len(features))
        return [(f, 0.0) for f in features]

    X = np.array([f.to_vector() for f in features])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200,
    )
    model.fit(X_scaled)

    # decision_function: negative = anomalous, positive = normal
    scores = model.decision_function(X_scaled)

    anomalous_count = int(np.sum(scores < 0))
    logger.info(
        "Anomaly detection: %d/%d wallets flagged", anomalous_count, len(features)
    )

    return list(zip(features, scores.tolist()))


def compute_smart_money_score(
    anomaly_score: float,
    win_rate: float,
    volume_usd: float,
    recency_weight: float = 0.2,
) -> float:
    """Combine anomaly score with trading metrics into a 0-1 smart money score.

    Higher score = more likely to be smart money.
    """
    # Normalize anomaly score: more negative → higher smart-money likelihood
    anomaly_component = max(0.0, min(1.0, -anomaly_score))

    # Win rate component (0 to 1)
    win_component = max(0.0, min(1.0, win_rate))

    # Volume component (log-scaled, guard against negative values)
    vol_component = min(1.0, np.log1p(max(0.0, volume_usd)) / 40.0)

    score = (
        0.4 * anomaly_component
        + 0.35 * win_component
        + 0.25 * vol_component
    )
    return float(max(0.0, min(1.0, score)))
