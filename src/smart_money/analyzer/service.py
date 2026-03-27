"""Analyzer service — public facade for the wallet analysis pipeline.

Other modules import only this service, never the internal files.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..shared.constants import Chain
from ..shared.events import Event, EventBus, EventType
from ..shared.models import AnalysisParams, Transaction, WalletFeatures, WalletProfile
from .anomaly import compute_smart_money_score, detect_anomalies
from .clustering import cluster_wallets
from .features import extract_features

logger = logging.getLogger(__name__)


class AnalyzerServiceImpl:
    """Orchestrates feature extraction → clustering → anomaly detection."""

    def __init__(self, event_bus: EventBus, params: AnalysisParams | None = None) -> None:
        self._event_bus = event_bus
        self._params = params or AnalysisParams()
        self._profiles: dict[str, WalletProfile] = {}
        self._last_features: list[WalletFeatures] = []

    def update_params(self, params: AnalysisParams) -> None:
        self._params = params
        logger.info("Analyzer params updated: %s", params.model_dump_json())

    async def analyze_wallets(self, txs: list[Transaction]) -> list[WalletProfile]:
        """Full pipeline: features → clustering → anomaly → smart money scoring."""
        features = self.extract_features(txs)
        self._last_features = features
        if not features:
            return []

        # Step 1: Cluster wallets
        profiles = cluster_wallets(
            features,
            min_samples=self._params.cluster_min_samples,
        )

        # Step 2: Anomaly detection
        anomalies = detect_anomalies(
            features,
            contamination=self._params.anomaly_contamination,
        )

        # Step 3: Merge anomaly scores into profiles and compute smart-money score
        anomaly_map = {feat.address.lower(): score for feat, score in anomalies}
        feat_map = {f.address.lower(): f for f in features}

        for profile in profiles:
            key = profile.address.lower()
            a_score = anomaly_map.get(key, 0.0)
            feat = feat_map.get(key)
            if feat:
                profile.smart_money_score = compute_smart_money_score(
                    anomaly_score=a_score,
                    win_rate=feat.win_rate,
                    volume_usd=feat.total_volume_usd,
                )
                profile.is_smart_money = (
                    profile.smart_money_score >= self._params.signal_confidence_threshold
                )
                if profile.is_smart_money:
                    profile.labels.append("smart_money")

            # Cache profile
            self._profiles[profile.address] = profile

        smart_count = sum(1 for p in profiles if p.is_smart_money)
        logger.info(
            "Analysis complete: %d wallets, %d identified as smart money",
            len(profiles),
            smart_count,
        )

        # Publish smart money identifications
        smart_wallets = [p for p in profiles if p.is_smart_money]
        if smart_wallets:
            await self._event_bus.publish(
                Event(
                    event_type=EventType.SMART_MONEY_IDENTIFIED,
                    payload=smart_wallets,
                )
            )

        return profiles

    def extract_features(self, txs: list[Transaction]) -> list[WalletFeatures]:
        return extract_features(txs)

    def identify_smart_money(self, profiles: list[WalletProfile]) -> list[WalletProfile]:
        return [p for p in profiles if p.is_smart_money]

    def get_profile(self, address: str) -> WalletProfile | None:
        return self._profiles.get(address.lower())

    def get_all_profiles(self) -> list[WalletProfile]:
        return list(self._profiles.values())

    def get_last_features(self) -> list[WalletFeatures]:
        """Return the features from the most recent analyze_wallets call."""
        return self._last_features
