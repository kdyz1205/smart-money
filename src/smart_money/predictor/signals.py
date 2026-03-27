"""Signal generation and risk scoring.

Transforms raw accumulation detections and wallet profiles into
actionable Signal objects with risk levels.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from ..shared.constants import RiskLevel, SignalType
from ..shared.models import Signal, WalletProfile
from .timeseries import AccumulationSignal

logger = logging.getLogger(__name__)


def create_signal_from_accumulation(
    acc: AccumulationSignal,
    profiles: dict[str, WalletProfile],
    chain_str: str = "ethereum",
) -> Signal:
    """Convert an AccumulationSignal into a full Signal with risk scoring."""
    from ..shared.constants import Chain

    chain = Chain(chain_str)

    # Compute aggregate smart-money score from contributing wallets
    sm_scores = [
        profiles[addr].smart_money_score
        for addr in acc.wallet_addresses
        if addr in profiles
    ]
    avg_sm_score = sum(sm_scores) / len(sm_scores) if sm_scores else 0.0

    # Confidence = accumulation probability weighted by smart-money score
    confidence = acc.buy_probability * 0.6 + avg_sm_score * 0.4

    # Risk scoring
    risk_score = _compute_risk_score(
        confidence=confidence,
        num_wallets=len(acc.wallet_addresses),
        predicted_volume=acc.predicted_volume_usd,
    )

    return Signal(
        signal_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        token_address=acc.token_address,
        token_symbol=acc.token_symbol,
        chain=chain,
        signal_type=SignalType.ACCUMULATION,
        confidence=round(confidence, 4),
        risk_level=_risk_level_from_score(risk_score),
        risk_score=round(risk_score, 4),
        contributing_wallets=acc.wallet_addresses,
        predicted_buy_volume_usd=acc.predicted_volume_usd,
        metadata={
            "momentum": acc.momentum_score,
            "avg_smart_money_score": round(avg_sm_score, 4),
        },
    )


def create_coordinated_buy_signal(
    token_address: str,
    token_symbol: str,
    wallet_addresses: list[str],
    profiles: dict[str, WalletProfile],
    chain_str: str = "ethereum",
) -> Signal:
    """Create a signal for coordinated buying activity."""
    from ..shared.constants import Chain

    chain = Chain(chain_str)

    sm_scores = [
        profiles[addr].smart_money_score
        for addr in wallet_addresses
        if addr in profiles
    ]
    avg_sm_score = sum(sm_scores) / len(sm_scores) if sm_scores else 0.0

    # More wallets buying together → higher confidence
    wallet_factor = min(1.0, len(wallet_addresses) / 10.0)
    confidence = avg_sm_score * 0.5 + wallet_factor * 0.5

    risk_score = _compute_risk_score(
        confidence=confidence,
        num_wallets=len(wallet_addresses),
        predicted_volume=0.0,
    )

    return Signal(
        signal_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        token_address=token_address,
        token_symbol=token_symbol,
        chain=chain,
        signal_type=SignalType.COORDINATED_BUY,
        confidence=round(confidence, 4),
        risk_level=_risk_level_from_score(risk_score),
        risk_score=round(risk_score, 4),
        contributing_wallets=wallet_addresses,
        metadata={
            "num_coordinated_wallets": len(wallet_addresses),
            "avg_smart_money_score": round(avg_sm_score, 4),
        },
    )


def _compute_risk_score(
    confidence: float,
    num_wallets: int,
    predicted_volume: float,
) -> float:
    """Lower confidence or fewer wallets → higher risk (closer to 1.0)."""
    base_risk = 1.0 - confidence
    # Fewer confirming wallets → higher risk
    wallet_risk = max(0.0, 1.0 - num_wallets / 10.0)
    return min(1.0, base_risk * 0.6 + wallet_risk * 0.4)


def _risk_level_from_score(score: float) -> RiskLevel:
    if score < 0.25:
        return RiskLevel.LOW
    if score < 0.5:
        return RiskLevel.MEDIUM
    if score < 0.75:
        return RiskLevel.HIGH
    return RiskLevel.CRITICAL
