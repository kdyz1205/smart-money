"""Signal and recommendation API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...integration.smart_money_agent import SmartMoneyAgent
from ...predictor.service import PredictorServiceImpl
from ...shared.models import AnalysisParams, Recommendation, Signal

router = APIRouter(prefix="/signals", tags=["signals"])

_predictor: PredictorServiceImpl | None = None
_smart_money_agent: SmartMoneyAgent | None = None


def configure(
    predictor: PredictorServiceImpl, smart_money_agent: SmartMoneyAgent
) -> None:
    global _predictor, _smart_money_agent
    _predictor = predictor
    _smart_money_agent = smart_money_agent


@router.get("/", response_model=list[Signal])
async def list_signals(limit: int = Query(default=50, le=200)) -> list[Signal]:
    """Get latest trading signals."""
    if not _predictor:
        raise HTTPException(status_code=503, detail="Predictor not ready")
    return _predictor.get_latest_signals(limit)


@router.get("/recommendations", response_model=list[Recommendation])
async def list_recommendations(
    limit: int = Query(default=50, le=200),
) -> list[Recommendation]:
    """Get latest recommendations (signals enriched with market context)."""
    if not _smart_money_agent:
        raise HTTPException(status_code=503, detail="Agent not ready")
    return _smart_money_agent.get_latest_recommendations(limit)


@router.put("/params")
async def update_params(params: AnalysisParams) -> dict:
    """Update analysis parameters (AI or manual tuning).

    This endpoint allows the control panel to adjust:
    - Anomaly detection sensitivity
    - Clustering parameters
    - Signal confidence thresholds
    - Prediction windows
    - Risk/volume/recency weights
    """
    if not _smart_money_agent:
        raise HTTPException(status_code=503, detail="Agent not ready")
    from ...shared.events import Event, EventType

    await _smart_money_agent._event_bus.publish(
        Event(event_type=EventType.PARAMS_UPDATED, payload=params)
    )
    return {"status": "updated", "params": params.model_dump()}


@router.get("/params", response_model=AnalysisParams)
async def get_params() -> AnalysisParams:
    """Get current analysis parameters."""
    if not _smart_money_agent:
        raise HTTPException(status_code=503, detail="Agent not ready")
    return _smart_money_agent._params
