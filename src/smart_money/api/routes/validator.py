"""Validator API endpoints — fill-speed alerts, volume surges, backtest results."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ...shared.models import BacktestResult, FillSpeedAlert, VolumeSurge
from ...validator.service import ValidatorService

router = APIRouter(prefix="/validator", tags=["validator"])

_validator: ValidatorService | None = None


def configure(validator: ValidatorService) -> None:
    global _validator
    _validator = validator


@router.get("/backtest", response_model=BacktestResult | None)
async def get_latest_backtest() -> BacktestResult | None:
    """Get the latest backtest result."""
    if not _validator:
        raise HTTPException(status_code=503, detail="Validator not ready")
    return _validator.get_latest_backtest()


@router.post("/backtest/run", response_model=BacktestResult)
async def trigger_backtest() -> BacktestResult:
    """Manually trigger a backtest run."""
    if not _validator:
        raise HTTPException(status_code=503, detail="Validator not ready")
    return await _validator.run_nightly_backtest()


@router.get("/high-confidence-wallets")
async def get_high_confidence_wallets() -> dict:
    """Get wallets with win_rate >= 65%."""
    if not _validator:
        raise HTTPException(status_code=503, detail="Validator not ready")
    wallets = _validator.get_high_confidence_wallets()
    return {"count": len(wallets), "wallets": wallets}


@router.get("/smart-money-addresses")
async def get_smart_money_addresses() -> dict:
    """Get all currently tracked smart-money addresses."""
    if not _validator:
        raise HTTPException(status_code=503, detail="Validator not ready")
    addrs = sorted(_validator.get_smart_money_addresses())
    return {"count": len(addrs), "addresses": addrs}
