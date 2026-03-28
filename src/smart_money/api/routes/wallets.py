"""Wallet-related API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ...analyzer.service import AnalyzerServiceImpl
from ...collector.service import CollectorServiceImpl
from ...shared.models import WalletProfile

router = APIRouter(prefix="/wallets", tags=["wallets"])

# These get injected by the app factory
_analyzer: AnalyzerServiceImpl | None = None
_collector: CollectorServiceImpl | None = None


def configure(analyzer: AnalyzerServiceImpl, collector: CollectorServiceImpl) -> None:
    global _analyzer, _collector
    _analyzer = analyzer
    _collector = collector


@router.get("/", response_model=list[WalletProfile])
async def list_wallets() -> list[WalletProfile]:
    """List all analyzed wallet profiles."""
    if not _analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not ready")
    return _analyzer.get_all_profiles()


@router.get("/smart-money", response_model=list[WalletProfile])
async def list_smart_money() -> list[WalletProfile]:
    """List wallets identified as smart money."""
    if not _analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not ready")
    profiles = _analyzer.get_all_profiles()
    return [p for p in profiles if p.is_smart_money]


@router.get("/{address}", response_model=WalletProfile)
async def get_wallet(address: str) -> WalletProfile:
    """Get profile for a specific wallet."""
    if not _analyzer:
        raise HTTPException(status_code=503, detail="Analyzer not ready")
    profile = _analyzer.get_profile(address)
    if not profile:
        raise HTTPException(status_code=404, detail="Wallet not found")
    return profile


@router.post("/track/{address}")
async def track_wallet(address: str) -> dict:
    """Start tracking a wallet address."""
    if not _collector:
        raise HTTPException(status_code=503, detail="Collector not ready")
    # Basic address validation
    addr = address.strip()
    if not addr or len(addr) < 10:
        raise HTTPException(status_code=400, detail="Invalid wallet address")
    await _collector.add_wallet(addr)
    return {"status": "tracking", "address": addr.lower()}


@router.delete("/track/{address}")
async def untrack_wallet(address: str) -> dict:
    """Stop tracking a wallet address."""
    if not _collector:
        raise HTTPException(status_code=503, detail="Collector not ready")
    await _collector.remove_wallet(address)
    return {"status": "untracked", "address": address}
