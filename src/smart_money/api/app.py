"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..analyzer.service import AnalyzerServiceImpl
from ..collector.service import CollectorServiceImpl
from ..integration.smart_money_agent import SmartMoneyAgent
from ..predictor.service import PredictorServiceImpl
from .routes import health, signals, wallets


def create_app(
    analyzer: AnalyzerServiceImpl,
    collector: CollectorServiceImpl,
    predictor: PredictorServiceImpl,
    smart_money_agent: SmartMoneyAgent,
) -> FastAPI:
    """Wire all dependencies and return a configured FastAPI app."""
    app = FastAPI(
        title="Smart Money Chain Agent",
        description="API for smart-money wallet tracking, signals, and recommendations",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Inject dependencies into route modules
    wallets.configure(analyzer=analyzer, collector=collector)
    signals.configure(predictor=predictor, smart_money_agent=smart_money_agent)

    # Register routers
    app.include_router(health.router)
    app.include_router(wallets.router)
    app.include_router(signals.router)

    return app
