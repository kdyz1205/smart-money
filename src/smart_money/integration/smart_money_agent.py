"""Smart Money Agent — the main orchestrator.

This agent owns the full pipeline:
  Collector → Analyzer → Predictor → Signal → Recommendation

It understands the CryptoAnalysisAgent's capabilities:
  - Subscribes to MARKET_CONTEXT_UPDATED to enrich its own signals
  - Combines on-chain smart-money signals with market fundamentals
  - Produces final RECOMMENDATION_READY events for external consumers

The SmartMoneyAgent needs from the CryptoAnalysisAgent:
  1. Current price & volume (to gauge market impact of smart-money moves)
  2. Trend direction (to filter signals — e.g., don't follow smart money
     buying into a crashing market unless conviction is very high)
  3. Volatility (to adjust risk scores)
  4. Liquidity (to assess whether the predicted volume can be absorbed)

Communication contract:
  - Consumes: NEW_TRANSACTIONS, MARKET_CONTEXT_UPDATED, PARAMS_UPDATED
  - Produces: SIGNAL_GENERATED, RECOMMENDATION_READY
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ..analyzer.service import AnalyzerServiceImpl
from ..collector.service import CollectorServiceImpl
from ..predictor.service import PredictorServiceImpl
from ..shared.constants import RiskLevel, Trend
from ..shared.events import Event, EventBus, EventType
from ..shared.models import (
    AnalysisParams,
    MarketContext,
    Recommendation,
    Signal,
    Transaction,
)

logger = logging.getLogger(__name__)


class SmartMoneyAgent:
    """Main agent that orchestrates smart-money detection and recommendation.

    Subscribes to on-chain data events and market context updates,
    runs the full analysis pipeline, and produces recommendations
    that combine both on-chain intelligence and market fundamentals.
    """

    def __init__(
        self,
        event_bus: EventBus,
        collector: CollectorServiceImpl,
        analyzer: AnalyzerServiceImpl,
        predictor: PredictorServiceImpl,
    ) -> None:
        self._event_bus = event_bus
        self._collector = collector
        self._analyzer = analyzer
        self._predictor = predictor

        # Market context cache: token_symbol → latest MarketContext
        self._market_contexts: dict[str, MarketContext] = {}
        self._recommendations: list[Recommendation] = []
        self._params = AnalysisParams()

    async def start(self) -> None:
        """Subscribe to all relevant events."""
        self._event_bus.subscribe(
            EventType.NEW_TRANSACTIONS, self._on_new_transactions
        )
        self._event_bus.subscribe(
            EventType.MARKET_CONTEXT_UPDATED, self._on_market_context
        )
        self._event_bus.subscribe(
            EventType.PARAMS_UPDATED, self._on_params_updated
        )
        logger.info("SmartMoneyAgent started")

    async def stop(self) -> None:
        logger.info("SmartMoneyAgent stopped")

    async def handle_event(self, event: Event) -> None:
        handlers = {
            EventType.NEW_TRANSACTIONS: self._on_new_transactions,
            EventType.MARKET_CONTEXT_UPDATED: self._on_market_context,
            EventType.PARAMS_UPDATED: self._on_params_updated,
        }
        handler = handlers.get(event.event_type)
        if handler:
            await handler(event)

    async def _on_new_transactions(self, event: Event) -> None:
        """Full pipeline: analyze → predict → recommend."""
        txs: list[Transaction] = event.payload

        # Step 1: Wallet analysis
        profiles = await self._analyzer.analyze_wallets(txs)

        # Step 2: Feature extraction
        features = self._analyzer.extract_features(txs)

        # Step 3: Prediction
        signals = await self._predictor.predict(profiles, features, txs)

        # Step 4: Enrich signals with market context → recommendations
        for signal in signals:
            recommendation = self._create_recommendation(signal)
            self._recommendations.append(recommendation)
            await self._event_bus.publish(
                Event(
                    event_type=EventType.RECOMMENDATION_READY,
                    payload=recommendation,
                )
            )

        if self._recommendations and len(self._recommendations) > 500:
            self._recommendations = self._recommendations[-500:]

        logger.info(
            "Pipeline complete: %d txs → %d signals → %d recommendations",
            len(txs),
            len(signals),
            len(signals),
        )

    async def _on_market_context(self, event: Event) -> None:
        """Cache market context from the CryptoAnalysisAgent."""
        ctx: MarketContext = event.payload
        self._market_contexts[ctx.token_symbol] = ctx
        logger.debug("Market context updated for %s: %s", ctx.token_symbol, ctx.trend)

    async def _on_params_updated(self, event: Event) -> None:
        """Update analysis parameters from control panel."""
        params: AnalysisParams = event.payload
        self._params = params
        self._analyzer.update_params(params)
        logger.info("Parameters updated via control panel")

    def _create_recommendation(self, signal: Signal) -> Recommendation:
        """Combine a smart-money signal with market context to produce a recommendation.

        This is where the two agents' knowledge merges:
        - SmartMoneyAgent provides: wallet behavior, accumulation patterns, confidence
        - CryptoAnalysisAgent provides: price trend, volatility, liquidity
        """
        market = self._market_contexts.get(signal.token_symbol)
        action = self._decide_action(signal, market)
        reasoning = self._build_reasoning(signal, market, action)

        return Recommendation(
            signal=signal,
            market_context=market,
            action=action,
            reasoning=reasoning,
            timestamp=datetime.now(timezone.utc),
        )

    def _decide_action(self, signal: Signal, market: MarketContext | None) -> str:
        """Decision logic combining on-chain and market signals.

        Rules:
        1. High confidence + bullish/neutral trend → "buy"
        2. High confidence + bearish trend → "watch" (wait for confirmation)
        3. Medium confidence → "watch"
        4. High risk or low liquidity → "avoid"
        5. Smart-money selling pattern → "sell"
        """
        if signal.risk_level == RiskLevel.CRITICAL:
            return "avoid"

        if signal.signal_type.value == "smart_exit":
            return "sell"

        # If we have market context, factor it in
        if market:
            # Low liquidity → can't safely enter
            if market.liquidity_usd and market.liquidity_usd < 50_000:
                return "avoid"

            # High volatility + low confidence → too risky
            if (market.volatility_24h or 0) > 0.7 and signal.confidence < 0.7:
                return "avoid"

            if signal.confidence >= self._params.signal_confidence_threshold:
                if market.trend in (Trend.BULLISH, Trend.NEUTRAL):
                    return "buy"
                elif market.trend == Trend.BEARISH:
                    # Smart money buying into a bearish market — watch closely
                    if signal.confidence > 0.8:
                        return "buy"  # Very high conviction overrides trend
                    return "watch"

        # No market context available — rely purely on on-chain signal
        if signal.confidence >= self._params.signal_confidence_threshold:
            return "buy"

        return "watch"

    def _build_reasoning(
        self, signal: Signal, market: MarketContext | None, action: str
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        parts = [
            f"{signal.signal_type.value} detected for {signal.token_symbol}",
            f"confidence={signal.confidence:.0%}",
            f"risk={signal.risk_level.value}",
            f"{len(signal.contributing_wallets)} smart-money wallets involved",
        ]

        if market:
            parts.append(f"market trend={market.trend.value}")
            if market.price_change_24h_pct is not None:
                parts.append(f"24h change={market.price_change_24h_pct:+.1f}%")
            if market.volatility_24h is not None:
                parts.append(f"volatility={market.volatility_24h:.0%}")

        parts.append(f"→ action: {action}")
        return " | ".join(parts)

    def get_latest_recommendations(self, limit: int = 50) -> list[Recommendation]:
        return self._recommendations[-limit:]
