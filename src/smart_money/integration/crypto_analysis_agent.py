"""Crypto Analysis Agent — provides market context and fundamental analysis.

This agent understands the SmartMoneyAgent's needs:
  - It listens for SIGNAL_GENERATED events and enriches them with market data
  - It proactively publishes MARKET_CONTEXT_UPDATED for tokens being tracked
  - It responds to the SmartMoneyAgent's requests for specific token analysis

The CryptoAnalysisAgent focuses on:
  1. Price, volume, and liquidity data
  2. Market sentiment (trend detection)
  3. Volatility assessment
  4. Providing context so the SmartMoneyAgent can make better recommendations

Communication contract:
  - Consumes: SIGNAL_GENERATED (to know which tokens matter)
  - Produces: MARKET_CONTEXT_UPDATED (enrichment data for SmartMoneyAgent)
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

import aiohttp

from ..shared.config import Settings
from ..shared.constants import Chain, Trend
from ..shared.events import Event, EventBus, EventType
from ..shared.models import MarketContext, Signal

logger = logging.getLogger(__name__)


class CryptoAnalysisAgent:
    """Provides market context and fundamental analysis for tokens.

    Subscribes to signals from the SmartMoneyAgent and enriches them
    with market data. Also independently monitors market conditions
    for tracked tokens.
    """

    def __init__(self, event_bus: EventBus, config: Settings) -> None:
        self._event_bus = event_bus
        self._config = config
        self._tracked_tokens: set[str] = set()
        self._market_cache: dict[str, tuple[float, MarketContext]] = {}  # (timestamp, ctx)
        self._cache_ttl_sec = 300.0  # 5 minute cache TTL
        self._running = False
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Subscribe to events and start the market monitoring loop."""
        self._event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal)
        self._running = True
        logger.info("CryptoAnalysisAgent started")

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("CryptoAnalysisAgent stopped")

    async def handle_event(self, event: Event) -> None:
        if event.event_type == EventType.SIGNAL_GENERATED:
            await self._on_signal(event)

    async def _on_signal(self, event: Event) -> None:
        """When SmartMoneyAgent generates a signal, fetch market context for that token."""
        signal: Signal = event.payload
        token = signal.token_symbol
        self._tracked_tokens.add(token)

        context = await self.get_market_context(token)
        if context:
            await self._event_bus.publish(
                Event(
                    event_type=EventType.MARKET_CONTEXT_UPDATED,
                    payload=context,
                )
            )

    async def get_market_context(self, token_symbol: str) -> MarketContext | None:
        """Fetch and compute market context for a token.

        Combines price data, volume, and basic trend analysis.
        In production, this would call CoinGecko, DEX APIs, etc.
        """
        # Check cache with TTL
        import time as _time
        cached = self._market_cache.get(token_symbol)
        if cached:
            cache_ts, cache_ctx = cached
            if _time.time() - cache_ts < self._cache_ttl_sec:
                return cache_ctx

        try:
            price_data = await self._fetch_price_data(token_symbol)
            if not price_data:
                # Return stale cache if available
                return cached[1] if cached else None

            trend = self._compute_trend(price_data)
            volatility = self._compute_volatility(price_data)

            context = MarketContext(
                token_symbol=token_symbol,
                price_usd=price_data.get("price", 0.0),
                price_change_24h_pct=price_data.get("price_change_24h", 0.0),
                volume_24h_usd=price_data.get("volume_24h", 0.0),
                market_cap_usd=price_data.get("market_cap"),
                liquidity_usd=price_data.get("liquidity"),
                trend=trend,
                volatility_24h=volatility,
                timestamp=datetime.now(timezone.utc),
            )

            self._market_cache[token_symbol] = (_time.time(), context)
            return context

        except Exception:
            logger.exception("Failed to get market context for %s", token_symbol)
            return cached[1] if cached else None

    async def _fetch_price_data(self, token_symbol: str) -> dict | None:
        """Fetch price data from external API.

        Currently uses a mock/fallback. Replace with CoinGecko, OKX, etc.
        """
        # In production: call real API. For now, return structure expected downstream.
        try:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10),
                    trust_env=True,
                )

            # OKX market ticker endpoint
            url = "https://www.okx.com/api/v5/market/ticker"
            params = {"instId": f"{token_symbol}-USDT"}
            async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                tickers = data.get("data", [])
                if not tickers:
                    return None
                t = tickers[0]
                last = float(t.get("last", 0))
                open_24h = float(t.get("open24h", last))
                change_pct = ((last - open_24h) / open_24h * 100) if open_24h else 0
                return {
                    "price": last,
                    "price_change_24h": change_pct,
                    "volume_24h": float(t.get("vol24h", 0)) * last,
                    "market_cap": None,
                    "liquidity": None,
                }
        except Exception:
            logger.debug("OKX API unavailable for %s, using cached data", token_symbol)
            return None

    def _compute_trend(self, price_data: dict) -> Trend:
        change = price_data.get("price_change_24h", 0.0)
        if change is None:
            return Trend.NEUTRAL
        if change > 3.0:
            return Trend.BULLISH
        if change < -3.0:
            return Trend.BEARISH
        return Trend.NEUTRAL

    def _compute_volatility(self, price_data: dict) -> float:
        """Simple volatility estimate from 24h price change."""
        change = abs(price_data.get("price_change_24h", 0.0) or 0.0)
        return min(1.0, change / 20.0)

    def get_cached_context(self, token_symbol: str) -> MarketContext | None:
        cached = self._market_cache.get(token_symbol)
        return cached[1] if cached else None

    async def monitor_markets(self, interval_sec: float = 60.0) -> None:
        """Periodically refresh market data for all tracked tokens."""
        try:
            while self._running:
                for token in list(self._tracked_tokens):
                    context = await self.get_market_context(token)
                    if context:
                        await self._event_bus.publish(
                            Event(
                                event_type=EventType.MARKET_CONTEXT_UPDATED,
                                payload=context,
                            )
                        )
                await asyncio.sleep(interval_sec)
        except asyncio.CancelledError:
            pass
