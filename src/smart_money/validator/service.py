"""Validator service — orchestrates all validation sub-modules.

Sits after the predictor in the pipeline:
  Collector → Analyzer → Predictor → **Validator** → Recommendation

Subscribes to:
  - NEW_TRANSACTIONS: run fill-speed and volume-surge analysis
  - SIGNAL_GENERATED: record for backtest correlation
  - SMART_MONEY_IDENTIFIED: track wallet performance

Publishes:
  - FILL_SPEED_ALERT
  - VOLUME_SURGE_DETECTED
  - BREAKOUT_PRESIGNAL
  - BACKTEST_COMPLETE (nightly)
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone

from ..shared.events import Event, EventBus, EventType
from ..shared.models import (
    AnalysisParams,
    BacktestResult,
    BreakoutPresignal,
    FillSpeedAlert,
    Signal,
    Transaction,
    VolumeSurge,
    WalletPerformanceRecord,
    WalletProfile,
)
from .backtest import PriceEvent, run_backtest
from .breakout import (
    detect_buy_sell_asymmetry,
    detect_concentration_surge,
    detect_stealth_then_aggressive,
)
from .fill_speed import detect_fill_speed_alerts
from .volume_filter import detect_volume_surges
from .wallet_performance import compute_wallet_performance, detect_cex_cross_flow

logger = logging.getLogger(__name__)


class ValidatorService:
    """Runs validation checks on incoming data and published signals.

    Maintains state for:
    - Historical fill speeds per wallet (for percentile calculation)
    - Signal history (for backtesting)
    - Wallet performance records
    - Recent transaction windows (for breakout detection)
    """

    def __init__(self, event_bus: EventBus, params: AnalysisParams | None = None) -> None:
        self._event_bus = event_bus
        self._params = params or AnalysisParams()

        # State
        self._smart_money_addresses: set[str] = set()
        self._wallet_fill_speeds: dict[str, list[float]] = defaultdict(list)
        self._signal_history: list[Signal] = []
        self._wallet_performance: dict[str, WalletPerformanceRecord] = {}
        self._recent_txs: list[Transaction] = []  # rolling window
        self._max_recent_txs = 50_000
        self._latest_backtest: BacktestResult | None = None

    async def start(self) -> None:
        """Subscribe to relevant events."""
        self._event_bus.subscribe(EventType.NEW_TRANSACTIONS, self._on_new_transactions)
        self._event_bus.subscribe(EventType.SIGNAL_GENERATED, self._on_signal)
        self._event_bus.subscribe(EventType.SMART_MONEY_IDENTIFIED, self._on_smart_money)
        self._event_bus.subscribe(EventType.PARAMS_UPDATED, self._on_params_updated)
        logger.info("ValidatorService started")

    async def stop(self) -> None:
        logger.info("ValidatorService stopped")

    async def _on_new_transactions(self, event: Event) -> None:
        """Run fill-speed, volume-surge, and breakout detection on new txs."""
        txs: list[Transaction] = event.payload
        self._recent_txs.extend(txs)
        if len(self._recent_txs) > self._max_recent_txs:
            self._recent_txs = self._recent_txs[-self._max_recent_txs:]

        # Run all validation checks concurrently
        await asyncio.gather(
            self._check_fill_speed(txs),
            self._check_volume_surge(txs),
            self._check_breakout_presignals(txs),
            self._check_cex_cross_flow(txs),
        )

    async def _on_signal(self, event: Event) -> None:
        """Record signal for backtesting."""
        signal: Signal = event.payload
        self._signal_history.append(signal)
        if len(self._signal_history) > 10_000:
            self._signal_history = self._signal_history[-10_000:]

    async def _on_smart_money(self, event: Event) -> None:
        """Update smart money address set."""
        profiles: list[WalletProfile] = event.payload
        for p in profiles:
            if p.is_smart_money:
                self._smart_money_addresses.add(p.address.lower())

    async def _on_params_updated(self, event: Event) -> None:
        self._params = event.payload

    async def _check_fill_speed(self, txs: list[Transaction]) -> None:
        """Check fill speed for each smart-money wallet in the batch."""
        # Group txs by wallet → token
        wallet_token_txs: dict[str, dict[str, list[Transaction]]] = defaultdict(lambda: defaultdict(list))
        for tx in txs:
            addr = tx.from_addr.lower()
            if addr in self._smart_money_addresses:
                token = tx.token_address or tx.to_addr
                wallet_token_txs[addr][token].append(tx)

        for wallet_addr, token_txs in wallet_token_txs.items():
            historical = self._wallet_fill_speeds.get(wallet_addr, [])
            alerts = detect_fill_speed_alerts(
                wallet_txs_by_token=token_txs,
                wallet_address=wallet_addr,
                historical_speeds=historical,
                market_volumes={},  # would be filled from market data
                percentile_threshold=self._params.fill_speed_percentile,
                stealth_threshold=self._params.fill_speed_stealth_threshold,
                rapid_interval_sec=self._params.fill_speed_interval_sec,
            )
            for alert in alerts:
                # Record speed for future percentile calc
                self._wallet_fill_speeds[wallet_addr].append(
                    alert.fill_speed_usd_per_sec
                )
                await self._event_bus.publish(
                    Event(event_type=EventType.FILL_SPEED_ALERT, payload=alert)
                )
                logger.warning(
                    "FILL SPEED ALERT: %s on %s — %.2f USD/s (stealth=%.1f)",
                    wallet_addr[:10],
                    alert.token_symbol,
                    alert.fill_speed_usd_per_sec,
                    alert.stealth_score,
                )

    async def _check_volume_surge(self, txs: list[Transaction]) -> None:
        """Check for smart-money volume surges per token."""
        # Group by token
        by_token: dict[str, list[Transaction]] = defaultdict(list)
        for tx in txs:
            token = tx.token_address or tx.to_addr
            by_token[token].append(tx)

        for token_addr, token_txs in by_token.items():
            total_vol = sum(float(tx.value_wei) for tx in token_txs)
            for window_min in (5, 15):
                surge = detect_volume_surges(
                    txs=token_txs,
                    smart_money_addresses=self._smart_money_addresses,
                    total_market_volume_usd=total_vol,
                    avg_volume_24h_usd=total_vol * 288,  # rough estimate
                    current_price=1.0,
                    price_at_window_start=1.0,
                    window_minutes=window_min,
                    sm_ratio_threshold=self._params.volume_surge_sm_ratio,
                    multiplier_threshold=self._params.volume_surge_multiplier,
                )
                if surge:
                    await self._event_bus.publish(
                        Event(
                            event_type=EventType.VOLUME_SURGE_DETECTED,
                            payload=surge,
                        )
                    )
                    logger.warning(
                        "VOLUME SURGE: %s %dmin — SM ratio=%.0f%% %s",
                        surge.token_symbol,
                        window_min,
                        surge.sm_volume_ratio * 100,
                        "STEALTH" if surge.is_stealth_accumulation else "",
                    )

    async def _check_breakout_presignals(self, txs: list[Transaction]) -> None:
        """Run all breakout pre-signal detectors."""
        if not self._smart_money_addresses:
            return

        # Split recent txs into two halves for concentration comparison
        mid = len(self._recent_txs) // 2
        older = self._recent_txs[:mid]
        recent = self._recent_txs[mid:]

        # Group by token for breakout checks
        by_token: dict[str, list[Transaction]] = defaultdict(list)
        for tx in txs:
            token = tx.token_address or tx.to_addr
            by_token[token].append(tx)

        for token_addr, token_txs in by_token.items():
            older_token = [t for t in older if (t.token_address or t.to_addr) == token_addr]
            recent_token = [t for t in recent if (t.token_address or t.to_addr) == token_addr]

            presignals: list[BreakoutPresignal] = []

            # A. Concentration surge
            cs = detect_concentration_surge(
                recent_token, older_token,
                self._smart_money_addresses,
                threshold_pct=self._params.breakout_concentration_pct,
            )
            if cs:
                presignals.append(cs)

            # B. Buy/sell asymmetry
            bsa = detect_buy_sell_asymmetry(
                token_txs,
                self._smart_money_addresses,
                current_price_wei=1.0,
                ratio_threshold=self._params.breakout_buy_sell_ratio,
            )
            if bsa:
                presignals.append(bsa)

            # D. Stealth → aggressive (per wallet)
            for addr in self._smart_money_addresses:
                wallet_txs = [t for t in token_txs if t.from_addr.lower() == addr]
                sta = detect_stealth_then_aggressive(wallet_txs, addr)
                if sta:
                    presignals.append(sta)

            for ps in presignals:
                await self._event_bus.publish(
                    Event(event_type=EventType.BREAKOUT_PRESIGNAL, payload=ps)
                )

    async def _check_cex_cross_flow(self, txs: list[Transaction]) -> None:
        """Check for CEX-to-DEX capital flow patterns."""
        flows = detect_cex_cross_flow(txs, self._smart_money_addresses)
        for flow in flows:
            logger.info("CEX→DEX flow: %s", flow)

    async def run_nightly_backtest(
        self, price_history: list[PriceEvent] | None = None
    ) -> BacktestResult:
        """Run the nightly backtest and publish results."""
        history = price_history or []
        result = run_backtest(
            signals=self._signal_history,
            price_history=history,
            lookback_days=self._params.prediction_window_hours // 24 or 90,
        )
        self._latest_backtest = result
        await self._event_bus.publish(
            Event(event_type=EventType.BACKTEST_COMPLETE, payload=result)
        )
        return result

    async def run_backtest_loop(self, interval_hours: float = 24.0) -> None:
        """Long-running loop that triggers nightly backtesting."""
        try:
            while True:
                await asyncio.sleep(interval_hours * 3600)
                logger.info("Running nightly backtest")
                await self.run_nightly_backtest()
        except asyncio.CancelledError:
            pass

    def get_latest_backtest(self) -> BacktestResult | None:
        return self._latest_backtest

    def get_wallet_performance(self, address: str) -> WalletPerformanceRecord | None:
        return self._wallet_performance.get(address.lower())

    def get_high_confidence_wallets(self) -> list[str]:
        """Return addresses with win_rate >= threshold."""
        return [
            addr for addr, record in self._wallet_performance.items()
            if record.is_high_confidence
        ]

    def get_smart_money_addresses(self) -> set[str]:
        return self._smart_money_addresses
