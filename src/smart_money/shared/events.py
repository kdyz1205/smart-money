"""Async event bus for decoupled inter-module and inter-agent communication.

All modules publish and subscribe to events through this bus.
Payloads are always Pydantic models from shared.models.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    # Collector → Analyzer
    NEW_TRANSACTIONS = "new_transactions"
    NEW_TOKEN_TRANSFERS = "new_token_transfers"

    # Analyzer → Predictor
    WALLET_PROFILE_UPDATED = "wallet_profile_updated"
    ANOMALY_DETECTED = "anomaly_detected"
    SMART_MONEY_IDENTIFIED = "smart_money_identified"

    # Predictor → Integration
    SIGNAL_GENERATED = "signal_generated"

    # CryptoAnalysisAgent → SmartMoneyAgent
    MARKET_CONTEXT_UPDATED = "market_context_updated"

    # SmartMoneyAgent → external consumers
    RECOMMENDATION_READY = "recommendation_ready"

    # Validator → Integration / Visualization
    FILL_SPEED_ALERT = "fill_speed_alert"
    VOLUME_SURGE_DETECTED = "volume_surge_detected"
    BREAKOUT_PRESIGNAL = "breakout_presignal"
    BACKTEST_COMPLETE = "backtest_complete"

    # Control panel → modules
    PARAMS_UPDATED = "params_updated"


@dataclass
class Event:
    event_type: EventType
    payload: Any
    timestamp: float = field(default_factory=time.time)


Handler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """In-process async event bus backed by asyncio.Queue.

    Single instance shared across all services. Replace the implementation
    with Redis Streams or NATS later without changing subscribers.
    """

    def __init__(self) -> None:
        self._subscribers: dict[EventType, list[Handler]] = {}
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._running = False

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._subscribers.setdefault(event_type, []).append(handler)
        logger.debug("Subscribed %s to %s", handler.__qualname__, event_type)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        handlers = self._subscribers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    async def publish(self, event: Event) -> None:
        await self._queue.put(event)

    async def run(self) -> None:
        """Main dispatch loop — run as a long-lived asyncio task."""
        self._running = True
        logger.info("EventBus started")
        try:
            while self._running:
                event = await self._queue.get()
                handlers = list(self._subscribers.get(event.event_type, []))
                for handler in handlers:
                    asyncio.create_task(self._safe_dispatch(handler, event))
        except asyncio.CancelledError:
            logger.info("EventBus shutting down")

    async def _safe_dispatch(self, handler: Handler, event: Event) -> None:
        try:
            await handler(event)
        except Exception:
            logger.exception(
                "Handler %s failed for event %s",
                handler.__qualname__,
                event.event_type,
            )

    def stop(self) -> None:
        self._running = False
