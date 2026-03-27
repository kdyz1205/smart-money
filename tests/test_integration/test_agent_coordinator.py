"""Tests for inter-agent event flow."""

import asyncio

import pytest

from smart_money.shared.events import Event, EventBus, EventType
from smart_money.shared.models import MarketContext, Signal
from smart_money.shared.constants import Chain, SignalType, Trend


@pytest.mark.asyncio
async def test_event_bus_pubsub() -> None:
    bus = EventBus()
    received: list[Event] = []

    async def handler(event: Event) -> None:
        received.append(event)

    bus.subscribe(EventType.SIGNAL_GENERATED, handler)

    # Start bus in background
    task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)

    await bus.publish(
        Event(
            event_type=EventType.SIGNAL_GENERATED,
            payload={"test": True},
        )
    )
    await asyncio.sleep(0.1)

    assert len(received) == 1
    assert received[0].payload == {"test": True}

    bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_event_bus_multiple_subscribers() -> None:
    bus = EventBus()
    count_a = []
    count_b = []

    async def handler_a(event: Event) -> None:
        count_a.append(1)

    async def handler_b(event: Event) -> None:
        count_b.append(1)

    bus.subscribe(EventType.MARKET_CONTEXT_UPDATED, handler_a)
    bus.subscribe(EventType.MARKET_CONTEXT_UPDATED, handler_b)

    task = asyncio.create_task(bus.run())
    await asyncio.sleep(0.05)

    await bus.publish(
        Event(event_type=EventType.MARKET_CONTEXT_UPDATED, payload=None)
    )
    await asyncio.sleep(0.1)

    assert len(count_a) == 1
    assert len(count_b) == 1

    bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
