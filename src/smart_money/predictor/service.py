"""Predictor service — orchestrates time-series analysis and signal generation.

Consumes wallet profiles + features, produces actionable signals.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from ..shared.events import Event, EventBus, EventType
from ..shared.models import Signal, Transaction, WalletFeatures, WalletProfile
from .signals import create_coordinated_buy_signal, create_signal_from_accumulation
from .timeseries import detect_accumulation, detect_coordinated_buying

logger = logging.getLogger(__name__)


class PredictorServiceImpl:
    """Runs prediction pipeline and publishes generated signals."""

    def __init__(self, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._signals: list[Signal] = []
        self._max_signals = 1000

    async def predict(
        self,
        profiles: list[WalletProfile],
        features: list[WalletFeatures],
        txs: list[Transaction] | None = None,
    ) -> list[Signal]:
        """Run full prediction pipeline.

        1. Group transactions by token
        2. Detect accumulation patterns per token
        3. Detect coordinated buying
        4. Generate and publish signals
        """
        profile_map = {p.address: p for p in profiles}
        signals: list[Signal] = []

        if not txs:
            return signals

        # Group by token
        token_txs: dict[str, list[Transaction]] = defaultdict(list)
        for tx in txs:
            key = tx.token_address or tx.to_addr
            if key:
                token_txs[key].append(tx)

        smart_addrs = {p.address.lower() for p in profiles if p.is_smart_money}

        for token_addr, ttxs in token_txs.items():
            if not token_addr or not ttxs:
                continue

            token_symbol = next(
                (tx.token_symbol for tx in ttxs if tx.token_symbol), "UNKNOWN"
            )
            # Infer chain from first tx
            tx_chain = ttxs[0].chain if ttxs else None

            # Separate buy/sell volumes (heuristic: from smart-money = buy intent tracking)
            buy_vols = [
                float(tx.value_wei) for tx in ttxs if tx.from_addr.lower() in smart_addrs
            ]
            sell_vols = [
                float(tx.value_wei) for tx in ttxs if tx.from_addr.lower() not in smart_addrs
            ]
            timestamps = [tx.timestamp.timestamp() for tx in ttxs]
            wallet_addrs = list({tx.from_addr.lower() for tx in ttxs if tx.from_addr.lower() in smart_addrs})

            # 1. Accumulation detection
            acc = detect_accumulation(
                token_address=token_addr,
                token_symbol=token_symbol,
                buy_volumes=buy_vols,
                sell_volumes=sell_vols,
                timestamps=timestamps,
                wallet_addresses=wallet_addrs,
            )
            if acc:
                from ..shared.constants import Chain
                sig = create_signal_from_accumulation(
                    acc, profile_map, chain=tx_chain or Chain.ETH
                )
                signals.append(sig)

            # 2. Coordinated buying detection
            wallet_buy_times: dict[str, list[float]] = defaultdict(list)
            for tx in ttxs:
                addr_lower = tx.from_addr.lower()
                if addr_lower in smart_addrs:
                    wallet_buy_times[addr_lower].append(tx.timestamp.timestamp())

            coordinated = detect_coordinated_buying(wallet_buy_times)
            if coordinated:
                from ..shared.constants import Chain
                sig = create_coordinated_buy_signal(
                    token_address=token_addr,
                    token_symbol=token_symbol,
                    wallet_addresses=coordinated,
                    profiles=profile_map,
                    chain=tx_chain or Chain.ETH,
                )
                signals.append(sig)

        # Publish each signal
        for sig in signals:
            self._signals.append(sig)
            await self._event_bus.publish(
                Event(event_type=EventType.SIGNAL_GENERATED, payload=sig)
            )

        # Keep bounded history
        if len(self._signals) > self._max_signals:
            self._signals = self._signals[-self._max_signals:]

        logger.info("Prediction complete: %d signals generated", len(signals))
        return signals

    def get_latest_signals(self, limit: int = 50) -> list[Signal]:
        return self._signals[-limit:]
