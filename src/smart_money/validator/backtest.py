"""Historical backtest engine — validate signal quality against real outcomes.

Runs nightly (or on-demand):
  1. Find all tokens that had a breakout (15%+ rise in 30 min) in the lookback period
  2. Check if our signals appeared before those breakouts
  3. Compute precision, recall, F1, and average lead time
  4. Output to the model health dashboard
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..shared.models import BacktestResult, Signal

logger = logging.getLogger(__name__)


@dataclass
class PriceEvent:
    """A historical price point for backtesting."""

    token_address: str
    token_symbol: str
    timestamp: datetime
    price_usd: float


@dataclass
class BreakoutEvent:
    """A detected breakout in historical data."""

    token_address: str
    token_symbol: str
    breakout_time: datetime
    price_before: float
    price_peak: float
    rise_pct: float


def find_breakouts(
    price_history: list[PriceEvent],
    rise_threshold_pct: float = 15.0,
    window_minutes: int = 30,
) -> list[BreakoutEvent]:
    """Identify all breakout events in price history.

    A breakout is defined as price rising >= threshold% within window_minutes.
    """
    if not price_history:
        return []

    # Group by token
    by_token: dict[str, list[PriceEvent]] = {}
    for pe in price_history:
        by_token.setdefault(pe.token_address, []).append(pe)

    breakouts: list[BreakoutEvent] = []

    for token_addr, events in by_token.items():
        sorted_events = sorted(events, key=lambda e: e.timestamp)

        for i, start in enumerate(sorted_events):
            window_end = start.timestamp + timedelta(minutes=window_minutes)
            peak_price = start.price_usd

            for j in range(i + 1, len(sorted_events)):
                if sorted_events[j].timestamp > window_end:
                    break
                peak_price = max(peak_price, sorted_events[j].price_usd)

            if start.price_usd > 0:
                rise_pct = (peak_price - start.price_usd) / start.price_usd * 100
                if rise_pct >= rise_threshold_pct:
                    breakouts.append(
                        BreakoutEvent(
                            token_address=token_addr,
                            token_symbol=start.token_symbol,
                            breakout_time=start.timestamp,
                            price_before=start.price_usd,
                            price_peak=peak_price,
                            rise_pct=rise_pct,
                        )
                    )
                    # Skip ahead to avoid double-counting overlapping breakouts
                    break

    logger.info("Found %d breakout events in price history", len(breakouts))
    return breakouts


def run_backtest(
    signals: list[Signal],
    price_history: list[PriceEvent],
    lookback_days: int = 90,
    rise_threshold_pct: float = 15.0,
    max_lead_time_minutes: float = 60.0,
) -> BacktestResult:
    """Run full backtest: compare signals against actual breakouts.

    For each breakout, check if we had a signal within max_lead_time_minutes
    before the breakout started. Compute precision/recall/F1.
    """
    breakouts = find_breakouts(price_history, rise_threshold_pct)

    # Match signals to breakouts
    signals_matched = 0
    breakouts_caught = 0
    lead_times: list[float] = []
    matched_breakout_ids: set[int] = set()

    by_type: dict[str, dict[str, int]] = {}

    for b_idx, breakout in enumerate(breakouts):
        for signal in signals:
            if signal.token_address != breakout.token_address:
                continue

            lead_time = (
                breakout.breakout_time - signal.timestamp
            ).total_seconds() / 60

            if 0 < lead_time <= max_lead_time_minutes:
                if b_idx not in matched_breakout_ids:
                    breakouts_caught += 1
                    matched_breakout_ids.add(b_idx)
                signals_matched += 1
                lead_times.append(lead_time)

                # Track per signal type
                st = signal.signal_type.value
                if st not in by_type:
                    by_type[st] = {"matched": 0, "total": 0}
                by_type[st]["matched"] += 1

    # Count total signals per type
    for signal in signals:
        st = signal.signal_type.value
        if st not in by_type:
            by_type[st] = {"matched": 0, "total": 0}
        by_type[st]["total"] += 1

    total_signals = len(signals)
    total_breakouts = len(breakouts)

    precision = signals_matched / max(total_signals, 1)
    recall = breakouts_caught / max(total_breakouts, 1)
    f1 = (
        2 * precision * recall / max(precision + recall, 0.001)
    )
    avg_lead = float(sum(lead_times) / max(len(lead_times), 1))

    # Per-type stats
    per_signal_type: dict[str, dict[str, float]] = {}
    for st, counts in by_type.items():
        total = counts["total"]
        matched = counts["matched"]
        per_signal_type[st] = {
            "precision": matched / max(total, 1),
            "matched": float(matched),
            "total": float(total),
        }

    result = BacktestResult(
        run_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        lookback_days=lookback_days,
        total_breakouts_found=total_breakouts,
        signals_before_breakout=signals_matched,
        signals_total=total_signals,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        avg_lead_time_minutes=round(avg_lead, 1),
        per_signal_type=per_signal_type,
    )

    logger.info(
        "Backtest complete: P=%.2f R=%.2f F1=%.2f avg_lead=%.1fm (%d breakouts, %d signals)",
        precision,
        recall,
        f1,
        avg_lead,
        total_breakouts,
        total_signals,
    )
    return result
