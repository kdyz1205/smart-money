"""Transaction cache for deduplication."""

from __future__ import annotations

from collections import OrderedDict


class TxCache:
    """LRU-based transaction hash cache to prevent duplicate processing."""

    def __init__(self, max_size: int = 100_000) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._max_size = max_size

    def is_new(self, tx_hash: str) -> bool:
        """Return True if this tx_hash has not been seen before, and record it."""
        if tx_hash in self._seen:
            return False
        self._seen[tx_hash] = None
        if len(self._seen) > self._max_size:
            self._seen.popitem(last=False)
        return True

    def __len__(self) -> int:
        return len(self._seen)
