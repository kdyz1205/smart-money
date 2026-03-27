"""Tests for the ingestion pipeline and cache."""

from smart_money.collector.cache import TxCache


def test_tx_cache_dedup() -> None:
    cache = TxCache(max_size=100)
    assert cache.is_new("0xabc")
    assert not cache.is_new("0xabc")
    assert cache.is_new("0xdef")
    assert len(cache) == 2


def test_tx_cache_eviction() -> None:
    cache = TxCache(max_size=3)
    cache.is_new("0x1")
    cache.is_new("0x2")
    cache.is_new("0x3")
    cache.is_new("0x4")  # should evict 0x1
    assert len(cache) == 3
    assert cache.is_new("0x1")  # 0x1 was evicted, so it's "new" again
    assert not cache.is_new("0x4")  # 0x4 is still cached
