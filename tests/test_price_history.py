# tests/test_price_history.py
"""Tests for PriceHistoryFetcher — CLOB price history retrieval and caching."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
from datetime import date, datetime

from backtesting import PriceHistoryFetcher, PriceSnapshot


class TestPriceSnapshot:
    def test_from_clob_response(self):
        """Parse CLOB history response into PriceSnapshot list."""
        raw = {"history": [
            {"t": 1711929600, "p": 0.35},
            {"t": 1711933200, "p": 0.37},
            {"t": 1711936800, "p": 0.40},
        ]}
        snaps = PriceSnapshot.from_clob_response(raw)
        assert len(snaps) == 3
        assert snaps[0].price == 0.35
        assert snaps[0].timestamp == 1711929600
        assert isinstance(snaps[0].dt, datetime)

    def test_empty_history(self):
        snaps = PriceSnapshot.from_clob_response({"history": []})
        assert snaps == []

    def test_missing_history_key(self):
        snaps = PriceSnapshot.from_clob_response({})
        assert snaps == []


class TestPriceAtTime:
    def test_price_at_exact_time(self):
        """When queried at an exact snapshot time, return that price."""
        fetcher = PriceHistoryFetcher(cache_dir=tempfile.mkdtemp())
        snaps = [
            PriceSnapshot(timestamp=100, price=0.30),
            PriceSnapshot(timestamp=200, price=0.35),
            PriceSnapshot(timestamp=300, price=0.40),
        ]
        assert fetcher._price_at_time(snaps, 200) == 0.35

    def test_price_between_snapshots(self):
        """When queried between snapshots, return the most recent prior."""
        fetcher = PriceHistoryFetcher(cache_dir=tempfile.mkdtemp())
        snaps = [
            PriceSnapshot(timestamp=100, price=0.30),
            PriceSnapshot(timestamp=300, price=0.40),
        ]
        # At t=200, most recent known price is t=100 -> 0.30
        assert fetcher._price_at_time(snaps, 200) == 0.30

    def test_price_before_first_snapshot(self):
        """Before any data, return None."""
        fetcher = PriceHistoryFetcher(cache_dir=tempfile.mkdtemp())
        snaps = [PriceSnapshot(timestamp=100, price=0.30)]
        assert fetcher._price_at_time(snaps, 50) is None

    def test_empty_snaps(self):
        fetcher = PriceHistoryFetcher(cache_dir=tempfile.mkdtemp())
        assert fetcher._price_at_time([], 100) is None


class TestCacheRoundTrip:
    def test_save_and_load(self):
        """Cache saves to JSON and reloads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PriceHistoryFetcher(cache_dir=tmpdir)
            token_id = "12345678901234567890"
            snaps = [
                PriceSnapshot(timestamp=100, price=0.35),
                PriceSnapshot(timestamp=200, price=0.40),
            ]
            fetcher._save_cache(token_id, snaps)

            loaded = fetcher._load_cache(token_id)
            assert len(loaded) == 2
            assert loaded[0].price == 0.35
            assert loaded[1].timestamp == 200

    def test_load_missing_cache(self):
        """Missing cache file returns empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PriceHistoryFetcher(cache_dir=tmpdir)
            assert fetcher._load_cache("nonexistent") == []


class TestDecisionTimePrice:
    def test_get_decision_time_price(self):
        """Given a target_date and days_out, compute the decision timestamp
        and look up the price at that time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fetcher = PriceHistoryFetcher(cache_dir=tmpdir)
            # Inject cached snapshots directly
            token_id = "test_token_123"
            # Decision time: 5 days before April 10 = April 5 at noon UTC
            # April 5 2026 noon UTC = 1775390400
            april5_noon = 1775390400
            fetcher._cache[token_id] = [
                PriceSnapshot(timestamp=april5_noon - 3600, price=0.25),
                PriceSnapshot(timestamp=april5_noon, price=0.30),
                PriceSnapshot(timestamp=april5_noon + 3600, price=0.32),
            ]
            price = fetcher.get_decision_time_price(
                token_id, target_date=date(2026, 4, 10), days_out=5
            )
            # Decision = April 5 noon. Exact match at t=april5_noon -> 0.30
            assert price == 0.30
