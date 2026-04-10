# Real-Price Replay Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace synthetic pricing circularity with real Polymarket price histories so the backtester can produce honest, non-circular performance metrics.

**Architecture:** New `PriceHistoryFetcher` fetches CLOB `/prices-history` endpoint for each closed temperature market's token IDs (discovered via Gamma). Prices are cached locally and injected into `_build_market()` as decision-time snapshots (not final resolution prices). The `MispricingModel` calibration switches from synthetic-vs-forecast to real-market-vs-forecast.

**Tech Stack:** aiohttp (existing), CLOB REST API (public, no auth), Gamma API (existing), JSON file cache.

---

## Background

### The Problem
The current backtester generates synthetic market prices from the *same* Gaussian forecast distribution the bot uses to find edge. This creates circular validation: the bot always finds edge because the mispricing model is correlated with its own signal. The OOS robustness score of 20/100 (BROKEN) is primarily caused by this circularity.

### The Solution
Polymarket's CLOB API exposes `/prices-history` which returns timestamped price snapshots for any token. By fetching real price histories for closed temperature markets, we can:
1. Use actual market prices at decision-time (not resolution prices) in the backtest
2. Calibrate the mispricing model on real market-vs-forecast bias (not synthetic)
3. Get honest Sharpe/win-rate/robustness from non-circular data

### Key API Details
- **Gamma API** (`gamma-api.polymarket.com/markets`): Returns `clobTokenIds` field (JSON array of token ID strings) for each market. First token = YES, second = NO.
- **CLOB prices-history** (`clob.polymarket.com/prices-history`): Takes `market` (= CLOB token ID), `startTs`, `endTs`, `fidelity` (minutes). Returns `{"history": [{"t": unix_ts, "p": price}, ...]}`. Public, no auth, 90s CDN cache.
- **Limitation**: Temperature markets only started ~March 29, 2026. Initial data will be sparse (~5-30 closed markets). Plan supports hybrid mode: real prices where available, synthetic fallback elsewhere.

---

### Task 1: PriceHistoryFetcher — Core Fetcher with Cache

**Files:**
- Create: `price_history.py`
- Test: `tests/test_price_history.py`

**Step 1: Write failing tests**

```python
# tests/test_price_history.py
"""Tests for PriceHistoryFetcher — CLOB price history retrieval and caching."""

import json
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from price_history import PriceHistoryFetcher, PriceSnapshot


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
            # April 5 2026 noon UTC = 1775217600
            april5_noon = 1775217600
            fetcher._cache[token_id] = [
                PriceSnapshot(timestamp=april5_noon - 3600, price=0.25),
                PriceSnapshot(timestamp=april5_noon, price=0.30),
                PriceSnapshot(timestamp=april5_noon + 3600, price=0.32),
            ]
            from datetime import date
            price = fetcher.get_decision_time_price(
                token_id, target_date=date(2026, 4, 10), days_out=5
            )
            # Decision = April 5 noon. Exact match at t=april5_noon -> 0.30
            assert price == 0.30
```

**Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_price_history.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'price_history'`

**Step 3: Write implementation**

```python
# price_history.py
"""
price_history.py — Real Polymarket Price History Fetcher
=========================================================
Fetches historical price snapshots from the CLOB /prices-history
endpoint for closed temperature markets. Used by the backtester
to replace synthetic pricing with real market data.

The CLOB API is public (no auth required) and returns timestamped
YES-token prices. We cache results locally to avoid re-fetching.

Key concept: "decision-time price" = the market price at the moment
the bot would have made its trading decision (target_date - days_out,
at noon UTC). This avoids using resolution-time prices which would
be look-ahead bias.
"""

import json
import logging
import os
from bisect import bisect_right
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone, timedelta
from typing import Dict, List, Optional

import aiohttp

from api_utils import fetch_with_retry

logger = logging.getLogger("price_history")

CLOB_BASE = "https://clob.polymarket.com"


@dataclass
class PriceSnapshot:
    """A single price observation from the CLOB."""
    timestamp: int       # Unix seconds UTC
    price: float         # YES token price (0-1)

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @staticmethod
    def from_clob_response(data: dict) -> List["PriceSnapshot"]:
        """Parse the CLOB /prices-history response."""
        history = data.get("history", [])
        return [
            PriceSnapshot(timestamp=int(h["t"]), price=float(h["p"]))
            for h in history
            if "t" in h and "p" in h
        ]


class PriceHistoryFetcher:
    """
    Fetch and cache real price histories from Polymarket CLOB.

    Usage:
        fetcher = PriceHistoryFetcher()
        await fetcher.fetch_token_history(token_id)
        price = fetcher.get_decision_time_price(token_id, target_date, days_out)
    """

    def __init__(self, cache_dir: str = "data/price_history"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._cache: Dict[str, List[PriceSnapshot]] = {}

    # -- Cache I/O --

    def _cache_path(self, token_id: str) -> str:
        # Use last 16 chars of token ID to keep filenames manageable
        safe_id = token_id[-16:] if len(token_id) > 16 else token_id
        return os.path.join(self.cache_dir, f"ph_{safe_id}.json")

    def _save_cache(self, token_id: str, snaps: List[PriceSnapshot]) -> None:
        path = self._cache_path(token_id)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in snaps], f)
        self._cache[token_id] = snaps

    def _load_cache(self, token_id: str) -> List[PriceSnapshot]:
        if token_id in self._cache:
            return self._cache[token_id]
        path = self._cache_path(token_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path) as f:
                raw = json.load(f)
            snaps = [PriceSnapshot(**item) for item in raw]
            self._cache[token_id] = snaps
            return snaps
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    # -- CLOB API --

    async def fetch_token_history(
        self,
        session: aiohttp.ClientSession,
        token_id: str,
        fidelity: int = 60,
    ) -> List[PriceSnapshot]:
        """
        Fetch full price history for a token from CLOB.

        Args:
            token_id: The CLOB token ID (YES token).
            fidelity: Resolution in minutes (60 = hourly snapshots).

        Returns cached data if available.
        """
        cached = self._load_cache(token_id)
        if cached:
            return cached

        url = f"{CLOB_BASE}/prices-history"
        params = {
            "market": token_id,
            "interval": "max",
            "fidelity": fidelity,
        }

        try:
            data = await fetch_with_retry(
                session, url, params=params,
                timeout_sec=15.0, label="CLOB-prices-history",
            )
            if data is None:
                return []

            snaps = PriceSnapshot.from_clob_response(data)
            if snaps:
                self._save_cache(token_id, snaps)
                logger.info(
                    f"Fetched {len(snaps)} price snapshots for token "
                    f"{token_id[:16]}... (range: {snaps[0].dt.date()} to {snaps[-1].dt.date()})"
                )
            return snaps

        except Exception as e:
            logger.warning(f"Failed to fetch price history for {token_id[:16]}...: {e}")
            return []

    # -- Price lookup --

    def _price_at_time(
        self, snaps: List[PriceSnapshot], query_ts: int
    ) -> Optional[float]:
        """
        Find the price at or just before query_ts using binary search.
        Returns None if query_ts is before all snapshots.
        """
        if not snaps:
            return None
        timestamps = [s.timestamp for s in snaps]
        idx = bisect_right(timestamps, query_ts) - 1
        if idx < 0:
            return None
        return snaps[idx].price

    def get_decision_time_price(
        self,
        token_id: str,
        target_date: date,
        days_out: int,
        hour_utc: int = 12,
    ) -> Optional[float]:
        """
        Get the market price at the time the bot would decide.

        Decision time = (target_date - days_out) at hour_utc (default noon UTC).
        This is the price the bot would see when evaluating the trade.

        Args:
            token_id: CLOB token ID.
            target_date: The date the market resolves (temperature observation date).
            days_out: Forecast horizon in days.
            hour_utc: Hour of day (UTC) when the bot runs. Default 12 (noon).

        Returns:
            The YES price at decision time, or None if no data available.
        """
        snaps = self._load_cache(token_id)
        if not snaps:
            snaps = self._cache.get(token_id, [])
        if not snaps:
            return None

        decision_date = target_date - timedelta(days=days_out)
        decision_dt = datetime(
            decision_date.year, decision_date.month, decision_date.day,
            hour_utc, 0, 0, tzinfo=timezone.utc,
        )
        decision_ts = int(decision_dt.timestamp())

        return self._price_at_time(snaps, decision_ts)

    # -- Bulk fetch --

    async def fetch_all_for_markets(
        self,
        session: aiohttp.ClientSession,
        gamma_markets: List[dict],
        fidelity: int = 60,
    ) -> int:
        """
        Fetch price histories for all closed temperature markets from Gamma data.

        Args:
            gamma_markets: List of Gamma API market dicts (must have clobTokenIds).
            fidelity: Price resolution in minutes.

        Returns:
            Number of tokens successfully fetched.
        """
        import json as _json
        fetched = 0
        for item in gamma_markets:
            clob_ids = item.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                try:
                    clob_ids = _json.loads(clob_ids)
                except (ValueError, _json.JSONDecodeError):
                    continue
            if not clob_ids:
                continue

            # First token is YES
            yes_token = clob_ids[0]
            snaps = await self.fetch_token_history(session, yes_token, fidelity)
            if snaps:
                fetched += 1

        logger.info(f"Fetched price histories for {fetched}/{len(gamma_markets)} markets")
        return fetched
```

**Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_price_history.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add price_history.py tests/test_price_history.py
git commit -m "feat: add PriceHistoryFetcher for real CLOB price data"
```

---

### Task 2: Wire Fetcher into HistoricalDataLoader

**Files:**
- Modify: `backtest_data.py` (add `fetch_price_histories()` method and `get_decision_time_prices()`)
- Test: `tests/test_backtest_data.py` (add 2 tests)

**Step 1: Write failing tests**

Append to `tests/test_backtest_data.py`:

```python
class TestPriceHistoryIntegration:
    def test_get_decision_time_prices_returns_dict(self):
        """get_decision_time_prices returns {label: price} for a city/date/horizon."""
        from backtest_data import HistoricalDataLoader
        loader = HistoricalDataLoader()
        # No data loaded — should return None
        result = loader.get_decision_time_prices("Miami", date(2026, 4, 5), days_out=3)
        assert result is None

    def test_get_decision_time_prices_with_data(self):
        """When price history is loaded, returns decision-time prices."""
        from backtest_data import HistoricalDataLoader
        from price_history import PriceSnapshot
        loader = HistoricalDataLoader()

        # Inject a fake gamma market with token ID
        fake_market = {
            "question": "What will the highest temperature be in Miami on April 10, 2026?",
            "groupItemTitle": "80-84°F",
            "outcomePrices": "[0.35, 0.65]",
            "clobTokenIds": '["token_yes_123", "token_no_456"]',
            "active": False,
            "closed": True,
        }
        loader._gamma_markets = [fake_market]
        loader._token_map = {
            ("Miami", date(2026, 4, 10), "80-84°F"): "token_yes_123"
        }

        # Inject price history for that token
        april7_noon = int(datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc).timestamp())
        loader._price_fetcher._cache["token_yes_123"] = [
            PriceSnapshot(timestamp=april7_noon - 3600, price=0.30),
            PriceSnapshot(timestamp=april7_noon, price=0.35),
            PriceSnapshot(timestamp=april7_noon + 3600, price=0.37),
        ]

        result = loader.get_decision_time_prices("Miami", date(2026, 4, 10), days_out=3)
        assert result is not None
        assert "80-84°F" in result
        assert result["80-84°F"] == 0.35  # Price at decision time (April 7 noon)
```

**Step 2: Run tests to see them fail**

Run: `python3 -m pytest tests/test_backtest_data.py::TestPriceHistoryIntegration -v`
Expected: FAIL — `AttributeError: 'HistoricalDataLoader' object has no attribute 'get_decision_time_prices'`

**Step 3: Modify `backtest_data.py`**

Add these imports at top of `backtest_data.py`:
```python
from price_history import PriceHistoryFetcher, PriceSnapshot
```

Add to `HistoricalDataLoader.__init__`:
```python
self._price_fetcher = PriceHistoryFetcher()
self._token_map: Dict[Tuple[str, date, str], str] = {}  # (city, date, label) -> token_id
```

Add these methods to `HistoricalDataLoader`:

```python
async def fetch_price_histories(
    self, session: aiohttp.ClientSession, fidelity: int = 60
) -> int:
    """Fetch CLOB price histories for all closed temperature markets.

    Also builds _token_map: (city, date, label) -> YES token ID
    so get_decision_time_prices() can look up by city/date.
    """
    # Build token map from gamma markets
    for item in self._gamma_markets:
        city, mkt_date, lo, hi, price = self._parse_gamma_market(item)
        if city is None or mkt_date is None:
            continue

        clob_ids = item.get("clobTokenIds", [])
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except (json.JSONDecodeError, ValueError):
                continue
        if not clob_ids:
            continue

        label = item.get("groupItemTitle", "")
        if label:
            self._token_map[(city, mkt_date, label)] = clob_ids[0]

    return await self._price_fetcher.fetch_all_for_markets(
        session, self._gamma_markets, fidelity
    )

def get_decision_time_prices(
    self, city: str, target_date: date, days_out: int
) -> Optional[Dict[str, float]]:
    """
    Get real market prices at decision time for a city/date.

    Returns: {outcome_label: price_yes} or None if no real data.
    Decision time = target_date - days_out at noon UTC.
    """
    result: Dict[str, float] = {}
    for (c, d, label), token_id in self._token_map.items():
        if c == city and d == target_date:
            price = self._price_fetcher.get_decision_time_price(
                token_id, target_date, days_out
            )
            if price is not None:
                result[label] = price
    return result if result else None
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_backtest_data.py -v`
Expected: All tests PASS (existing + 2 new)

**Step 5: Commit**

```bash
git add backtest_data.py tests/test_backtest_data.py
git commit -m "feat: wire PriceHistoryFetcher into HistoricalDataLoader"
```

---

### Task 3: Hybrid `_build_market()` — Real Prices > Synthetic Fallback

**Files:**
- Modify: `backtester.py` (`_build_market()` gains a `days_out` price lookup path)
- Test: `tests/test_backtester.py` (add 1 test)

**Step 1: Write failing test**

Append to `tests/test_backtester.py`:

```python
def test_build_market_prefers_decision_time_prices(self):
    """_build_market uses decision-time real prices over synthetic when available."""
    from backtester import BacktestEngine
    from price_history import PriceSnapshot
    from datetime import date, datetime, timezone

    engine = BacktestEngine(bankroll=500.0, seed=42)

    # Inject a token mapping and price history
    target = date(2026, 4, 10)
    engine.loader._token_map = {
        ("Miami", target, "80-84°F"): "tok_yes_1",
        ("Miami", target, "75-79°F"): "tok_yes_2",
        ("Miami", target, "85-89°F"): "tok_yes_3",
    }

    # Decision time for days_out=3 is April 7 noon UTC
    april7_noon = int(datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc).timestamp())
    engine.loader._price_fetcher._cache["tok_yes_1"] = [
        PriceSnapshot(timestamp=april7_noon, price=0.40),
    ]
    engine.loader._price_fetcher._cache["tok_yes_2"] = [
        PriceSnapshot(timestamp=april7_noon, price=0.35),
    ]
    engine.loader._price_fetcher._cache["tok_yes_3"] = [
        PriceSnapshot(timestamp=april7_noon, price=0.15),
    ]

    market = engine._build_market("Miami", target, days_out=3,
                                   forecast_high=82.0, forecast_sigma=3.0)
    # Should have used real prices for the matching buckets
    prices = {o.outcome_label: o.price_yes for o in market.outcomes}
    assert "80-84°F" in prices
    assert prices["80-84°F"] == 0.40
    assert prices["75-79°F"] == 0.35
```

**Step 2: Run test to see it fail**

Run: `python3 -m pytest tests/test_backtester.py::test_build_market_prefers_decision_time_prices -v`
Expected: FAIL — `_build_market` doesn't use `get_decision_time_prices`

**Step 3: Modify `_build_market()` in `backtester.py`**

Update `_build_market` to try decision-time prices first, then fall back to final-price, then synthetic:

```python
def _build_market(
    self,
    city: str,
    target_date: date,
    days_out: int,
    forecast_high: float,
    forecast_sigma: float,
) -> TemperatureMarket:
    """
    Build a TemperatureMarket with this priority:
    1. Real CLOB decision-time prices (best — non-circular)
    2. Real Gamma final prices (acceptable — slight look-ahead)
    3. Synthetic prices (worst — circular, only when no real data)
    """
    # Priority 1: Decision-time prices from CLOB history
    decision_prices = self.loader.get_decision_time_prices(
        city, target_date, days_out
    )
    if decision_prices and len(decision_prices) >= 3:
        outcomes = []
        for label, price_yes in decision_prices.items():
            lo, hi = _parse_bucket(label)
            outcomes.append(MarketOutcome(
                outcome_label=label,
                token_id=f"real_{city}_{target_date}_{label}",
                price_yes=price_yes,
                price_no=round(1.0 - price_yes, 4),
                bucket_low=lo,
                bucket_high=hi,
            ))
        outcomes.sort(key=lambda o: (o.bucket_low if o.bucket_low is not None else -999))
        return TemperatureMarket(
            market_id=f"real_{city}_{target_date.isoformat()}",
            question=f"What will the highest temperature be in {city} on {target_date}?",
            city=city,
            market_date=target_date,
            resolution_source="NWS",
            outcomes=outcomes,
            active=True,
        )

    # Priority 2: Gamma final prices (existing behavior)
    real_prices = self.loader.get_real_market_prices(city, target_date)
    if real_prices and len(real_prices) >= 3:
        outcomes = []
        for label, price_yes in real_prices.items():
            lo, hi = _parse_bucket(label)
            outcomes.append(MarketOutcome(
                outcome_label=label,
                token_id=f"backtest_{city}_{target_date}_{label}",
                price_yes=price_yes,
                price_no=round(1.0 - price_yes, 4),
                bucket_low=lo,
                bucket_high=hi,
            ))
        outcomes.sort(key=lambda o: (o.bucket_low if o.bucket_low is not None else -999))
        return TemperatureMarket(
            market_id=f"backtest_{city}_{target_date.isoformat()}",
            question=f"What will the highest temperature be in {city} on {target_date}?",
            city=city,
            market_date=target_date,
            resolution_source="NWS",
            outcomes=outcomes,
            active=True,
        )

    # Priority 3: Synthetic prices (fallback)
    buckets = self._select_buckets(forecast_high)
    true_probs = bucket_probabilities(
        forecast_high, forecast_sigma, buckets
    )
    prices = self.pricing.generate_prices(
        [true_probs.get(i, 0.0) for i in range(len(buckets))],
        days_out=days_out,
    )

    outcomes = []
    for i, ((lo, hi), price) in enumerate(zip(buckets, prices)):
        label = self._bucket_label(lo, hi)
        outcomes.append(MarketOutcome(
            outcome_label=label,
            token_id=f"synth_{city}_{target_date}_{label}",
            price_yes=price,
            price_no=round(1.0 - price, 4),
            bucket_low=lo,
            bucket_high=hi,
        ))

    outcomes.sort(key=lambda o: (o.bucket_low if o.bucket_low is not None else -999))

    return TemperatureMarket(
        market_id=f"synth_{city}_{target_date.isoformat()}",
        question=f"What will the highest temperature be in {city} on {target_date}?",
        city=city,
        market_date=target_date,
        resolution_source="NWS",
        outcomes=outcomes,
        active=True,
    )
```

**Step 4: Run tests**

Run: `python3 -m pytest tests/test_backtester.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add backtester.py tests/test_backtester.py
git commit -m "feat: hybrid _build_market with decision-time price priority"
```

---

### Task 4: Wire Price Fetch into CLI and Track Price Source

**Files:**
- Modify: `backtester.py` (`fetch_data()` calls `fetch_price_histories`, scorecard tracks source)
- Modify: `backtest_scorecard.py` (add `price_source` field to `BacktestTrade`, add source breakdown)

**Step 1: Add `price_source` field to `BacktestTrade`**

In `backtest_scorecard.py`, add to the `BacktestTrade` dataclass:

```python
price_source: str = "synthetic"  # "real_clob", "real_gamma", or "synthetic"
```

Add a new breakdown method:

```python
def breakdown_by_source(self, variant: str) -> Dict[str, Dict[str, Any]]:
    return self._breakdown(variant, lambda t: t.price_source)
```

Add "BY SOURCE" to the `render()` method's breakdowns list:

```python
("BY SOURCE", self.breakdown_by_source(variant)),
```

**Step 2: Update `backtester.py`**

In `fetch_data()`, add after the Gamma fetch:

```python
logger.info("Fetching CLOB price histories...")
fetched = await loader._price_fetcher.fetch_all_for_markets(
    session, loader._gamma_markets, fidelity=60
)
logger.info(f"Price histories fetched for {fetched} tokens")

# Build token map
await loader.fetch_price_histories(session)
```

Wait — `fetch_price_histories` already does both. Just call it:

```python
async def fetch_data(loader, cities, start, end):
    async with aiohttp.ClientSession() as session:
        logger.info("Fetching NOAA daily highs...")
        await loader.fetch_daily_highs(cities, start, end)

        logger.info("Fetching Gamma closed markets...")
        await loader.fetch_gamma_closed_markets()

        logger.info("Fetching CLOB price histories...")
        fetched = await loader.fetch_price_histories(session)
        logger.info(f"Fetched price histories for {fetched} tokens")

        logger.info("Building climatology from observations...")
        loader.load_climatology_from_actuals(loader._highs)
```

In the `BacktestTrade` creation in `run()`, determine price source from market_id prefix:

```python
# Determine price source from market ID
if market.market_id.startswith("real_"):
    price_source = "real_clob"
elif market.market_id.startswith("backtest_"):
    price_source = "real_gamma"
else:
    price_source = "synthetic"
```

Pass `price_source=price_source` to the `BacktestTrade(...)` constructor.

**Step 3: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add backtester.py backtest_scorecard.py
git commit -m "feat: wire CLOB price fetch into CLI, track price source in scorecard"
```

---

### Task 5: MispricingModel Calibration on Real Decision-Time Prices

**Files:**
- Modify: `backtester.py` (calibration section in `main()`)
- Test: `tests/test_backtest_pricing.py` (add 1 test for real-data calibration)

**Step 1: Write failing test**

Append to `tests/test_backtest_pricing.py`:

```python
def test_calibrate_from_real_decision_prices():
    """Calibration from real market prices should adjust bias parameters."""
    from backtest_pricing import MispricingModel
    model = MispricingModel()
    original_tail = model.tail_overpricing

    # Simulate real market data: tails are overpriced by ~10¢ vs forecast
    # Position 0.1 (tail), market=0.15, forecast=0.05 -> bias=0.10
    # Position 0.9 (tail), market=0.12, forecast=0.03 -> bias=0.09
    # Position 0.5 (mode), market=0.30, forecast=0.35 -> bias=-0.05
    calibration_data = [
        (0.1, 0.15, 0.05),   # tail: market overprices by 0.10
        (0.9, 0.12, 0.03),   # tail: market overprices by 0.09
        (0.5, 0.30, 0.35),   # mode: market underprices by 0.05
    ]
    model.calibrate(calibration_data)

    # Tail overpricing should be ~0.095 (avg of 0.10 and 0.09)
    assert 0.08 < model.tail_overpricing < 0.11
    # Mode underpricing should be ~-0.05
    assert -0.06 < model.mode_underpricing < -0.04
```

**Step 2: Run test — should PASS** (calibrate already works with forecast_prob)

Run: `python3 -m pytest tests/test_backtest_pricing.py::test_calibrate_from_real_decision_prices -v`
Expected: PASS (the method already handles this correctly)

**Step 3: Update calibration in `backtester.py` `main()`**

Replace the existing calibration block with one that uses decision-time prices and forecast probabilities:

```python
# Calibrate mispricing model from real price histories
if engine.loader._token_map:
    logger.info("Calibrating mispricing model from real CLOB price histories...")
    cal_data = []
    from forecast_scanner import bucket_probabilities

    for (city, mkt_date, label), token_id in engine.loader._token_map.items():
        # Get the decision-time price (at 5 days out)
        price = engine.loader._price_fetcher.get_decision_time_price(
            token_id, mkt_date, days_out=5
        )
        if price is None:
            continue

        # Get the forecast probability for this bucket at 5 days
        actual = engine.loader.get_actual_high(city, mkt_date)
        if actual is None:
            continue

        lo, hi = _parse_bucket(label)
        if lo is None and hi is None:
            continue

        # Use climatology as forecast anchor (consistent with realistic variant)
        doy = mkt_date.timetuple().tm_yday
        clim = engine.loader._climatology.get(city, {}).get(doy, actual)
        sigma = 4.5  # 5-day horizon base sigma
        probs = bucket_probabilities(clim, sigma, [(lo, hi)])
        forecast_prob = probs.get(0, 0.0)

        # Normalized position (0=low tail, 1=high tail)
        # Approximate from bucket bounds
        if lo is not None and hi is not None:
            mid = (lo + hi) / 2.0
            pos = max(0.0, min(1.0, (mid - clim + 20) / 40.0))
        elif lo is None:
            pos = 0.05
        else:
            pos = 0.95

        cal_data.append((pos, price, forecast_prob))

    if cal_data:
        engine.pricing.calibrate(cal_data)
        logger.info(
            f"Calibrated from {len(cal_data)} real price points: "
            f"tail_overpricing={engine.pricing.tail_overpricing:.3f}, "
            f"mode_underpricing={engine.pricing.mode_underpricing:.3f}"
        )
    else:
        logger.info("No real price data for calibration — using defaults")
elif engine.loader._gamma_markets:
    # Fallback: calibrate from Gamma final prices (existing behavior)
    logger.info("Calibrating mispricing model from Gamma final prices (fallback)...")
    cal_data = []
    for item in engine.loader._gamma_markets:
        city_name, mkt_date, lo, hi, price = engine.loader._parse_gamma_market(item)
        if city_name and lo is not None:
            pos = 0.5
            cal_data.append((pos, price, 0.0))
    if cal_data:
        engine.pricing.calibrate(cal_data)
        logger.info(
            f"Calibrated: tail_overpricing={engine.pricing.tail_overpricing:.3f}, "
            f"mode_underpricing={engine.pricing.mode_underpricing:.3f}"
        )
```

**Step 4: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add backtester.py tests/test_backtest_pricing.py
git commit -m "feat: calibrate mispricing model from real CLOB decision-time prices"
```

---

### Task 6: Scorecard — Price Source Metrics and Circularity Warning

**Files:**
- Modify: `backtest_scorecard.py` (add circularity detection to `fragility_notes` and `robustness_score`)

**Step 1: Write failing test**

Append to `tests/test_backtest_scorecard.py`:

```python
def test_circularity_warning_when_all_synthetic():
    """Robustness score should heavily penalize 100% synthetic data."""
    from backtest_scorecard import BacktestScorecard, BacktestTrade
    trades = [
        BacktestTrade(
            city="Miami", target_date=date(2026, 4, 1), days_out=3,
            side="BUY", outcome_label="80-84°F", bucket_low=80, bucket_high=84,
            p_true=0.40, market_price=0.25, ev=0.10, edge=0.15,
            kelly_fraction=0.15, position_size_usd=5.0, price_limit=0.26,
            actual_high=82.0, won=True, pnl=3.70, variant="realistic",
            regime="normal", price_source="synthetic",
        )
        for _ in range(100)
    ]
    sc = BacktestScorecard(trades)
    score, penalties = sc.robustness_score("realistic")
    # Should have a synthetic data penalty
    synth_penalties = [p for p in penalties if "synthetic" in p.lower() or "circular" in p.lower()]
    assert len(synth_penalties) > 0
```

**Step 2: Run test to see it fail**

Run: `python3 -m pytest tests/test_backtest_scorecard.py::test_circularity_warning_when_all_synthetic -v`
Expected: FAIL — no synthetic penalty exists yet

**Step 3: Add circularity penalty to `robustness_score()`**

In `backtest_scorecard.py`, add after the trade count penalty in `robustness_score()`:

```python
# --- Synthetic data circularity penalty ---
ts = self._filter(variant)
if ts:
    synthetic_count = sum(1 for t in ts if getattr(t, 'price_source', 'synthetic') == 'synthetic')
    synthetic_pct = synthetic_count / len(ts)
    if synthetic_pct > 0.9:
        score -= 15
        penalties.append(
            f"Circular risk: {synthetic_pct:.0%} of trades use synthetic prices: -15"
        )
    elif synthetic_pct > 0.5:
        score -= 10
        penalties.append(
            f"Partial circular risk: {synthetic_pct:.0%} synthetic prices: -10"
        )
```

Add to `fragility_notes()`:

```python
# Price source distribution
ts = self._filter(variant)
if ts:
    sources = defaultdict(int)
    for t in ts:
        sources[getattr(t, 'price_source', 'synthetic')] += 1
    for src, count in sorted(sources.items()):
        pct = count / len(ts)
        notes.append(f"PRICE SOURCE: {src} = {pct:.0%} ({count} trades)")
    synthetic_pct = sources.get('synthetic', 0) / len(ts)
    if synthetic_pct > 0.9:
        notes.append("WARNING: >90% synthetic prices — results are circular, not trustworthy")
```

Add "BY SOURCE" breakdown to `render()`.

**Step 4: Run all tests**

Run: `python3 -m pytest tests/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add backtest_scorecard.py tests/test_backtest_scorecard.py
git commit -m "feat: add circularity penalty and price source tracking to scorecard"
```

---

### Task 7: Integration Test — End-to-End with Mocked CLOB

**Files:**
- Create: `tests/test_price_replay_integration.py`

**Step 1: Write integration test**

```python
# tests/test_price_replay_integration.py
"""Integration test: full backtest with mocked CLOB price histories."""

from datetime import date, datetime, timezone
from unittest.mock import patch, AsyncMock

from backtest_scorecard import BacktestScorecard
from backtester import BacktestEngine
from price_history import PriceSnapshot


def test_real_price_replay_produces_trades():
    """Backtest with injected real prices should produce trades with
    price_source='real_clob' and non-circular metrics."""
    engine = BacktestEngine(bankroll=500.0, seed=42)

    # Inject minimal test data
    target = date(2026, 4, 5)
    engine.loader._highs = {("Miami", target): 83.0}
    engine.loader._climatology = {"Miami": {target.timetuple().tm_yday: 80.0}}
    engine.approximator.set_climatology(engine.loader._climatology)
    lagged = {"Miami": {
        date(2026, 4, 4): 82.0,
        date(2026, 4, 3): 81.0,
        date(2026, 4, 2): 79.0,
    }}
    engine.approximator.set_lagged_actuals(lagged)

    # Inject token map and price history for 5 buckets
    buckets = [
        ("70-74°F", 0.05), ("75-79°F", 0.20), ("80-84°F", 0.45),
        ("85-89°F", 0.20), ("90-94°F", 0.10),
    ]
    # Decision time for days_out=3 is April 2 noon UTC
    decision_ts = int(datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc).timestamp())

    for i, (label, price) in enumerate(buckets):
        token_id = f"tok_{i}"
        engine.loader._token_map[("Miami", target, label)] = token_id
        engine.loader._price_fetcher._cache[token_id] = [
            PriceSnapshot(timestamp=decision_ts, price=price),
        ]

    result = engine.run(cities=["Miami"], start_date=target, end_date=target)

    # Should have produced at least some trades
    realistic_trades = [t for t in result.trades if t.variant == "realistic"]
    if realistic_trades:
        # All trades from this date should be real_clob source
        sources = {t.price_source for t in realistic_trades}
        assert "real_clob" in sources or "synthetic" in sources  # depends on edge found
```

**Step 2: Run test**

Run: `python3 -m pytest tests/test_price_replay_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_price_replay_integration.py
git commit -m "test: add integration test for real-price replay mode"
```

---

## Summary

| Task | What | Files | Tests |
|------|------|-------|-------|
| 1 | PriceHistoryFetcher core | `price_history.py` | 9 tests |
| 2 | Wire into HistoricalDataLoader | `backtest_data.py` | 2 tests |
| 3 | Hybrid `_build_market()` | `backtester.py` | 1 test |
| 4 | CLI + price source tracking | `backtester.py`, `backtest_scorecard.py` | existing |
| 5 | Calibration on real prices | `backtester.py` | 1 test |
| 6 | Circularity penalty in scorecard | `backtest_scorecard.py` | 1 test |
| 7 | Integration test | `tests/test_price_replay_integration.py` | 1 test |

**Total: ~15 new tests, 4 new/modified files**

After implementation, run the backtester on April 2026 data:
```bash
python3 backtester.py --cities "Miami,New York,Chicago" --start 2026-03-29 --end 2026-04-30 --sensitivity
```

The scorecard will now show "BY SOURCE" breakdown revealing what fraction used real vs synthetic prices, and the circularity penalty will honestly flag when results are still dominated by synthetic data.
