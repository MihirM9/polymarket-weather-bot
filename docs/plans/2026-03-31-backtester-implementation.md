# Backtester Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a historical backtester that replays 12-18 months of temperature data through the exact same DecisionEngine.evaluate() code path the live bot uses, producing a scorecard with Sharpe, win rate, drawdown, breakdowns, and parameter sensitivity analysis.

**Architecture:** Six new files, zero modifications to existing production code. The backtester constructs synthetic CityForecast and TemperatureMarket objects from historical NOAA data + calibrated mispricing model, feeds them to the real DecisionEngine, scores results against actual temperatures, and outputs a terminal scorecard + CSV. A MockTracker enforces the same risk controls as live.

**Tech Stack:** Python 3.10+, aiohttp (NOAA API), math.erf (Gaussian CDF — no scipy), csv/json for caching, existing DecisionEngine/ResolutionTracker imports.

---

### Task 1: MockTracker (`backtest_tracker.py`)

**Files:**
- Create: `backtest_tracker.py`
- Test: `tests/test_backtest_tracker.py`

This is the simplest component — a lightweight PositionTracker substitute that enforces risk caps without disk I/O or CLOB polling.

**Step 1: Write the failing test**

```python
# tests/test_backtest_tracker.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

def test_mock_tracker_enforces_per_market_cap():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    # Per-market cap = 3% of 500 = $15
    assert t.can_trade("mkt1", "bucket_a", 10.0, "New York") is True
    t.record_trade("mkt1", "bucket_a", 10.0, "New York")
    assert t.can_trade("mkt1", "bucket_b", 10.0, "New York") is False  # would exceed $15

def test_mock_tracker_dedup():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    t.record_trade("mkt1", "72-73°F", 5.0, "Miami")
    assert t.has_active_order("mkt1", "72-73°F") is True
    assert t.has_active_order("mkt1", "74-75°F") is False

def test_mock_tracker_daily_reset():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    t.record_trade("mkt1", "72-73°F", 10.0, "Miami")
    t.reset_day()
    assert t.has_active_order("mkt1", "72-73°F") is False
    assert t.daily_exposure == 0.0

def test_mock_tracker_correlated_group_cap():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    # NYC and Chicago are correlated (northeast group)
    # Group cap = 3% * 500 * 1.5 = $22.50
    t.record_trade("mkt1", "a", 10.0, "New York")
    t.record_trade("mkt2", "b", 10.0, "Chicago")
    # $20 used in northeast group, $2.50 left
    assert t.can_trade("mkt3", "c", 5.0, "New York") is False

def test_mock_tracker_daily_cap():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    # Daily cap = 30% of 500 = $150
    for i in range(15):
        t.record_trade(f"mkt{i}", f"b{i}", 10.0, "Miami")
    assert t.can_trade("mkt99", "x", 5.0, "Houston") is False  # $150 used

def test_mock_tracker_exposure_properties():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    t.record_trade("mkt1", "a", 8.0, "New York")
    t.record_trade("mkt2", "b", 5.0, "Miami")
    assert t.total_exposure == 13.0
    assert t.daily_exposure == 13.0
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_tracker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_tracker'`

**Step 3: Write minimal implementation**

```python
# backtest_tracker.py
"""
backtest_tracker.py — Lightweight PositionTracker for backtesting
=================================================================
Enforces the same risk controls as the live PositionTracker without
disk I/O, CLOB polling, or order lifecycle management. Tracks exposure
per city, per correlation group, and per day.
"""

from typing import Dict, Set

CORRELATION_GROUPS = {
    "northeast": ["New York", "Chicago"],
    "gulf": ["Houston", "Miami"],
    "south_central": ["Dallas", "Houston"],
    "west": ["Los Angeles"],
}
CORRELATED_GROUP_CAP_MULT = 1.5
PER_MARKET_MAX_PCT = 0.03
DAILY_EXPOSURE_CAP_PCT = 0.30
MAX_POSITION_USD = 10.0


class MockTracker:
    """
    Simulates PositionTracker risk controls for backtesting.

    Enforces:
    - Per-market cap (3% of bankroll)
    - Absolute position cap ($10)
    - Correlated exposure caps (NYC+Chicago share 1.5x cap)
    - Daily exposure cap (30% of bankroll)
    - Deduplication via has_active_order()
    """

    def __init__(self, bankroll: float = 500.0):
        self.bankroll = bankroll
        self._per_market_cap = PER_MARKET_MAX_PCT * bankroll
        self._daily_cap = DAILY_EXPOSURE_CAP_PCT * bankroll
        self._city_exposure: Dict[str, float] = {}
        self._group_exposure: Dict[str, float] = {}
        self._active_orders: Set[str] = set()  # "market_id:outcome_label"
        self._daily_exposure: float = 0.0

    def can_trade(self, market_id: str, outcome_label: str, size: float, city: str) -> bool:
        """Check if a trade is allowed under all risk caps."""
        key = f"{market_id}:{outcome_label}"
        if key in self._active_orders:
            return False

        if size > MAX_POSITION_USD:
            return False

        # Per-market (city) cap
        city_exp = self._city_exposure.get(city, 0.0)
        if city_exp + size > self._per_market_cap:
            return False

        # Correlated group cap
        group_cap = self._per_market_cap * CORRELATED_GROUP_CAP_MULT
        for group_name, group_cities in CORRELATION_GROUPS.items():
            if city in group_cities:
                group_exp = self._group_exposure.get(group_name, 0.0)
                if group_exp + size > group_cap:
                    return False

        # Daily cap
        if self._daily_exposure + size > self._daily_cap:
            return False

        return True

    def record_trade(self, market_id: str, outcome_label: str, size: float, city: str):
        """Record a trade, updating all exposure counters."""
        key = f"{market_id}:{outcome_label}"
        self._active_orders.add(key)
        self._city_exposure[city] = self._city_exposure.get(city, 0.0) + size
        self._daily_exposure += size

        for group_name, group_cities in CORRELATION_GROUPS.items():
            if city in group_cities:
                self._group_exposure[group_name] = self._group_exposure.get(group_name, 0.0) + size

    def has_active_order(self, market_id: str, outcome_label: str) -> bool:
        return f"{market_id}:{outcome_label}" in self._active_orders

    def is_cooled_down(self, market_id: str, outcome_label: str) -> bool:
        return False  # no cooldowns in backtest

    def reset_day(self):
        """Reset for a new trading day."""
        self._city_exposure.clear()
        self._group_exposure.clear()
        self._active_orders.clear()
        self._daily_exposure = 0.0

    @property
    def total_exposure(self) -> float:
        return self._daily_exposure

    @property
    def daily_exposure(self) -> float:
        return self._daily_exposure

    @property
    def realized_exposure(self) -> float:
        return self._daily_exposure

    @property
    def pending_exposure(self) -> float:
        return 0.0
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_tracker.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add backtest_tracker.py tests/test_backtest_tracker.py
git commit -m "feat(backtest): add MockTracker with risk cap enforcement"
```

---

### Task 2: Historical Data Loader (`backtest_data.py`)

**Files:**
- Create: `backtest_data.py`
- Create: `data/` directory (for caches)
- Test: `tests/test_backtest_data.py`

Fetches and caches NOAA daily highs, climate normals, and Gamma closed markets.

**Step 1: Write the failing test**

```python
# tests/test_backtest_data.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date
import json
import csv
from pathlib import Path

def test_parse_noaa_observation_response():
    """Test parsing of NWS observations API response format."""
    from backtest_data import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir="/tmp/backtest_test_data")

    # Simulated NWS response
    mock_response = {
        "features": [
            {"properties": {"temperature": {"value": 28.3}}},  # 82.9°F
            {"properties": {"temperature": {"value": 30.0}}},  # 86.0°F (max)
            {"properties": {"temperature": {"value": 27.1}}},  # 80.8°F
            {"properties": {"temperature": {"value": None}}},   # missing
        ]
    }
    result = loader._extract_max_temp(mock_response)
    assert result is not None
    assert abs(result - 86.0) < 0.1  # 30°C = 86°F

def test_parse_noaa_empty_response():
    from backtest_data import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir="/tmp/backtest_test_data")
    assert loader._extract_max_temp({"features": []}) is None
    assert loader._extract_max_temp({}) is None

def test_cache_write_and_read(tmp_path):
    from backtest_data import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir=str(tmp_path))

    # Write cache
    loader._cache_daily_high("Miami", date(2025, 7, 15), 91.2)
    loader._cache_daily_high("Miami", date(2025, 7, 16), 89.5)

    # Read cache
    cached = loader._load_cached_highs()
    assert cached[("Miami", date(2025, 7, 15))] == 91.2
    assert cached[("Miami", date(2025, 7, 16))] == 89.5

def test_climatology_lookup():
    from backtest_data import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir="/tmp/backtest_test_data")

    # Manually set climatology for test
    loader._climatology = {
        "Miami": {1: 76.5, 196: 91.3},   # Jan 1, Jul 15
        "New York": {1: 38.2, 196: 84.1},
    }
    assert loader.get_climatology("Miami", date(2025, 7, 15)) == 91.3  # day 196
    assert loader.get_climatology("Miami", date(2025, 1, 1)) == 76.5
    assert loader.get_climatology("Dallas", date(2025, 1, 1)) is None  # no data

def test_gamma_market_parsing():
    """Test parsing of closed Gamma API temperature market."""
    from backtest_data import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir="/tmp/backtest_test_data")

    mock_market = {
        "id": "12345",
        "question": "What will the highest temperature be in Miami on July 15?",
        "groupItemTitle": "82-83°F",
        "outcomePrices": "[0.35, 0.65]",
        "clobTokenIds": "[\"token_abc\", \"token_def\"]",
        "negRiskMarketID": "group_xyz",
        "active": False,
        "closed": True,
        "volume": "5000",
    }
    city, mkt_date, bucket_lo, bucket_hi, price_yes = loader._parse_gamma_market(mock_market)
    assert city == "Miami"
    assert mkt_date == date(2025, 7, 15)  # current year assumed
    assert bucket_lo == 82.0
    assert bucket_hi == 84.0  # 82-83 means [82, 84)
    assert abs(price_yes - 0.35) < 0.01
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_data'`

**Step 3: Write minimal implementation**

```python
# backtest_data.py
"""
backtest_data.py — Historical Data Loader for Backtesting
==========================================================
Fetches and caches:
  1. NOAA daily observed highs (api.weather.gov station observations)
  2. Climate normals (30-year daily averages per station, NCEI)
  3. Gamma API closed temperature markets (real Polymarket prices)

All data is cached to disk (data/ directory) so API calls happen once.
"""

import asyncio
import csv
import json
import logging
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

from api_utils import fetch_with_retry
from config import cfg
from polymarket_parser import _parse_bucket, _match_city, _extract_date, _detect_unit

logger = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "(polymarket-weather-bot-backtest, contact@example.com)",
    "Accept": "application/geo+json",
}
GAMMA_BASE = "https://gamma-api.polymarket.com"


class HistoricalDataLoader:
    """Loads and caches historical weather and market data for backtesting."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self._highs_cache_file = self.data_dir / "noaa_daily_highs.csv"
        self._climatology_file = self.data_dir / "climate_normals.csv"
        self._gamma_cache_file = self.data_dir / "gamma_closed_markets.json"

        self._highs: Dict[Tuple[str, date], float] = {}
        self._climatology: Dict[str, Dict[int, float]] = {}  # city -> day_of_year -> normal_high
        self._gamma_markets: List[dict] = []

    # ── NOAA Daily Highs ──────────────────────────────────────────────

    def _extract_max_temp(self, response: dict) -> Optional[float]:
        """Extract maximum temperature (°F) from NWS observations response."""
        features = response.get("features", [])
        if not features:
            return None

        max_temp_f = None
        for obs in features:
            temp_c = obs.get("properties", {}).get("temperature", {}).get("value")
            if temp_c is not None:
                temp_f = temp_c * 9.0 / 5.0 + 32.0
                if max_temp_f is None or temp_f > max_temp_f:
                    max_temp_f = temp_f
        return max_temp_f

    def _cache_daily_high(self, city: str, target_date: date, high_f: float):
        """Append a daily high to the CSV cache."""
        self._highs[(city, target_date)] = high_f
        write_header = not self._highs_cache_file.exists()
        with open(self._highs_cache_file, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["city", "date", "high_f"])
            w.writerow([city, target_date.isoformat(), f"{high_f:.1f}"])

    def _load_cached_highs(self) -> Dict[Tuple[str, date], float]:
        """Load all cached daily highs from CSV."""
        result: Dict[Tuple[str, date], float] = {}
        if not self._highs_cache_file.exists():
            return result
        with open(self._highs_cache_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    key = (row["city"], date.fromisoformat(row["date"]))
                    result[key] = float(row["high_f"])
                except (KeyError, ValueError):
                    continue
        return result

    async def fetch_daily_highs(
        self,
        cities: List[str],
        start_date: date,
        end_date: date,
    ) -> Dict[Tuple[str, date], float]:
        """
        Fetch NOAA observed daily highs for all cities in date range.
        Uses cache — only fetches missing dates from API.
        """
        self._highs = self._load_cached_highs()
        missing: List[Tuple[str, date]] = []

        current = start_date
        while current <= end_date:
            for city in cities:
                if (city, current) not in self._highs:
                    missing.append((city, current))
            current += timedelta(days=1)

        if not missing:
            logger.info(f"All {len(self._highs)} daily highs loaded from cache")
            return self._highs

        logger.info(f"Fetching {len(missing)} missing daily highs from NWS...")

        async with aiohttp.ClientSession() as session:
            # Group by city to minimize API calls (fetch 7-day windows)
            by_city: Dict[str, List[date]] = {}
            for city, d in missing:
                by_city.setdefault(city, []).append(d)

            for city, dates in by_city.items():
                station_id = cfg.noaa_stations.get(city)
                if not station_id:
                    logger.warning(f"No NOAA station for {city}, skipping")
                    continue

                dates.sort()
                # Fetch in 7-day windows
                window_start = dates[0]
                while window_start <= dates[-1]:
                    window_end = window_start + timedelta(days=7)
                    offset_hours = cfg.CITY_UTC_OFFSETS.get(city, -5)

                    start_utc = datetime(
                        window_start.year, window_start.month, window_start.day,
                        tzinfo=timezone.utc
                    ) - timedelta(hours=offset_hours)
                    end_utc = datetime(
                        window_end.year, window_end.month, window_end.day,
                        tzinfo=timezone.utc
                    ) - timedelta(hours=offset_hours)

                    url = (
                        f"{NWS_BASE}/stations/{station_id}/observations"
                        f"?start={start_utc.isoformat().replace('+00:00', 'Z')}"
                        f"&end={end_utc.isoformat().replace('+00:00', 'Z')}"
                    )

                    data = await fetch_with_retry(
                        session, url, headers=NWS_HEADERS,
                        label=f"NWS-hist-{city}-{window_start}", timeout_sec=30.0,
                    )

                    if data:
                        # Group observations by local date
                        features = data.get("features", [])
                        daily_temps: Dict[date, float] = {}
                        for obs in features:
                            props = obs.get("properties", {})
                            temp_c = props.get("temperature", {}).get("value")
                            ts_str = props.get("timestamp", "")
                            if temp_c is None or not ts_str:
                                continue
                            try:
                                obs_time = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                                local_time = obs_time + timedelta(hours=offset_hours)
                                obs_date = local_time.date()
                                temp_f = temp_c * 9.0 / 5.0 + 32.0
                                if obs_date not in daily_temps or temp_f > daily_temps[obs_date]:
                                    daily_temps[obs_date] = temp_f
                            except (ValueError, TypeError):
                                continue

                        for obs_date, high_f in daily_temps.items():
                            if (city, obs_date) not in self._highs:
                                self._cache_daily_high(city, obs_date, high_f)

                    window_start = window_end
                    await asyncio.sleep(0.5)  # rate limit

        logger.info(f"Total daily highs available: {len(self._highs)}")
        return self._highs

    def get_actual_high(self, city: str, target_date: date) -> Optional[float]:
        """Look up a cached actual high."""
        return self._highs.get((city, target_date))

    # ── Climate Normals ───────────────────────────────────────────────

    def get_climatology(self, city: str, target_date: date) -> Optional[float]:
        """Get the 30-year normal high for city on this day-of-year."""
        day_of_year = target_date.timetuple().tm_yday
        city_normals = self._climatology.get(city)
        if city_normals is None:
            return None
        return city_normals.get(day_of_year)

    def load_climatology_from_actuals(self, highs: Dict[Tuple[str, date], float]):
        """
        Build approximate climatology from cached daily highs.

        Uses a 15-day rolling average around each day-of-year to smooth
        out individual year noise. This is a fallback when NOAA Climate
        Normals CSVs aren't available.
        """
        from collections import defaultdict
        by_city_doy: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

        for (city, d), high in highs.items():
            doy = d.timetuple().tm_yday
            by_city_doy[city][doy].append(high)

        for city, doy_map in by_city_doy.items():
            smoothed: Dict[int, float] = {}
            for doy in range(1, 367):
                temps = []
                for offset in range(-7, 8):  # 15-day window
                    neighbor = ((doy - 1 + offset) % 366) + 1
                    temps.extend(doy_map.get(neighbor, []))
                if temps:
                    smoothed[doy] = sum(temps) / len(temps)
            self._climatology[city] = smoothed

        logger.info(
            f"Built climatology for {len(self._climatology)} cities "
            f"from {len(highs)} observations"
        )

    # ── Gamma Closed Markets ─────────────────────────────────────────

    def _parse_gamma_market(
        self, item: dict
    ) -> Tuple[Optional[str], Optional[date], Optional[float], Optional[float], float]:
        """Parse a single Gamma API market into (city, date, lo, hi, price_yes)."""
        question = item.get("question", "")
        city = _match_city(question)
        mkt_date = _extract_date(question)

        group_title = item.get("groupItemTitle", "")
        bucket_source = group_title if group_title else question
        lo, hi = _parse_bucket(bucket_source)

        # Detect and convert Celsius
        unit = _detect_unit(question)
        if unit == "F" and group_title:
            unit = _detect_unit(group_title)
        if unit == "C":
            if lo is not None:
                lo = lo * 9.0 / 5.0 + 32.0
            if hi is not None:
                hi = hi * 9.0 / 5.0 + 32.0

        outcome_prices = item.get("outcomePrices", [])
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = json.loads(outcome_prices)
            except (json.JSONDecodeError, ValueError):
                outcome_prices = []
        price_yes = float(outcome_prices[0]) if outcome_prices else 0.0

        return city, mkt_date, lo, hi, price_yes

    async def fetch_gamma_closed_markets(self) -> List[dict]:
        """
        Fetch closed temperature markets from Gamma API.
        Caches to data/gamma_closed_markets.json.
        """
        if self._gamma_cache_file.exists():
            with open(self._gamma_cache_file) as f:
                self._gamma_markets = json.load(f)
            logger.info(f"Loaded {len(self._gamma_markets)} closed markets from cache")
            return self._gamma_markets

        logger.info("Fetching closed temperature markets from Gamma API...")
        raw_markets: List[dict] = []

        async with aiohttp.ClientSession() as session:
            offset = 0
            limit = 100
            max_pages = 50

            while offset < max_pages * limit:
                url = (
                    f"{GAMMA_BASE}/markets"
                    f"?closed=true&limit={limit}&offset={offset}"
                    f"&order=createdAt&ascending=false"
                )
                data = await fetch_with_retry(
                    session, url, timeout_sec=20.0, label="Gamma-closed"
                )
                if not data:
                    break

                for item in data:
                    q = (item.get("question") or "").lower()
                    if "highest" in q and "temperature" in q:
                        raw_markets.append(item)

                if len(data) < limit:
                    break
                offset += limit
                await asyncio.sleep(0.3)

        self._gamma_markets = raw_markets

        with open(self._gamma_cache_file, "w") as f:
            json.dump(raw_markets, f)

        logger.info(f"Fetched and cached {len(raw_markets)} closed temperature markets")
        return raw_markets

    def get_real_market_prices(
        self, city: str, target_date: date
    ) -> Optional[Dict[str, float]]:
        """
        Look up real Gamma market prices for a city/date.
        Returns: {outcome_label: price_yes} or None if no real data.
        """
        result: Dict[str, float] = {}
        for item in self._gamma_markets:
            c, d, lo, hi, price = self._parse_gamma_market(item)
            if c == city and d == target_date and lo is not None:
                label = item.get("groupItemTitle", f"bucket_{lo}_{hi}")
                result[label] = price
        return result if result else None
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_data.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add backtest_data.py tests/test_backtest_data.py
git commit -m "feat(backtest): add HistoricalDataLoader with NOAA + Gamma fetching"
```

---

### Task 3: Historical Forecast Approximator (`backtest_forecast.py`)

**Files:**
- Create: `backtest_forecast.py`
- Test: `tests/test_backtest_forecast.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest_forecast.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import date

def test_realistic_forecast_uses_climatology_not_actual():
    """Core anti-leakage test: realistic forecast must NOT center on actual."""
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})  # Jul 15 normal = 91°F

    random.seed(42)
    results = []
    for _ in range(1000):
        r, o = approx.generate("Miami", date(2025, 7, 15), days_out=3, actual_high=98.0)
        results.append(r.high_f)

    mean_forecast = sum(results) / len(results)
    # Realistic should center near climatology (91) NOT actual (98)
    assert abs(mean_forecast - 91.0) < 3.0, f"Mean {mean_forecast:.1f} too close to actual 98"
    assert abs(mean_forecast - 98.0) > 3.0, f"Mean {mean_forecast:.1f} leaked actual high"

def test_optimistic_forecast_centers_on_actual():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    random.seed(42)
    results = []
    for _ in range(1000):
        r, o = approx.generate("Miami", date(2025, 7, 15), days_out=3, actual_high=98.0)
        results.append(o.high_f)

    mean_forecast = sum(results) / len(results)
    # Optimistic should center near actual (98)
    assert abs(mean_forecast - 98.0) < 2.0

def test_sigma_increases_with_horizon():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"New York": {1: 38.0}})

    random.seed(42)
    r0, _ = approx.generate("New York", date(2025, 1, 1), days_out=0, actual_high=40.0)
    r5, _ = approx.generate("New York", date(2025, 1, 1), days_out=5, actual_high=40.0)
    assert r5.sigma > r0.sigma

def test_seasonal_sigma_multiplier():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {105: 85.0, 196: 91.0}})  # Apr 15, Jul 15

    random.seed(42)
    r_apr, _ = approx.generate("Miami", date(2025, 4, 15), days_out=3, actual_high=85.0)
    r_jul, _ = approx.generate("Miami", date(2025, 7, 15), days_out=3, actual_high=91.0)
    # April σ should be higher (1.35x) than July (0.9x)
    assert r_apr.sigma > r_jul.sigma

def test_regime_inference():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    # 10°F above normal → heat regime
    r, _ = approx.generate("Miami", date(2025, 7, 15), days_out=1, actual_high=101.0)
    assert r.regime in ("heat", "frontal", "extreme")

    # Normal range
    r, _ = approx.generate("Miami", date(2025, 7, 15), days_out=1, actual_high=91.0)
    assert r.regime in ("normal", "stable")

def test_confidence_from_sigma():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    r0, _ = approx.generate("Miami", date(2025, 7, 15), days_out=0, actual_high=91.0)
    r5, _ = approx.generate("Miami", date(2025, 7, 15), days_out=5, actual_high=91.0)
    # Closer horizon → higher confidence
    assert r0.confidence > r5.confidence
    assert 0.0 <= r0.confidence <= 1.0
    assert 0.0 <= r5.confidence <= 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_forecast.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_forecast'`

**Step 3: Write minimal implementation**

```python
# backtest_forecast.py
"""
backtest_forecast.py — Historical Forecast Approximator
========================================================
Generates plausible CityForecast-like objects for backtesting WITHOUT
seeing the actual daily high (avoiding look-ahead bias).

Two variants per city/date/horizon:
  - Realistic: climatology + calibrated NWS bias + Gaussian noise
  - Optimistic: centered on actual high with reduced noise (upper bound)

The realistic variant is the primary test. The gap between realistic
and optimistic quantifies how much look-ahead leakage would help.
"""

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

from forecast_scanner import compute_confidence

HORIZON_SIGMA = {
    0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0, 4: 3.8, 5: 4.5, 6: 5.3, 7: 6.0,
}

# NWS systematic bias: slight under-forecasting of highs at longer horizons
HORIZON_BIAS = {
    0: 0.0, 1: -0.3, 2: -0.5, 3: -0.7, 4: -0.85, 5: -1.0, 6: -1.1, 7: -1.2,
}

REGIME_BIAS = {
    "heat": -1.5,       # forecasters underestimate heat waves
    "cold": 1.0,        # overshoots on cold days
    "convective": -1.5, # miss cooling from thunderstorms
    "frontal": 1.0,     # conservative near frontal boundaries
    "normal": 0.0,
    "stable": 0.0,
    "extreme": -2.0,    # under-forecast extremes broadly
}

REGIME_SIGMA_MULT = {
    "heat": 1.2, "cold": 1.2, "convective": 1.3, "frontal": 1.2,
    "normal": 1.0, "stable": 0.9, "extreme": 1.4,
}

# Seasonal σ multipliers — spring/fall more volatile, summer/winter tighter
SEASONAL_SIGMA_MULT = {
    1: 1.0, 2: 1.05, 3: 1.25, 4: 1.35, 5: 1.2, 6: 1.1,
    7: 0.9, 8: 0.95, 9: 1.1, 10: 1.15, 11: 1.2, 12: 1.0,
}


@dataclass
class SyntheticForecast:
    """A generated forecast for backtesting."""
    high_f: float
    sigma: float
    confidence: float
    regime: str
    variant: str  # "realistic" or "optimistic"


class HistoricalForecastApproximator:
    """
    Generate plausible forecasts WITHOUT seeing the actual high.

    The realistic variant uses climatology as its anchor, applying:
    - Horizon-dependent bias (NWS verification statistics)
    - Regime-dependent bias (convective, frontal, etc.)
    - Gaussian noise scaled by horizon, season, and regime

    The optimistic variant centers on the actual high with reduced noise,
    serving as an upper bound / look-ahead leakage detector.
    """

    def __init__(self):
        self._climatology: Dict[str, Dict[int, float]] = {}

    def set_climatology(self, climatology: Dict[str, Dict[int, float]]):
        """Set climatology data: {city: {day_of_year: normal_high_f}}."""
        self._climatology = climatology

    def _infer_regime(
        self, city: str, target_date: date, actual_high: float, climatology: float
    ) -> str:
        """Infer a weather regime from actual conditions vs climatology."""
        delta = actual_high - climatology

        if abs(delta) > 12:
            return "extreme"
        if delta > 8:
            return "heat"
        if delta < -8:
            return "cold"

        # Southern cities in summer → convective likelihood
        month = target_date.month
        if city in ("Miami", "Houston", "Dallas") and month in (6, 7, 8, 9):
            if delta < -3:
                return "convective"

        # Spring/fall → frontal activity
        if month in (3, 4, 5, 10, 11):
            if abs(delta) > 5:
                return "frontal"

        if abs(delta) <= 2:
            return "stable"
        return "normal"

    def generate(
        self,
        city: str,
        target_date: date,
        days_out: int,
        actual_high: float,
    ) -> Tuple[SyntheticForecast, SyntheticForecast]:
        """
        Generate (realistic, optimistic) forecast pair.

        Args:
            city: City name
            target_date: The market resolution date
            days_out: Days before resolution (5 = far out, 0 = same day)
            actual_high: The actual observed high (used ONLY for regime
                         inference and optimistic variant — NOT leaked
                         into realistic forecast)
        """
        doy = target_date.timetuple().tm_yday
        month = target_date.month

        # Get climatology anchor
        city_clim = self._climatology.get(city, {})
        clim = city_clim.get(doy)
        if clim is None:
            # Fallback: use actual (degrades to optimistic)
            clim = actual_high

        # Infer regime
        regime = self._infer_regime(city, target_date, actual_high, clim)

        # Compute σ
        base_sigma = HORIZON_SIGMA.get(days_out, 6.0)
        seasonal_mult = SEASONAL_SIGMA_MULT.get(month, 1.0)
        regime_mult = REGIME_SIGMA_MULT.get(regime, 1.0)
        sigma = base_sigma * seasonal_mult * regime_mult

        # Realistic variant: climatology + bias + noise (no leakage)
        horizon_bias = HORIZON_BIAS.get(days_out, -1.2)
        regime_bias = REGIME_BIAS.get(regime, 0.0)
        total_bias = horizon_bias + regime_bias
        noise = random.gauss(0, sigma)
        realistic_high = clim + total_bias + noise

        # Confidence (reuse live bot's function)
        is_stable = regime in ("stable", "normal")
        confidence = compute_confidence(sigma, is_stable, regime_mult)

        realistic = SyntheticForecast(
            high_f=realistic_high,
            sigma=sigma,
            confidence=confidence,
            regime=regime,
            variant="realistic",
        )

        # Optimistic variant: centered on actual with reduced noise
        opt_noise = random.gauss(0, sigma * 0.5)
        optimistic = SyntheticForecast(
            high_f=actual_high + opt_noise,
            sigma=sigma,
            confidence=confidence,
            regime=regime,
            variant="optimistic",
        )

        return realistic, optimistic
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_forecast.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add backtest_forecast.py tests/test_backtest_forecast.py
git commit -m "feat(backtest): add forecast approximator with anti-leakage design"
```

---

### Task 4: Mispricing Model (`backtest_pricing.py`)

**Files:**
- Create: `backtest_pricing.py`
- Test: `tests/test_backtest_pricing.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest_pricing.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

def test_tail_buckets_overpriced():
    """Tail buckets should have prices higher than true probabilities."""
    from backtest_pricing import MispricingModel

    model = MispricingModel()
    true_probs = [0.02, 0.05, 0.15, 0.30, 0.25, 0.13, 0.07, 0.03]  # 8 buckets

    random.seed(42)
    results = {"tail_bias": [], "mode_bias": []}
    for _ in range(500):
        prices = model.generate_prices(true_probs, days_out=3)
        # First and last buckets are tails
        results["tail_bias"].append(prices[0] - true_probs[0])
        results["tail_bias"].append(prices[-1] - true_probs[-1])
        # Bucket 3 (0.30) is the mode
        results["mode_bias"].append(prices[3] - true_probs[3])

    avg_tail_bias = sum(results["tail_bias"]) / len(results["tail_bias"])
    avg_mode_bias = sum(results["mode_bias"]) / len(results["mode_bias"])

    assert avg_tail_bias > 0.02, f"Tail bias {avg_tail_bias:.3f} too low"
    assert avg_mode_bias < 0.0, f"Mode bias {avg_mode_bias:.3f} should be negative"

def test_prices_clamped():
    from backtest_pricing import MispricingModel

    model = MispricingModel()
    true_probs = [0.001, 0.999]  # extreme

    random.seed(42)
    for _ in range(100):
        prices = model.generate_prices(true_probs, days_out=5)
        for p in prices:
            assert 0.02 <= p <= 0.98, f"Price {p} out of bounds"

def test_convergence_near_resolution():
    """Prices should be more accurate (less biased) at days_out=0 vs days_out=5."""
    from backtest_pricing import MispricingModel

    model = MispricingModel()
    true_probs = [0.05, 0.15, 0.30, 0.30, 0.15, 0.05]

    random.seed(42)
    errors_far = []
    errors_near = []
    for _ in range(500):
        prices_5 = model.generate_prices(true_probs, days_out=5)
        prices_0 = model.generate_prices(true_probs, days_out=0)
        errors_far.append(sum(abs(p - t) for p, t in zip(prices_5, true_probs)))
        errors_near.append(sum(abs(p - t) for p, t in zip(prices_0, true_probs)))

    avg_far = sum(errors_far) / len(errors_far)
    avg_near = sum(errors_near) / len(errors_near)
    assert avg_near < avg_far, f"Near-resolution error ({avg_near:.3f}) should be < far ({avg_far:.3f})"

def test_calibrate_from_markets():
    """Test that calibration updates model parameters."""
    from backtest_pricing import MispricingModel

    model = MispricingModel()
    # Simulate closed market data: tail buckets were overpriced
    closed_data = [
        # (bucket_position_normalized, market_price, true_outcome)
        (0.0, 0.15, 0.0),   # tail, overpriced
        (0.0, 0.12, 0.0),   # tail, overpriced
        (0.5, 0.25, 1.0),   # mode, underpriced (won)
        (0.5, 0.28, 1.0),   # mode, underpriced
        (1.0, 0.10, 0.0),   # tail, overpriced
        (1.0, 0.14, 0.0),   # tail, overpriced
    ]
    model.calibrate(closed_data)
    # After calibration, tail_overpricing should be positive
    assert model.tail_overpricing > 0.0
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_pricing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_pricing'`

**Step 3: Write minimal implementation**

```python
# backtest_pricing.py
"""
backtest_pricing.py — Calibrated Mispricing Model for Backtesting
==================================================================
Generates synthetic market prices that reproduce the systematic biases
observed in real Polymarket temperature markets:
  - Tail buckets are overpriced (retail loves long shots)
  - Mode bucket is underpriced (retail underweights the most likely outcome)
  - Prices converge toward truth as resolution approaches

Calibrated from real Gamma API closed market data.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class MispricingModel:
    """
    Generate synthetic market prices with calibrated retail mispricing.

    Default parameters are conservative estimates. Call calibrate() with
    real closed market data to fit from observations.
    """

    # Fitted parameters (calibrated from Gamma closed markets)
    tail_overpricing: float = 0.07    # tails priced ~7¢ too high
    mode_underpricing: float = -0.04  # mode priced ~4¢ too low
    noise_sigma: float = 0.03        # random noise level
    convergence_rate: float = 0.4     # how fast biases shrink near resolution

    def generate_prices(
        self,
        true_probs: List[float],
        days_out: int,
    ) -> List[float]:
        """
        Generate synthetic market prices from true bucket probabilities.

        Args:
            true_probs: True probability per bucket (from forecast model)
            days_out: Days until market resolution (0-7)

        Returns:
            List of synthetic market prices (one per bucket)
        """
        n = len(true_probs)
        if n == 0:
            return []

        # Find mode bucket (highest true probability)
        mode_idx = max(range(n), key=lambda i: true_probs[i])

        # Convergence: biases shrink as resolution approaches
        convergence = max(0.0, 1.0 - (days_out / 7.0) * self.convergence_rate)

        prices = []
        for i, p_true in enumerate(true_probs):
            # Tail distance: normalized 0 (mode) to 1 (extreme tail)
            if n > 1:
                tail_distance = abs(i - mode_idx) / max(1, (n - 1) / 2)
                tail_distance = min(1.0, tail_distance)
            else:
                tail_distance = 0.0

            # Tail overpricing
            tail_bias = self.tail_overpricing * tail_distance

            # Mode underpricing
            mode_bias = self.mode_underpricing if i == mode_idx else 0.0

            # Apply convergence (biases fade near resolution)
            total_bias = (tail_bias + mode_bias) * convergence

            # Random noise (also fades near resolution)
            noise = random.gauss(0, self.noise_sigma * convergence)

            raw_price = p_true + total_bias + noise
            prices.append(max(0.02, min(0.98, raw_price)))

        return prices

    def calibrate(
        self,
        closed_data: List[Tuple[float, float, float]],
    ):
        """
        Calibrate model parameters from real closed market observations.

        Args:
            closed_data: List of (bucket_position_normalized, market_price, true_outcome)
                         where true_outcome is 1.0 for winning bucket, 0.0 otherwise.
                         bucket_position_normalized is 0.0 (tail) to 0.5 (mode) to 1.0 (tail).
        """
        if not closed_data:
            return

        tail_biases = []
        mode_biases = []
        all_noise = []

        for pos, price, outcome in closed_data:
            bias = price - outcome
            if pos < 0.2 or pos > 0.8:
                # Tail bucket
                tail_biases.append(bias)
            elif 0.4 <= pos <= 0.6:
                # Mode bucket
                mode_biases.append(bias)
            all_noise.append(abs(bias))

        if tail_biases:
            self.tail_overpricing = max(0.01, sum(tail_biases) / len(tail_biases))
        if mode_biases:
            self.mode_underpricing = min(-0.01, sum(mode_biases) / len(mode_biases))
        if all_noise:
            self.noise_sigma = max(0.01, sum(all_noise) / len(all_noise) * 0.5)
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_pricing.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add backtest_pricing.py tests/test_backtest_pricing.py
git commit -m "feat(backtest): add calibrated mispricing model"
```

---

### Task 5: Backtest Scorecard (`backtest_scorecard.py`)

**Files:**
- Create: `backtest_scorecard.py`
- Test: `tests/test_backtest_scorecard.py`

**Step 1: Write the failing test**

```python
# tests/test_backtest_scorecard.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

def _make_trade(city, target_date, days_out, side, price, size, won, pnl, variant="realistic"):
    from backtest_scorecard import BacktestTrade
    return BacktestTrade(
        city=city, target_date=target_date, days_out=days_out,
        side=side, outcome_label="72-73°F", bucket_low=72, bucket_high=74,
        p_true=0.30, market_price=price, ev=0.15, edge=0.10,
        kelly_fraction=0.05, position_size_usd=size, price_limit=price,
        actual_high=73.0, won=won, pnl=pnl, variant=variant,
        regime="normal",
    )

def test_win_rate():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("Miami", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 5.0),
        _make_trade("Miami", date(2025, 1, 2), 3, "BUY", 0.30, 10, True, 4.0),
        _make_trade("Miami", date(2025, 1, 3), 3, "BUY", 0.30, 10, False, -10.0),
    ]
    sc = BacktestScorecard(trades)
    assert abs(sc.win_rate("realistic") - 2/3) < 0.01

def test_max_drawdown():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("NY", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 10.0),
        _make_trade("NY", date(2025, 1, 2), 3, "BUY", 0.30, 10, True, 10.0),
        _make_trade("NY", date(2025, 1, 3), 3, "BUY", 0.30, 10, False, -15.0),
        _make_trade("NY", date(2025, 1, 4), 3, "BUY", 0.30, 10, False, -10.0),
        _make_trade("NY", date(2025, 1, 5), 3, "BUY", 0.30, 10, True, 8.0),
    ]
    sc = BacktestScorecard(trades)
    # Peak at +20, trough at -5, drawdown = 25
    assert sc.max_drawdown("realistic") == 25.0

def test_profit_factor():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("LA", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 20.0),
        _make_trade("LA", date(2025, 1, 2), 3, "BUY", 0.30, 10, False, -10.0),
    ]
    sc = BacktestScorecard(trades)
    assert abs(sc.profit_factor("realistic") - 2.0) < 0.01

def test_breakdown_by_city():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("Miami", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 5.0),
        _make_trade("Miami", date(2025, 1, 2), 3, "BUY", 0.30, 10, True, 3.0),
        _make_trade("New York", date(2025, 1, 1), 3, "BUY", 0.30, 10, False, -10.0),
    ]
    sc = BacktestScorecard(trades)
    by_city = sc.breakdown_by_city("realistic")
    assert by_city["Miami"]["win_rate"] == 1.0
    assert by_city["Miami"]["pnl"] == 8.0
    assert by_city["New York"]["win_rate"] == 0.0

def test_breakdown_by_horizon():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("Miami", date(2025, 1, 1), 0, "BUY", 0.30, 10, True, 5.0),
        _make_trade("Miami", date(2025, 1, 2), 0, "BUY", 0.30, 10, True, 5.0),
        _make_trade("Miami", date(2025, 1, 3), 5, "BUY", 0.30, 10, False, -10.0),
    ]
    sc = BacktestScorecard(trades)
    by_horizon = sc.breakdown_by_horizon("realistic")
    assert by_horizon[0]["win_rate"] == 1.0
    assert by_horizon[5]["win_rate"] == 0.0

def test_variant_comparison():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("Miami", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 5.0, "realistic"),
        _make_trade("Miami", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 8.0, "optimistic"),
    ]
    sc = BacktestScorecard(trades)
    assert sc.win_rate("realistic") == 1.0
    assert sc.win_rate("optimistic") == 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_scorecard.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtest_scorecard'`

**Step 3: Write minimal implementation**

```python
# backtest_scorecard.py
"""
backtest_scorecard.py — Backtest Results & Sensitivity Analysis
================================================================
Produces a terminal-formatted scorecard with:
  - Core metrics: Sharpe, Sortino, win rate, profit factor, max drawdown
  - Breakdowns: by city, month, horizon, regime, side
  - Variant comparison: realistic vs optimistic (look-ahead leakage)
  - Sensitivity sweeps: one-at-a-time parameter variation
  - Fragility notes: auto-generated warnings
  - CSV export
"""

import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class BacktestTrade:
    """A single backtest trade result."""
    city: str
    target_date: date
    days_out: int
    side: str               # "BUY" or "SELL"
    outcome_label: str
    bucket_low: Optional[float]
    bucket_high: Optional[float]
    p_true: float
    market_price: float
    ev: float
    edge: float
    kelly_fraction: float
    position_size_usd: float
    price_limit: float
    actual_high: float
    won: bool
    pnl: float
    variant: str            # "realistic" or "optimistic"
    regime: str = "normal"


class BacktestScorecard:
    """Compute and display backtest metrics."""

    def __init__(self, trades: List[BacktestTrade]):
        self.trades = trades

    def _filter(self, variant: str) -> List[BacktestTrade]:
        return [t for t in self.trades if t.variant == variant]

    # ── Core Metrics ──────────────────────────────────────────────────

    def win_rate(self, variant: str = "realistic") -> float:
        trades = self._filter(variant)
        if not trades:
            return 0.0
        return sum(1 for t in trades if t.won) / len(trades)

    def total_pnl(self, variant: str = "realistic") -> float:
        return sum(t.pnl for t in self._filter(variant))

    def trade_count(self, variant: str = "realistic") -> int:
        return len(self._filter(variant))

    def profit_factor(self, variant: str = "realistic") -> float:
        trades = self._filter(variant)
        gross_wins = sum(t.pnl for t in trades if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
        if gross_losses == 0:
            return float("inf") if gross_wins > 0 else 0.0
        return gross_wins / gross_losses

    def max_drawdown(self, variant: str = "realistic") -> float:
        trades = self._filter(variant)
        if not trades:
            return 0.0
        # Sort by date for sequential PnL
        sorted_trades = sorted(trades, key=lambda t: (t.target_date, t.days_out))
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in sorted_trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def avg_drawdown(self, variant: str = "realistic") -> float:
        trades = self._filter(variant)
        if not trades:
            return 0.0
        sorted_trades = sorted(trades, key=lambda t: (t.target_date, t.days_out))
        cumulative = 0.0
        peak = 0.0
        drawdowns = []
        for t in sorted_trades:
            cumulative += t.pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > 0:
                drawdowns.append(dd)
        return sum(drawdowns) / len(drawdowns) if drawdowns else 0.0

    def avg_ev(self, variant: str = "realistic") -> float:
        trades = self._filter(variant)
        if not trades:
            return 0.0
        return sum(t.ev for t in trades) / len(trades)

    def _daily_pnl_series(self, variant: str) -> List[float]:
        trades = self._filter(variant)
        daily: Dict[date, float] = defaultdict(float)
        for t in trades:
            daily[t.target_date] += t.pnl
        if not daily:
            return []
        return [daily[d] for d in sorted(daily.keys())]

    def sharpe_ratio(self, variant: str = "realistic") -> float:
        daily = self._daily_pnl_series(variant)
        if len(daily) < 2:
            return 0.0
        mean_r = sum(daily) / len(daily)
        variance = sum((r - mean_r) ** 2 for r in daily) / (len(daily) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.001
        return (mean_r / std_r) * math.sqrt(365)

    def sortino_ratio(self, variant: str = "realistic") -> float:
        daily = self._daily_pnl_series(variant)
        if len(daily) < 2:
            return 0.0
        mean_r = sum(daily) / len(daily)
        downside = [r for r in daily if r < 0]
        if not downside:
            return float("inf") if mean_r > 0 else 0.0
        downside_var = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var) if downside_var > 0 else 0.001
        return (mean_r / downside_std) * math.sqrt(365)

    def calmar_ratio(self, variant: str = "realistic") -> float:
        daily = self._daily_pnl_series(variant)
        if not daily:
            return 0.0
        total = sum(daily)
        annualized = total * (365 / max(1, len(daily)))
        dd = self.max_drawdown(variant)
        if dd == 0:
            return float("inf") if annualized > 0 else 0.0
        return annualized / dd

    # ── Breakdowns ────────────────────────────────────────────────────

    def _breakdown(self, variant: str, key_fn) -> Dict:
        trades = self._filter(variant)
        groups: Dict = defaultdict(list)
        for t in trades:
            groups[key_fn(t)].append(t)

        result = {}
        for key, group in sorted(groups.items()):
            wins = sum(1 for t in group if t.won)
            pnl = sum(t.pnl for t in group)
            result[key] = {
                "win_rate": wins / len(group) if group else 0.0,
                "pnl": pnl,
                "trades": len(group),
                "avg_ev": sum(t.ev for t in group) / len(group) if group else 0.0,
            }
        return result

    def breakdown_by_city(self, variant: str = "realistic") -> Dict:
        return self._breakdown(variant, lambda t: t.city)

    def breakdown_by_month(self, variant: str = "realistic") -> Dict:
        return self._breakdown(variant, lambda t: t.target_date.month)

    def breakdown_by_horizon(self, variant: str = "realistic") -> Dict:
        return self._breakdown(variant, lambda t: t.days_out)

    def breakdown_by_regime(self, variant: str = "realistic") -> Dict:
        return self._breakdown(variant, lambda t: t.regime)

    def breakdown_by_side(self, variant: str = "realistic") -> Dict:
        return self._breakdown(variant, lambda t: t.side)

    # ── Fragility Notes ───────────────────────────────────────────────

    def fragility_notes(self, variant: str = "realistic") -> List[str]:
        notes = []

        # Check city-level fragility
        by_city = self.breakdown_by_city(variant)
        for city, stats in by_city.items():
            if stats["trades"] >= 10 and stats["win_rate"] < 0.55:
                notes.append(f"⚠ {city}: win rate {stats['win_rate']:.0%} (below 55% threshold)")

        # Check regime fragility
        by_regime = self.breakdown_by_regime(variant)
        for regime, stats in by_regime.items():
            if stats["trades"] >= 10 and stats["win_rate"] < 0.55:
                notes.append(f"⚠ Regime '{regime}': win rate {stats['win_rate']:.0%}")

        # Check horizon
        by_horizon = self.breakdown_by_horizon(variant)
        for horizon, stats in by_horizon.items():
            if stats["trades"] >= 10 and stats["win_rate"] < 0.55:
                notes.append(f"⚠ Horizon {horizon}d: win rate {stats['win_rate']:.0%}")

        # Overall pass/fail
        sharpe = self.sharpe_ratio(variant)
        wr = self.win_rate(variant)
        dd_pct = self.max_drawdown(variant) / max(1, self.total_pnl(variant) + self.max_drawdown(variant)) * 100

        if sharpe >= 1.5:
            notes.append(f"✓ Sharpe {sharpe:.2f} passes threshold (≥1.5)")
        else:
            notes.append(f"⚠ Sharpe {sharpe:.2f} FAILS threshold (≥1.5)")
        if wr >= 0.62:
            notes.append(f"✓ Win rate {wr:.1%} passes threshold (≥62%)")
        else:
            notes.append(f"⚠ Win rate {wr:.1%} FAILS threshold (≥62%)")

        return notes

    # ── Terminal Output ───────────────────────────────────────────────

    def render(self, variant: str = "realistic") -> str:
        trades = self._filter(variant)
        if not trades:
            return "No trades to display."

        dates = sorted(set(t.target_date for t in trades))
        period = f"{dates[0]} → {dates[-1]} ({len(dates)} days)"
        cities = sorted(set(t.city for t in trades))

        lines = []
        lines.append("=" * 62)
        lines.append(f"  BACKTEST SCORECARD — {variant.title()} Variant")
        lines.append(f"  Period: {period}, {len(cities)} cities")
        lines.append("=" * 62)
        lines.append(f"  Sharpe:        {self.sharpe_ratio(variant):.2f}        "
                      f"Sortino:      {self.sortino_ratio(variant):.2f}")
        lines.append(f"  Win Rate:      {self.win_rate(variant):.1%}       "
                      f"Profit Factor: {self.profit_factor(variant):.2f}")
        lines.append(f"  Max Drawdown:  ${self.max_drawdown(variant):.2f}     "
                      f"Avg EV/trade:  {self.avg_ev(variant):+.3f}")
        lines.append(f"  Total PnL:     ${self.total_pnl(variant):+,.2f}     "
                      f"Trades: {self.trade_count(variant)}")
        lines.append(f"  Calmar:        {self.calmar_ratio(variant):.2f}")
        lines.append("-" * 62)

        # Variant comparison
        other = "optimistic" if variant == "realistic" else "realistic"
        other_sharpe = self.sharpe_ratio(other)
        if other_sharpe > 0:
            leakage = other_sharpe - self.sharpe_ratio(variant)
            lines.append(f"  VARIANT COMPARISON")
            lines.append(f"  {other.title()} Sharpe: {other_sharpe:.2f}    "
                          f"Look-ahead leakage: {leakage:+.2f}")
            lines.append("-" * 62)

        # By city
        lines.append(f"  {'BY CITY':<18} {'Sharpe':>7} {'WinRate':>8} {'PnL':>10} {'Trades':>7}")
        by_city = self.breakdown_by_city(variant)
        for city, stats in sorted(by_city.items(), key=lambda x: x[1]["pnl"], reverse=True):
            lines.append(f"  {city:<18} {'--':>7} {stats['win_rate']:>7.1%} "
                          f"${stats['pnl']:>+9.2f} {stats['trades']:>6}")
        lines.append("-" * 62)

        # By horizon
        lines.append(f"  {'BY HORIZON':<18} {'WinRate':>8} {'AvgEV':>8} {'Trades':>7}")
        by_horizon = self.breakdown_by_horizon(variant)
        for horizon in sorted(by_horizon.keys()):
            stats = by_horizon[horizon]
            lines.append(f"  {horizon}d out{' ':<13} {stats['win_rate']:>7.1%} "
                          f"{stats['avg_ev']:>+7.3f} {stats['trades']:>6}")
        lines.append("-" * 62)

        # Fragility
        lines.append("  FRAGILITY NOTES")
        for note in self.fragility_notes(variant):
            lines.append(f"  {note}")
        lines.append("=" * 62)

        # Methodology
        lines.append("")
        lines.append("  METHODOLOGY: Forecasts approximated from NOAA climatology +")
        lines.append("  calibrated horizon bias + Gaussian noise. Market prices from")
        lines.append("  Gamma API closed markets where available, synthetic elsewhere.")
        lines.append("  No execution simulation. Results represent signal quality.")
        lines.append("")

        return "\n".join(lines)

    # ── CSV Export ────────────────────────────────────────────────────

    def export_csv(self, filepath: str = "logs/backtest_results.csv"):
        path = Path(filepath)
        path.parent.mkdir(exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "variant", "city", "target_date", "days_out", "side",
                "outcome_label", "bucket_low", "bucket_high",
                "p_true", "market_price", "ev", "edge",
                "position_size_usd", "actual_high", "won", "pnl", "regime",
            ])
            for t in self.trades:
                w.writerow([
                    t.variant, t.city, t.target_date.isoformat(), t.days_out,
                    t.side, t.outcome_label, t.bucket_low, t.bucket_high,
                    f"{t.p_true:.4f}", f"{t.market_price:.4f}",
                    f"{t.ev:.4f}", f"{t.edge:.4f}",
                    f"{t.position_size_usd:.2f}", f"{t.actual_high:.1f}",
                    t.won, f"{t.pnl:.4f}", t.regime,
                ])
        return str(path)


class SensitivityAnalyzer:
    """
    Sweep parameters one-at-a-time and report how fragile the edge is.

    For each parameter, holds all others constant, varies one across
    a range, re-runs the backtest, and reports Sharpe/win rate/drawdown.
    """

    SWEEPS = {
        "min_edge": [0.05, 0.06, 0.08, 0.10, 0.12, 0.15],
        "kelly_fraction": [0.05, 0.10, 0.15, 0.20, 0.25],
        "sigma_mult": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        "max_kelly_mult": [0.75, 1.0, 1.25, 1.5],
    }

    def render_sweep_result(self, param: str, results: List[dict], current_value) -> str:
        lines = [f"  SENSITIVITY: {param}"]
        for r in results:
            marker = " ← current" if abs(r["value"] - current_value) < 0.001 else ""
            lines.append(
                f"  {r['value']:.2f}  → Sharpe={r['sharpe']:.2f}  "
                f"WR={r['win_rate']:.0%}  DD=${r['max_dd']:.0f}  "
                f"PnL=${r['pnl']:+.0f}  n={r['trades']}{marker}"
            )
        return "\n".join(lines)
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtest_scorecard.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add backtest_scorecard.py tests/test_backtest_scorecard.py
git commit -m "feat(backtest): add scorecard with metrics, breakdowns, fragility notes"
```

---

### Task 6: Backtest Engine + CLI (`backtester.py`)

**Files:**
- Create: `backtester.py`
- Test: `tests/test_backtester.py`

This is the integration task — ties all components together.

**Step 1: Write the failing test**

```python
# tests/test_backtester.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import date

def test_backtest_engine_produces_trades():
    """Integration test: engine should produce trades from synthetic data."""
    from backtester import BacktestEngine

    engine = BacktestEngine(bankroll=500.0, seed=42)

    # Inject minimal test data
    engine.loader._highs = {
        ("Miami", date(2025, 7, 15)): 91.0,
        ("Miami", date(2025, 7, 16)): 89.0,
        ("Miami", date(2025, 7, 17)): 93.0,
    }
    engine.loader._climatology = {
        "Miami": {196: 91.0, 197: 91.2, 198: 91.3},
    }
    engine.approximator.set_climatology(engine.loader._climatology)

    result = engine.run(
        cities=["Miami"],
        start_date=date(2025, 7, 15),
        end_date=date(2025, 7, 17),
    )

    # Should have generated some trades (both variants)
    assert len(result.trades) > 0
    variants = set(t.variant for t in result.trades)
    assert "realistic" in variants
    assert "optimistic" in variants

def test_backtest_deduplication():
    """Engine should not re-enter same market+bucket across horizons."""
    from backtester import BacktestEngine

    engine = BacktestEngine(bankroll=500.0, seed=42)

    engine.loader._highs = {
        ("Miami", date(2025, 7, 15)): 91.0,
    }
    engine.loader._climatology = {
        "Miami": {196: 91.0},
    }
    engine.approximator.set_climatology(engine.loader._climatology)

    result = engine.run(
        cities=["Miami"],
        start_date=date(2025, 7, 15),
        end_date=date(2025, 7, 15),
    )

    # For each variant, no duplicate market_id:outcome_label across horizons
    for variant in ["realistic", "optimistic"]:
        seen = set()
        for t in result.trades:
            if t.variant != variant:
                continue
            key = f"{t.city}_{t.target_date}_{t.outcome_label}_{t.variant}"
            assert key not in seen, f"Duplicate trade: {key}"
            seen.add(key)

def test_backtest_scoring_correct():
    """Verify BUY YES wins when actual temp is in bucket."""
    from backtester import BacktestEngine

    engine = BacktestEngine(bankroll=500.0, seed=42)

    # Actual high = 91°F, which falls in [90, 92) bucket
    engine.loader._highs = {
        ("Miami", date(2025, 7, 15)): 91.0,
    }
    engine.loader._climatology = {"Miami": {196: 91.0}}
    engine.approximator.set_climatology(engine.loader._climatology)

    result = engine.run(
        cities=["Miami"],
        start_date=date(2025, 7, 15),
        end_date=date(2025, 7, 15),
    )

    for t in result.trades:
        if t.variant != "realistic":
            continue
        if t.bucket_low is not None and t.bucket_high is not None:
            in_bucket = t.bucket_low <= 91.0 < t.bucket_high
            if t.side == "BUY":
                assert t.won == in_bucket
            else:  # SELL
                assert t.won == (not in_bucket)
```

**Step 2: Run test to verify it fails**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtester.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'backtester'`

**Step 3: Write minimal implementation**

```python
# backtester.py
"""
backtester.py — Historical Backtester for Weather Trading Bot
==============================================================
Replays 12-18 months of historical data through the exact same
DecisionEngine.evaluate() pipeline the live bot uses.

Produces a scorecard with Sharpe, win rate, drawdown, breakdowns,
parameter sensitivity analysis, and fragility notes.

Usage:
    python backtester.py                         # full backtest
    python backtester.py --quick                 # realistic only, no sensitivity
    python backtester.py --sensitivity           # include parameter sweeps
    python backtester.py --fetch-only            # just populate data caches
    python backtester.py --start 2025-01-01 --end 2026-03-31
    python backtester.py --cities "Miami,New York"
"""

import argparse
import asyncio
import logging
import random
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from backtest_data import HistoricalDataLoader
from backtest_forecast import HistoricalForecastApproximator, SyntheticForecast
from backtest_pricing import MispricingModel
from backtest_scorecard import BacktestScorecard, BacktestTrade, SensitivityAnalyzer
from backtest_tracker import MockTracker
from config import cfg
from decision_engine import DecisionEngine, TradeSignal
from forecast_scanner import CityForecast, bucket_probabilities
from polymarket_parser import TemperatureMarket, MarketOutcome
from resolution_tracker import ResolutionTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtester")


# Standard temperature buckets for synthetic markets (°F)
# Covers the range most US cities see across seasons
STANDARD_BUCKETS = [
    (None, 30), (30, 35), (35, 40), (40, 45), (45, 50),
    (50, 55), (55, 60), (60, 65), (65, 70), (70, 75),
    (75, 80), (80, 85), (85, 90), (90, 95), (95, 100),
    (100, None),
]


@dataclass
class BacktestResult:
    """Container for all backtest output."""
    trades: List[BacktestTrade] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    methodology_notes: List[str] = field(default_factory=list)

    def record(self, trade: BacktestTrade):
        self.trades.append(trade)


class BacktestEngine:
    """
    Replays historical city/date pairs through DecisionEngine.evaluate().

    For each city/date:
    1. Generate synthetic forecasts (realistic + optimistic)
    2. Build or load market prices
    3. Run the exact same DecisionEngine.evaluate()
    4. Score signals against actual observed temperature
    """

    def __init__(self, bankroll: float = 500.0, seed: int = 42):
        self.bankroll = bankroll
        self.seed = seed
        self.loader = HistoricalDataLoader()
        self.approximator = HistoricalForecastApproximator()
        self.pricing = MispricingModel()

    def _build_market(
        self,
        city: str,
        target_date: date,
        days_out: int,
        forecast_high: float,
        forecast_sigma: float,
    ) -> TemperatureMarket:
        """
        Build a TemperatureMarket from real Gamma data or synthetic prices.

        Uses real prices where available, falls back to the calibrated
        mispricing model for gaps.
        """
        # Try real Gamma prices first
        real_prices = self.loader.get_real_market_prices(city, target_date)

        if real_prices and len(real_prices) >= 3:
            # Build from real data
            outcomes = []
            for label, price_yes in real_prices.items():
                from polymarket_parser import _parse_bucket
                lo, hi = _parse_bucket(label)
                outcomes.append(MarketOutcome(
                    outcome_label=label,
                    token_id=f"backtest_{city}_{target_date}_{label}",
                    price_yes=price_yes,
                    price_no=round(1.0 - price_yes, 4),
                    bucket_low=lo,
                    bucket_high=hi,
                ))
        else:
            # Generate synthetic prices
            # Select relevant buckets based on forecast
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
                    token_id=f"backtest_{city}_{target_date}_{label}",
                    price_yes=price,
                    price_no=round(1.0 - price, 4),
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

    def _select_buckets(
        self, forecast_high: float
    ) -> List[Tuple[Optional[float], Optional[float]]]:
        """Select a subset of standard buckets centered on the forecast."""
        # Find the bucket containing the forecast
        center_idx = 0
        for i, (lo, hi) in enumerate(STANDARD_BUCKETS):
            if lo is not None and hi is not None:
                if lo <= forecast_high < hi:
                    center_idx = i
                    break
            elif lo is None and hi is not None:
                if forecast_high < hi:
                    center_idx = i
                    break
            elif hi is None and lo is not None:
                if forecast_high >= lo:
                    center_idx = i
                    break

        # Take 4 buckets on each side of center (9 total, like real markets)
        start = max(0, center_idx - 4)
        end = min(len(STANDARD_BUCKETS), center_idx + 5)
        return STANDARD_BUCKETS[start:end]

    @staticmethod
    def _bucket_label(lo: Optional[float], hi: Optional[float]) -> str:
        if lo is None:
            return f"{hi:.0f}°F or below"
        if hi is None:
            return f"{lo:.0f}°F or higher"
        return f"{lo:.0f}-{hi - 1:.0f}°F"

    def _score_signal(
        self, signal: TradeSignal, actual_high: float
    ) -> Tuple[bool, float]:
        """Score a trade signal against the actual temperature."""
        # Parse bucket bounds from signal
        # The signal has outcome_label but we need bucket bounds
        from polymarket_parser import _parse_bucket
        lo, hi = _parse_bucket(signal.outcome_label)

        # Check if actual temp is in bucket
        in_bucket = True
        if lo is not None and actual_high < lo:
            in_bucket = False
        if hi is not None and actual_high >= hi:
            in_bucket = False

        # BUY YES wins if in bucket, SELL wins if NOT in bucket
        if signal.side == "BUY":
            won = in_bucket
        else:
            won = not in_bucket

        # Calculate PnL (reuse resolution tracker logic)
        pnl = ResolutionTracker._calculate_pnl(
            signal.side, signal.price_limit,
            signal.position_size_usd, won,
            fee_rate=cfg.maker_fee_rate,
        )

        return won, pnl

    def run(
        self,
        cities: Optional[List[str]] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """
        Run the full backtest.

        Args:
            cities: List of cities to backtest (default: all configured)
            start_date: Start of backtest period
            end_date: End of backtest period

        Returns:
            BacktestResult with all trades
        """
        random.seed(self.seed)

        if cities is None:
            cities = cfg.cities
        if start_date is None:
            start_date = date.today() - timedelta(days=365)
        if end_date is None:
            end_date = date.today() - timedelta(days=1)

        result = BacktestResult(
            config_snapshot={
                "bankroll": self.bankroll,
                "seed": self.seed,
                "min_edge": cfg.min_edge,
                "kelly_fraction": cfg.kelly_fraction,
                "max_kelly_mult": cfg.max_kelly_mult,
                "fee_rate": cfg.fee_rate,
                "maker_fee_rate": cfg.maker_fee_rate,
            },
            methodology_notes=[
                "Forecasts approximated from NOAA climatology + calibrated bias + noise.",
                "Market prices from Gamma closed markets where available, synthetic elsewhere.",
                "No execution simulation (fills assumed at limit price).",
                f"Random seed: {self.seed}",
            ],
        )

        # Build climatology from loader data if not already set
        if not self.approximator._climatology:
            self.approximator.set_climatology(self.loader._climatology)

        engine = DecisionEngine()
        prev_target_date = None

        for city in cities:
            # One MockTracker per city (or reset daily)
            current = start_date
            while current <= end_date:
                actual_high = self.loader.get_actual_high(city, current)
                if actual_high is None:
                    current += timedelta(days=1)
                    continue

                # Reset tracker for each new target date
                mock_tracker = MockTracker(bankroll=self.bankroll)

                # Daily snapshots: evaluate at 5, 4, 3, 2, 1, 0 days out
                for days_out in range(5, -1, -1):
                    # Generate forecasts
                    forecast_r, forecast_o = self.approximator.generate(
                        city, current, days_out, actual_high
                    )

                    for variant_name, fc in [("realistic", forecast_r), ("optimistic", forecast_o)]:
                        # Build market using forecast-derived probabilities
                        market = self._build_market(
                            city, current, days_out,
                            fc.high_f, fc.sigma,
                        )

                        # Build CityForecast (same dataclass as live bot)
                        city_forecast = CityForecast(
                            city=city,
                            forecast_date=current,
                            high_f=fc.high_f,
                            sigma=fc.sigma,
                            confidence=fc.confidence,
                            weather_regime=fc.regime,
                            regime_multiplier=1.0,
                            is_stable=True,
                        )

                        # Run EXACT same decision engine as live
                        signals = engine.evaluate(
                            [(market, city_forecast)],
                            tracker=mock_tracker,
                        )

                        # Score each signal
                        for sig in signals:
                            won, pnl = self._score_signal(sig, actual_high)

                            # Record in mock tracker for dedup
                            mock_tracker.record_trade(
                                sig.market_id, sig.outcome_label,
                                sig.position_size_usd, city,
                            )

                            result.record(BacktestTrade(
                                city=city,
                                target_date=current,
                                days_out=days_out,
                                side=sig.side,
                                outcome_label=sig.outcome_label,
                                bucket_low=None,  # filled by scoring
                                bucket_high=None,
                                p_true=sig.p_true,
                                market_price=sig.market_price,
                                ev=sig.ev,
                                edge=sig.edge,
                                kelly_fraction=sig.kelly_fraction,
                                position_size_usd=sig.position_size_usd,
                                price_limit=sig.price_limit,
                                actual_high=actual_high,
                                won=won,
                                pnl=pnl,
                                variant=variant_name,
                                regime=fc.regime,
                            ))

                    # Reset engine daily PnL tracking between days
                    engine._reset_daily_if_needed()

                current += timedelta(days=1)

        logger.info(
            f"Backtest complete: {len(result.trades)} trades "
            f"({sum(1 for t in result.trades if t.variant == 'realistic')} realistic, "
            f"{sum(1 for t in result.trades if t.variant == 'optimistic')} optimistic)"
        )

        return result


# ── CLI ───────────────────────────────────────────────────────────────

async def fetch_data(loader: HistoricalDataLoader, cities: List[str], start: date, end: date):
    """Fetch all required historical data."""
    logger.info("Fetching NOAA daily highs...")
    await loader.fetch_daily_highs(cities, start, end)

    logger.info("Fetching Gamma closed markets...")
    await loader.fetch_gamma_closed_markets()

    logger.info("Building climatology from observations...")
    loader.load_climatology_from_actuals(loader._highs)


def main():
    parser = argparse.ArgumentParser(description="Backtest Weather Trading Strategy")
    parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--cities", type=str, default=None, help="Comma-separated cities")
    parser.add_argument("--sensitivity", action="store_true", help="Run parameter sweeps")
    parser.add_argument("--fetch-only", action="store_true", help="Only fetch data, don't run backtest")
    parser.add_argument("--quick", action="store_true", help="Realistic variant only, no sensitivity")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--bankroll", type=float, default=500.0, help="Simulated bankroll")
    args = parser.parse_args()

    start = date.fromisoformat(args.start) if args.start else date.today() - timedelta(days=365)
    end = date.fromisoformat(args.end) if args.end else date.today() - timedelta(days=1)
    cities = [c.strip() for c in args.cities.split(",")] if args.cities else cfg.cities

    engine = BacktestEngine(bankroll=args.bankroll, seed=args.seed)

    # Fetch data
    logger.info(f"Backtest: {start} → {end}, cities: {cities}")
    asyncio.run(fetch_data(engine.loader, cities, start, end))

    if args.fetch_only:
        logger.info("Data fetch complete. Exiting.")
        return

    # Set climatology on approximator
    engine.approximator.set_climatology(engine.loader._climatology)

    # Calibrate mispricing model from real Gamma data
    if engine.loader._gamma_markets:
        logger.info("Calibrating mispricing model from Gamma closed markets...")
        cal_data = []
        for item in engine.loader._gamma_markets:
            city, mkt_date, lo, hi, price = engine.loader._parse_gamma_market(item)
            if city and lo is not None:
                # Approximate bucket position (normalized 0-1)
                pos = 0.5  # simplified; improve with actual bucket context
                cal_data.append((pos, price, 0.0))  # 0.0 = lost (most buckets lose)
        if cal_data:
            engine.pricing.calibrate(cal_data)
            logger.info(
                f"Calibrated: tail_overpricing={engine.pricing.tail_overpricing:.3f}, "
                f"mode_underpricing={engine.pricing.mode_underpricing:.3f}"
            )

    # Run backtest
    logger.info("Running backtest...")
    result = engine.run(cities=cities, start_date=start, end_date=end)

    # Display scorecard
    scorecard = BacktestScorecard(result.trades)
    print(scorecard.render("realistic"))

    if not args.quick:
        print(scorecard.render("optimistic"))

    # Export CSV
    csv_path = scorecard.export_csv()
    logger.info(f"Results exported to {csv_path}")

    # Sensitivity analysis
    if args.sensitivity:
        logger.info("Running sensitivity sweeps...")
        analyzer = SensitivityAnalyzer()
        for param, values in analyzer.SWEEPS.items():
            sweep_results = []
            for val in values:
                # Override config for this sweep
                original = getattr(cfg, param, None)
                if original is not None:
                    setattr(cfg, param, val)
                    sweep_engine = BacktestEngine(bankroll=args.bankroll, seed=args.seed)
                    sweep_engine.loader = engine.loader  # reuse cached data
                    sweep_engine.approximator = engine.approximator
                    sweep_engine.pricing = engine.pricing
                    sweep_result = sweep_engine.run(cities=cities, start_date=start, end_date=end)
                    sweep_sc = BacktestScorecard(sweep_result.trades)
                    sweep_results.append({
                        "value": val,
                        "sharpe": sweep_sc.sharpe_ratio("realistic"),
                        "win_rate": sweep_sc.win_rate("realistic"),
                        "max_dd": sweep_sc.max_drawdown("realistic"),
                        "pnl": sweep_sc.total_pnl("realistic"),
                        "trades": sweep_sc.trade_count("realistic"),
                    })
                    setattr(cfg, param, original)  # restore

            current = getattr(cfg, param, 0)
            print(analyzer.render_sweep_result(param, sweep_results, current))
            print()

    # Final verdict
    sharpe = scorecard.sharpe_ratio("realistic")
    wr = scorecard.win_rate("realistic")
    dd = scorecard.max_drawdown("realistic")
    dd_pct = (dd / args.bankroll) * 100

    print("\n" + "=" * 62)
    print("  VERDICT")
    passed = True
    if sharpe < 1.5:
        print(f"  FAIL: Sharpe {sharpe:.2f} < 1.5")
        passed = False
    else:
        print(f"  PASS: Sharpe {sharpe:.2f} ≥ 1.5")
    if wr < 0.62:
        print(f"  FAIL: Win rate {wr:.1%} < 62%")
        passed = False
    else:
        print(f"  PASS: Win rate {wr:.1%} ≥ 62%")
    if dd_pct > 15:
        print(f"  FAIL: Max drawdown {dd_pct:.1f}% > 15%")
        passed = False
    else:
        print(f"  PASS: Max drawdown {dd_pct:.1f}% ≤ 15%")

    print(f"\n  {'STRATEGY VALIDATED ✓' if passed else 'STRATEGY NEEDS SURGERY ✗'}")
    print("=" * 62)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot" && python -m pytest tests/test_backtester.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add backtester.py tests/test_backtester.py
git commit -m "feat(backtest): add BacktestEngine with CLI, integrates all components"
```

---

### Task 7: Integration Test + First Real Run

**Files:**
- No new files
- Test: manual CLI run

**Step 1: Create data directory**

```bash
mkdir -p data
```

**Step 2: Quick validation with mock data (no API calls)**

```bash
cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot"
python -c "
from backtester import BacktestEngine
from datetime import date

e = BacktestEngine(seed=42)
# Inject minimal test data (skip API)
e.loader._highs = {
    ('Miami', date(2025, 7, d)): 88 + d % 5 for d in range(1, 31)
}
e.loader._climatology = {'Miami': {d: 90.5 for d in range(1, 366)}}
e.approximator.set_climatology(e.loader._climatology)

result = e.run(cities=['Miami'], start_date=date(2025, 7, 1), end_date=date(2025, 7, 30))

from backtest_scorecard import BacktestScorecard
sc = BacktestScorecard(result.trades)
print(sc.render('realistic'))
"
```

Expected: Scorecard printed to terminal with metrics.

**Step 3: Run all tests**

```bash
cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot"
python -m pytest tests/test_backtest_tracker.py tests/test_backtest_data.py tests/test_backtest_forecast.py tests/test_backtest_pricing.py tests/test_backtest_scorecard.py tests/test_backtester.py -v
```

Expected: All tests PASS

**Step 4: First real data run (fetches from APIs)**

```bash
cd "/Users/mihir/Downloads/Polymarket Weather Trading Bot"
python backtester.py --start 2025-06-01 --end 2025-12-31 --cities "Miami,New York" --quick
```

Expected: Data fetched, backtest runs, scorecard displayed with PASS/FAIL verdict.

**Step 5: Full run with sensitivity**

```bash
python backtester.py --start 2025-01-01 --end 2026-03-30 --sensitivity
```

Expected: Full scorecard + sensitivity sweep tables + verdict.

**Step 6: Commit all**

```bash
git add -A
git commit -m "feat(backtest): complete backtester with data loading, forecasting, pricing, scoring, and CLI"
```
