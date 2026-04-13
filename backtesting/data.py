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
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

from infrastructure.http import fetch_with_retry
from trading.markets import _detect_unit, _extract_date, _match_city, _parse_bucket

from .price_history import PriceHistoryFetcher

logger = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "(polymarket-weather-bot-backtest, contact@example.com)",
    "Accept": "application/geo+json",
}
NCEI_BASE = "https://www.ncei.noaa.gov/access/services/data/v1"
GAMMA_BASE = "https://gamma-api.polymarket.com"

# GHCND station IDs for NCEI daily summaries (historical TMAX)
GHCND_STATIONS = {
    "New York": "USW00094728",
    "Chicago": "USW00094846",
    "Los Angeles": "USW00023174",
    "Miami": "USW00012839",
    "Houston": "USW00012960",
    "Dallas": "USW00013911",
}


class HistoricalDataLoader:
    """Loads and caches historical weather and market data for backtesting."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self._highs_cache_file = self.data_dir / "noaa_daily_highs.csv"
        self._climatology_file = self.data_dir / "climate_normals.csv"
        self._gamma_cache_file = self.data_dir / "gamma_closed_markets.json"

        self._highs: Dict[Tuple[str, date], float] = {}
        self._climatology: Dict[str, Dict[int, float]] = {}
        self._gamma_markets: List[dict] = []
        self._price_fetcher = PriceHistoryFetcher()
        self._token_map: Dict[Tuple[str, date, str], str] = {}  # (city, date, label) -> token_id

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
        Uses NCEI Climate Data Online (daily-summaries TMAX) which has
        full historical data — unlike NWS observations which only keep ~14 days.
        Results cached to CSV so API calls happen once.
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

        logger.info(f"Fetching {len(missing)} missing daily highs from NCEI...")

        async with aiohttp.ClientSession() as session:
            by_city: Dict[str, List[date]] = {}
            for city, d in missing:
                by_city.setdefault(city, []).append(d)

            for city, dates in by_city.items():
                ghcnd_id = GHCND_STATIONS.get(city)
                if not ghcnd_id:
                    logger.warning(f"No GHCND station for {city}, skipping")
                    continue

                dates.sort()
                # Fetch in 90-day windows (NCEI supports large ranges)
                window_start = dates[0]
                while window_start <= dates[-1]:
                    window_end = min(window_start + timedelta(days=90), dates[-1])

                    url = (
                        f"{NCEI_BASE}?dataset=daily-summaries"
                        f"&stations={ghcnd_id}"
                        f"&startDate={window_start.isoformat()}"
                        f"&endDate={window_end.isoformat()}"
                        f"&dataTypes=TMAX&units=standard&format=json"
                    )

                    try:
                        async with session.get(
                            url, timeout=aiohttp.ClientTimeout(total=30)
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json(content_type=None)
                                if isinstance(data, list):
                                    for row in data:
                                        try:
                                            obs_date = date.fromisoformat(row["DATE"])
                                            tmax = float(row["TMAX"])
                                            if (city, obs_date) not in self._highs:
                                                self._cache_daily_high(city, obs_date, tmax)
                                        except (KeyError, ValueError, TypeError):
                                            continue
                                    logger.info(
                                        f"  {city}: fetched {len(data)} days "
                                        f"({window_start} to {window_end})"
                                    )
                                else:
                                    logger.warning(f"  {city}: unexpected response format")
                            else:
                                logger.warning(
                                    f"  {city}: NCEI returned {resp.status} "
                                    f"for {window_start} to {window_end}"
                                )
                    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                        logger.warning(f"  {city}: NCEI request failed: {e}")

                    window_start = window_end + timedelta(days=1)
                    await asyncio.sleep(0.3)  # rate limit

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
        Uses a 15-day rolling average around each day-of-year.
        """
        by_city_doy: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))

        for (city, d), high in highs.items():
            doy = d.timetuple().tm_yday
            by_city_doy[city][doy].append(high)

        for city, doy_map in by_city_doy.items():
            smoothed: Dict[int, float] = {}
            for doy in range(1, 367):
                temps = []
                for offset in range(-7, 8):
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

    async def fetch_price_histories(
        self, session: aiohttp.ClientSession, fidelity: int = 60
    ) -> int:
        """Fetch CLOB price histories for all closed temperature markets.

        Also builds _token_map: (city, date, label) -> YES token ID
        so get_decision_time_prices() can look up by city/date.
        """
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
