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
from collections import defaultdict
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
        self._climatology: Dict[str, Dict[int, float]] = {}
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
            by_city: Dict[str, List[date]] = {}
            for city, d in missing:
                by_city.setdefault(city, []).append(d)

            for city, dates in by_city.items():
                station_id = cfg.noaa_stations.get(city)
                if not station_id:
                    logger.warning(f"No NOAA station for {city}, skipping")
                    continue

                dates.sort()
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
                    await asyncio.sleep(0.5)

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
