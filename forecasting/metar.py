"""
metar_fetcher.py — METAR Aviation Weather Data Fetcher
======================================================
Fetches real-time METAR observations from aviationweather.gov as a
third ensemble source alongside NWS and OWM.

METAR reports provide actual observed conditions at airport weather
stations, which are highly accurate for current temperature. For
same-day markets, a recent observation is more informative than a
forecast — it anchors the ensemble to ground truth.

Temperature parsing handles two METAR formats:
  - Standard: "27/19" (temp/dewpoint in °C, "M" prefix = negative)
  - Precise T-group: "T02720189" (tenths of °C, 0=positive, 1=negative)

API: https://aviationweather.gov/api/data/metar (returns plain text, not JSON)
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import aiohttp

from config import Config, cfg

logger = logging.getLogger(__name__)

METAR_API_BASE = "https://aviationweather.gov/api/data/metar"


@dataclass
class MetarObservation:
    """Parsed METAR observation for a single station."""
    station: str
    observed_at: datetime
    temp_f: float
    raw_metar: str


def _parse_metar_temp(raw: str) -> Optional[float]:
    """
    Parse temperature from a raw METAR string.

    Tries precise T-group first (e.g., T02720189 = 27.2°C / 18.9°C),
    then falls back to standard temp/dewpoint group (e.g., 27/19 or M03/M07).

    Returns temperature in °F, or None if unparsable.
    """
    # Try precise T-group: T{sign}{temp_tenths}{sign}{dewpoint_tenths}
    # Sign: 0 = positive, 1 = negative
    # Example: T02720189 → temp = +27.2°C, dewpoint = +18.9°C
    # Example: T10301070 → temp = -3.0°C, dewpoint = -7.0°C
    t_match = re.search(r'\bT(\d{4})(\d{4})\b', raw)
    if t_match:
        temp_raw = t_match.group(1)
        sign = -1 if temp_raw[0] == '1' else 1
        temp_c = sign * int(temp_raw[1:]) / 10.0
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        return temp_f

    # Fall back to standard temp/dewpoint: XX/XX or MXX/MXX
    # M prefix means negative (e.g., M03 = -3°C)
    td_match = re.search(r'\b(M?\d{2})/(M?\d{2})\b', raw)
    if td_match:
        temp_str = td_match.group(1)
        if temp_str.startswith('M'):
            temp_c = -float(temp_str[1:])
        else:
            temp_c = float(temp_str)
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        return temp_f

    return None


class MetarFetcher:
    """
    Fetches and parses METAR observations for configured NOAA stations.

    Used as a third ensemble source for same-day forecasts, where
    actual observations are more reliable than model predictions.
    """

    def __init__(self, config: Config = cfg):
        self.config = config
        self._stations: Dict[str, str] = config.noaa_stations  # city → station_id

    async def fetch_observation(
        self, session: aiohttp.ClientSession, city: str
    ) -> Optional[MetarObservation]:
        """
        Fetch latest METAR observation for a city's configured station.

        Returns MetarObservation or None if unavailable/unparsable.
        The aviationweather.gov API returns plain text, not JSON.
        """
        station = self._stations.get(city)
        if not station:
            logger.debug(f"No METAR station configured for {city}")
            return None

        url = f"{METAR_API_BASE}?ids={station}&format=raw&taf=false&hours=1"

        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"METAR fetch failed for {station}: HTTP {resp.status}")
                    return None
                raw_text = await resp.text()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.warning(f"METAR fetch error for {station}: {e}")
            return None

        raw_text = raw_text.strip()
        if not raw_text:
            logger.debug(f"METAR: empty response for {station}")
            return None

        # Parse temperature
        temp_f = _parse_metar_temp(raw_text)
        if temp_f is None:
            logger.warning(f"METAR: could not parse temperature from {station}: {raw_text[:80]}")
            return None

        obs = MetarObservation(
            station=station,
            observed_at=datetime.now(timezone.utc),
            temp_f=temp_f,
            raw_metar=raw_text,
        )
        logger.info(f"METAR {station} ({city}): {temp_f:.1f}°F — {raw_text[:60]}")
        return obs

    async def fetch_all(self) -> Dict[str, MetarObservation]:
        """
        Fetch METAR observations for all configured cities.
        Returns dict keyed by city name → MetarObservation.
        """
        results: Dict[str, MetarObservation] = {}

        async with aiohttp.ClientSession() as session:
            tasks = []
            cities = []
            for city in self._stations:
                cities.append(city)
                tasks.append(self.fetch_observation(session, city))

            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            for city, result in zip(cities, fetched):
                if isinstance(result, MetarObservation):
                    results[city] = result
                elif isinstance(result, Exception):
                    logger.debug(f"METAR fetch exception for {city}: {result}")

        logger.info(f"Fetched {len(results)} METAR observations from {len(self._stations)} stations")
        return results
