"""
config.py — Central configuration loaded from .env
====================================================
Loads environment variables and provides typed access for all modules.
Ref: Research §6.2 (Recommended full-stack architecture) and §7 (Actionable next steps).
"""

import os
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # --- Credentials ---
    private_key: str = os.getenv("PRIVATE_KEY", "")
    chain_id: int = int(os.getenv("CHAIN_ID", "137"))
    polymarket_host: str = os.getenv("POLYMARKET_HOST", "https://clob.polymarket.com")
    funder: str = os.getenv("POLYMARKET_FUNDER", "")

    # --- Telegram ---
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # --- Trading params (§5.3 Kelly / §5.4 Daily loss caps) ---
    bankroll: float = float(os.getenv("BANKROLL", "500"))
    max_position_usd: float = float(os.getenv("MAX_POSITION_USD", "10"))
    daily_loss_cap: float = float(os.getenv("DAILY_LOSS_CAP", "50"))
    per_market_max_pct: float = float(os.getenv("PER_MARKET_MAX_PCT", "0.03"))
    kelly_fraction: float = float(os.getenv("KELLY_FRACTION", "0.15"))
    max_kelly_mult: float = float(os.getenv("MAX_KELLY_MULT", "1.25"))
    min_ev_threshold: float = float(os.getenv("MIN_EV_THRESHOLD", "0.03"))
    min_edge: float = float(os.getenv("MIN_EDGE", "0.08"))
    fee_rate: float = float(os.getenv("FEE_RATE", "0.02"))
    maker_spread_offset: float = float(os.getenv("MAKER_SPREAD_OFFSET", "0.005"))
    maker_fee_rate: float = float(os.getenv("MAKER_FEE_RATE", "0.0"))
    scan_interval_sec: int = int(os.getenv("SCAN_INTERVAL_SEC", "120"))
    telegram_min_ev: float = float(os.getenv("TELEGRAM_MIN_EV", "0.15"))

    # --- Mode ---
    mode: str = os.getenv("MODE", "dry-run")

    # --- Cities ---
    cities: List[str] = field(default_factory=list)
    nws_points: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def __post_init__(self):
        raw_cities = os.getenv("CITIES", "New York,Chicago,Los Angeles,Miami,Houston,Dallas")
        self.cities = [c.strip() for c in raw_cities.split(",") if c.strip()]

        raw_points = os.getenv(
            "NWS_POINTS",
            "New York:40.7789,-73.9692,Chicago:41.9742,-87.9073,"
            "Los Angeles:33.9425,-118.4081,Miami:25.7959,-80.2870,"
            "Houston:29.9844,-95.3414,Dallas:32.8998,-97.0403",
        )
        self.nws_points = {}
        # Format: City Name:lat,lon  (comma between lat and lon, comma between entries)
        # We split on the pattern "CityName:" to handle multi-word city names
        entries = []
        buf = ""
        for part in raw_points.split(","):
            buf = f"{buf},{part}" if buf else part
            # If buffer contains a colon AND a lat,lon pair after it, flush
            if ":" in buf:
                after_colon = buf.split(":", 1)[1]
                pieces = after_colon.split(",")
                if len(pieces) >= 2:
                    try:
                        float(pieces[0])
                        float(pieces[1])
                        entries.append(buf.strip())
                        buf = ""
                    except ValueError:
                        continue
        if buf.strip():
            entries.append(buf.strip())

        for entry in entries:
            if ":" not in entry:
                continue
            name, coords = entry.split(":", 1)
            parts = coords.split(",")
            if len(parts) >= 2:
                try:
                    self.nws_points[name.strip()] = (float(parts[0]), float(parts[1]))
                except ValueError:
                    continue

        raw_stations = os.getenv(
            "NOAA_STATIONS",
            "New York:KNYC,Chicago:KORD,Los Angeles:KLAX,"
            "Miami:KMIA,Houston:KIAH,Dallas:KDFW"
        )
        self.noaa_stations: Dict[str, str] = {}
        for pair in raw_stations.split(","):
            pair = pair.strip()
            if ":" in pair:
                city_name, station = pair.split(":", 1)
                self.noaa_stations[city_name.strip()] = station.strip()

    # City timezone UTC offsets (hours). Used for determining local date
    # and resolution timing. IANA names keep DST handling correct.
    CITY_TIMEZONES: Dict[str, str] = field(default_factory=lambda: {
        "New York": "America/New_York",
        "Chicago": "America/Chicago",
        "Los Angeles": "America/Los_Angeles",
        "Miami": "America/New_York",
        "Houston": "America/Chicago",
        "Dallas": "America/Chicago",
    })

    def utc_now(self) -> datetime:
        """Wrapper for testable current UTC time retrieval."""
        return datetime.now(timezone.utc)

    def city_zoneinfo(self, city: str) -> ZoneInfo:
        """Return the city's timezone, defaulting to New York for unknown cities."""
        tz_name = self.CITY_TIMEZONES.get(city, "America/New_York")
        return ZoneInfo(tz_name)

    @property
    def is_live(self) -> bool:
        return self.mode.lower() == "live"

    def city_local_now(self, city: str) -> datetime:
        """Get the current local time for a city using timezone-aware conversion."""
        return self.utc_now().astimezone(self.city_zoneinfo(city))

    def city_local_date(self, city: str) -> date:
        """Get the current local date for a city with DST-aware timezone conversion."""
        return self.city_local_now(city).date()

    def is_market_day_complete(self, city: str, market_date: date) -> bool:
        """
        Check if a market date is fully complete for a city.
        Returns True only if it's past 6 AM local time the NEXT day,
        ensuring all observations for the market date are in.
        """
        local_now = self.city_local_now(city)
        next_day_6am = datetime.combine(
            market_date + timedelta(days=1),
            time(hour=6, minute=0),
            tzinfo=self.city_zoneinfo(city),
        )
        return local_now >= next_day_6am


# Singleton
cfg = Config()
