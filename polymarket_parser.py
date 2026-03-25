"""
polymarket_parser.py — Module 2: Polymarket Market Parser
==========================================================
Ref: Research §2.3 (Example structures: daily high buckets),
     §3 (Patterns from top wallets — small multi-market positions),
     Core bot rules (Gamma API discovery).

Uses the Gamma API (gamma-api.polymarket.com) to:
  1. Discover active temperature markets.
  2. Parse bucket ranges from outcome labels.
  3. Extract token IDs, current prices, and liquidity.
  4. Match markets to forecast cities and dates.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import aiohttp

from api_utils import fetch_with_retry

logger = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"

# Regex patterns for parsing temperature market questions and outcomes
# Matches: "Highest temperature in New York City on March 24?" etc.
CITY_ALIASES: Dict[str, List[str]] = {
    "New York": ["new york", "nyc", "new york city"],
    "Chicago": ["chicago"],
    "Los Angeles": ["los angeles", "la", "l.a."],
    "Miami": ["miami"],
    "Houston": ["houston"],
    "Dallas": ["dallas", "dfw", "dallas-fort worth", "dallas/fort worth"],
}

# Parse bucket labels like "85° or higher", "80° to 84°", "Below 75°"
BUCKET_PATTERNS = [
    # "X° or higher" / "X°F or higher" / "X°+ " / "≥X°"
    (re.compile(r"(\d+)\s*°?\s*(?:F\s*)?(?:or\s+(?:higher|above|more)|\+|and\s+above)", re.I),
     lambda m: (float(m.group(1)), None)),
    # "≥X"
    (re.compile(r"[≥>=]+\s*(\d+)", re.I),
     lambda m: (float(m.group(1)), None)),
    # "Below X°" / "Under X°" / "Less than X°" / "<X°"
    (re.compile(r"(?:below|under|less\s+than|<)\s*(\d+)\s*°?\s*F?", re.I),
     lambda m: (None, float(m.group(1)))),
    # "X° to Y°" / "X-Y°" / "X°F - Y°F"
    (re.compile(r"(\d+)\s*°?\s*F?\s*(?:to|-|–)\s*(\d+)\s*°?\s*F?", re.I),
     lambda m: (float(m.group(1)), float(m.group(2)) + 1)),  # +1 because "80 to 84" means [80, 85)
]


@dataclass
class MarketOutcome:
    """A single outcome/bucket within a temperature market."""
    outcome_label: str
    token_id: str
    price_yes: float          # current Yes price (0–1)
    price_no: float           # 1 - price_yes
    bucket_low: Optional[float]   # °F lower bound (inclusive), None = unbounded
    bucket_high: Optional[float]  # °F upper bound (exclusive), None = unbounded


@dataclass
class TemperatureMarket:
    """A Polymarket temperature ladder market for one city/date."""
    market_id: str
    question: str
    city: str
    market_date: date
    resolution_source: str
    outcomes: List[MarketOutcome] = field(default_factory=list)
    active: bool = True
    volume: float = 0.0
    liquidity: float = 0.0


def _parse_bucket(label: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract (low, high) bounds from an outcome label string."""
    for pattern, extractor in BUCKET_PATTERNS:
        m = pattern.search(label)
        if m:
            return extractor(m)
    logger.debug(f"Could not parse bucket from label: {label}")
    return (None, None)


def _match_city(question: str) -> Optional[str]:
    """Match a market question to a configured city."""
    q_lower = question.lower()
    for city, aliases in CITY_ALIASES.items():
        for alias in aliases:
            if alias in q_lower:
                return city
    return None


def _extract_date(question: str) -> Optional[date]:
    """
    Extract date from market question text.
    Handles formats like:
      "... on March 24?" / "... on March 24, 2026?" / "... on 3/24/2026?"
    """
    # Try "Month Day, Year" or "Month Day"
    m = re.search(r"on\s+(\w+)\s+(\d{1,2})(?:,?\s*(\d{4}))?", question, re.I)
    if m:
        month_str, day_str, year_str = m.group(1), m.group(2), m.group(3)
        year = int(year_str) if year_str else datetime.now().year
        try:
            dt = datetime.strptime(f"{month_str} {day_str} {year}", "%B %d %Y")
            return dt.date()
        except ValueError:
            pass
        # Try abbreviated month
        try:
            dt = datetime.strptime(f"{month_str} {day_str} {year}", "%b %d %Y")
            return dt.date()
        except ValueError:
            pass

    # Try M/D/YYYY
    m = re.search(r"on\s+(\d{1,2})/(\d{1,2})/(\d{4})", question)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        except ValueError:
            pass

    return None


class PolymarketParser:
    """Discovers and parses active temperature ladder markets from the Gamma API."""

    def __init__(self) -> None:
        self._parse_stats: Dict[str, int] = {"total": 0, "success": 0, "failed": 0}
        self._failed_labels: List[str] = []  # track unique failed labels for debugging

    async def fetch_temperature_markets(self) -> List[TemperatureMarket]:
        """
        Query Gamma API for active temperature markets.
        Ref: Core bot rules — GET /markets?active=true&closed=false, filter "temperature".
        """
        self._parse_stats = {"total": 0, "success": 0, "failed": 0}
        markets: List[TemperatureMarket] = []
        offset = 0
        limit = 100  # Gamma API page size

        async with aiohttp.ClientSession() as session:
            while True:
                url = (
                    f"{GAMMA_BASE}/markets"
                    f"?active=true&closed=false&limit={limit}&offset={offset}"
                )
                data = await fetch_with_retry(
                    session, url, timeout_sec=20.0, label="Gamma-markets"
                )
                if not data:
                    break

                for item in data:
                    question = item.get("question", "")
                    # Filter: must mention temperature
                    q_lower = question.lower()
                    if "temperature" not in q_lower and "temp" not in q_lower:
                        continue

                    city = _match_city(question)
                    if not city:
                        continue

                    market_date = _extract_date(question)
                    if not market_date:
                        continue

                    # Build market object
                    mkt = TemperatureMarket(
                        market_id=item.get("id", item.get("conditionId", "")),
                        question=question,
                        city=city,
                        market_date=market_date,
                        resolution_source=item.get("resolutionSource", "NOAA/NWS"),
                        active=item.get("active", True),
                        volume=float(item.get("volume", 0) or 0),
                        liquidity=float(item.get("liquidity", 0) or 0),
                    )

                    # Parse outcomes / tokens
                    tokens = item.get("tokens", [])
                    outcomes_raw = item.get("outcomes", [])
                    if isinstance(outcomes_raw, str):
                        outcomes_raw = json.loads(outcomes_raw)

                    if tokens:
                        for tok in tokens:
                            outcome_label = tok.get("outcome", "")
                            token_id = tok.get("token_id", "")
                            price = float(tok.get("price", 0) or 0)
                            lo, hi = _parse_bucket(outcome_label)
                            self._parse_stats["total"] += 1
                            if lo is None and hi is None:
                                self._parse_stats["failed"] += 1
                                if outcome_label not in self._failed_labels:
                                    self._failed_labels.append(outcome_label)
                                    logger.warning(f"Unparsed bucket label: '{outcome_label}'")
                            else:
                                self._parse_stats["success"] += 1
                            mkt.outcomes.append(MarketOutcome(
                                outcome_label=outcome_label,
                                token_id=token_id,
                                price_yes=price,
                                price_no=round(1.0 - price, 4),
                                bucket_low=lo,
                                bucket_high=hi,
                            ))
                    elif outcomes_raw:
                        # Some markets list outcomes as string array + separate clobTokenIds
                        clob_ids = item.get("clobTokenIds", [])
                        if isinstance(clob_ids, str):
                            clob_ids = json.loads(clob_ids)
                        outcome_prices = item.get("outcomePrices", [])
                        if isinstance(outcome_prices, str):
                            outcome_prices = json.loads(outcome_prices)
                        for idx, label in enumerate(outcomes_raw):
                            token_id = clob_ids[idx] if idx < len(clob_ids) else ""
                            price = float(outcome_prices[idx]) if idx < len(outcome_prices) else 0.0
                            lo, hi = _parse_bucket(label)
                            self._parse_stats["total"] += 1
                            if lo is None and hi is None:
                                self._parse_stats["failed"] += 1
                                if label not in self._failed_labels:
                                    self._failed_labels.append(label)
                                    logger.warning(f"Unparsed bucket label: '{label}'")
                            else:
                                self._parse_stats["success"] += 1
                            mkt.outcomes.append(MarketOutcome(
                                outcome_label=label,
                                token_id=token_id,
                                price_yes=price,
                                price_no=round(1.0 - price, 4),
                                bucket_low=lo,
                                bucket_high=hi,
                            ))

                    if mkt.outcomes:
                        markets.append(mkt)

                if len(data) < limit:
                    break
                offset += limit

        if self._parse_stats["total"] > 0:
            fail_rate = self._parse_stats["failed"] / self._parse_stats["total"] * 100
            logger.info(
                f"Bucket parse stats: {self._parse_stats['success']}/{self._parse_stats['total']} "
                f"successful ({fail_rate:.1f}% failure rate)"
            )
            if fail_rate > 20:
                logger.warning(
                    f"High bucket parse failure rate ({fail_rate:.1f}%) — "
                    f"check regex patterns. Failed labels: {self._failed_labels[:5]}"
                )

        logger.info(f"Parsed {len(markets)} active temperature markets")
        return markets

    def match_forecasts(
        self,
        markets: List[TemperatureMarket],
        forecasts,  # List[CityForecast] — avoid circular import
    ) -> List[Tuple["TemperatureMarket", "CityForecast"]]:
        """
        Match markets to forecasts by city and date.
        Returns list of (market, forecast) pairs.
        """
        from forecast_scanner import CityForecast

        forecast_map: Dict[str, CityForecast] = {}
        for fc in forecasts:
            key = f"{fc.city}_{fc.forecast_date.isoformat()}"
            forecast_map[key] = fc

        matches = []
        for mkt in markets:
            key = f"{mkt.city}_{mkt.market_date.isoformat()}"
            fc = forecast_map.get(key)
            if fc:
                matches.append((mkt, fc))
            else:
                logger.debug(f"No forecast match for market: {mkt.city} {mkt.market_date}")

        logger.info(f"Matched {len(matches)} market-forecast pairs")
        return matches
