"""
polymarket_parser.py — Module 2: Polymarket Market Parser
==========================================================
Ref: Research §2.3 (Example structures: daily high buckets),
     §3 (Patterns from top wallets — small multi-market positions),
     Core bot rules (Gamma API discovery).

Uses the Gamma API (gamma-api.polymarket.com) to:
  1. Discover active temperature markets.
  2. Parse bucket ranges from outcome labels OR groupItemTitle/question text.
  3. Extract token IDs, current prices, and liquidity.
  4. Match markets to forecast cities and dates.

v4.2: Rewritten to handle Polymarket's new market format where each temperature
bucket is a separate binary (Yes/No) market, grouped by negRiskMarketID.
Old format: one market with multiple outcome buckets.
New format: each bucket = its own binary market with outcomes=["Yes","No"],
            bucket info in groupItemTitle or question text,
            sibling buckets share the same negRiskMarketID.
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

# Parse bucket labels from groupItemTitle or outcome labels.
# New Polymarket format uses groupItemTitle like:
#   "68-69°F", "67°F or below", "86°F or higher", "11°C", "8°C or below"
# Also handles old format labels like:
#   "85° or higher", "80° to 84°", "Below 75°"
BUCKET_PATTERNS = [
    # "X°F or below" / "X°C or below" / "X° or below" / "X or below"
    (re.compile(r"(\-?\d+)\s*°?\s*[FC]?\s*(?:or\s+(?:below|less|lower|under))", re.I),
     lambda m: (None, float(m.group(1)) + 1)),  # +1: "67°F or below" means [−∞, 68)
    # "Below X°" / "Under X°" / "Less than X°" / "<X°" / "≤X"
    (re.compile(r"(?:below|under|less\s+than|<|≤)\s*(\-?\d+)\s*°?\s*[FC]?", re.I),
     lambda m: (None, float(m.group(1)))),
    # "X°F or higher" / "X°C or higher" / "X°+ " / "≥X°"
    (re.compile(r"(\-?\d+)\s*°?\s*[FC]?\s*(?:or\s+(?:higher|above|more)|\+|and\s+above)", re.I),
     lambda m: (float(m.group(1)), None)),
    # "≥X" / ">=X"
    (re.compile(r"[≥>=]+\s*(\-?\d+)", re.I),
     lambda m: (float(m.group(1)), None)),
    # "between X-Y°F" / "X-Y°F" / "X–Y°F" / "X to Y°F"
    (re.compile(r"(?:between\s+)?(\-?\d+)\s*(?:°?\s*[FC]?\s*)?(?:to|-|–)\s*(\-?\d+)\s*°?\s*[FC]?", re.I),
     lambda m: (float(m.group(1)), float(m.group(2)) + 1)),  # +1: "68-69°F" means [68, 70)
    # Exact single temperature: "11°C" / "72°F" / "68" (binary: will it be exactly X?)
    # This is common in the new format — single degree bucket
    # Fix: Made ° optional (°?) to catch bare number labels like "68" or "68F"
    (re.compile(r"^(\-?\d+)\s*°?\s*[FC]?$", re.I),
     lambda m: (float(m.group(1)), float(m.group(1)) + 1)),  # "11°C" means [11, 12)
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
    """Extract (low, high) bounds from a bucket label string."""
    for pattern, extractor in BUCKET_PATTERNS:
        m = pattern.search(label)
        if m:
            return extractor(m)
    logger.debug(f"Could not parse bucket from label: {label}")
    return (None, None)


def _detect_unit(text: str) -> str:
    """Detect whether market uses °F or °C from question/label text."""
    if "°C" in text or "°c" in text:
        return "C"
    return "F"  # default to Fahrenheit


def _celsius_to_fahrenheit(temp_c: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return temp_c * 9.0 / 5.0 + 32.0


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
    """
    Discovers and parses active temperature ladder markets from the Gamma API.

    v4.2: Handles the new Polymarket format where each temperature bucket is a
    separate binary (Yes/No) market. Markets are discovered individually and then
    grouped by negRiskMarketID to reconstruct the temperature ladder.
    """

    def __init__(self) -> None:
        self._parse_stats: Dict[str, int] = {"total": 0, "success": 0, "failed": 0}
        self._failed_labels: List[str] = []  # track unique failed labels for debugging

    async def fetch_temperature_markets(self) -> List[TemperatureMarket]:
        """
        Query Gamma API for active temperature markets.

        Strategy: Since the Gamma API's paginated endpoint doesn't reliably return
        temperature markets, we use multiple discovery methods:
          1. Volume-sorted search (catches popular/recent markets)
          2. Paginated scan with 'highest' keyword filter
          3. Group sibling markets by negRiskMarketID to reconstruct ladders

        Each binary Yes/No market becomes one MarketOutcome in the reconstructed
        TemperatureMarket ladder.
        """
        self._parse_stats = {"total": 0, "success": 0, "failed": 0}
        self._failed_labels = []

        # Collect individual binary temperature markets
        raw_markets: Dict[str, dict] = {}  # market_id -> market data

        async with aiohttp.ClientSession() as session:
            # Method 1: Volume-sorted (catches most active temperature markets)
            for sort_order in ["volume", "createdAt", "liquidity"]:
                url = (
                    f"{GAMMA_BASE}/markets"
                    f"?active=true&closed=false&limit=100"
                    f"&order={sort_order}&ascending=false"
                )
                data = await fetch_with_retry(
                    session, url, timeout_sec=20.0, label=f"Gamma-{sort_order}"
                )
                if data:
                    for item in data:
                        q = (item.get("question") or "").lower()
                        if "highest" in q and "temperature" in q:
                            mid = str(item.get("id", item.get("conditionId", "")))
                            raw_markets[mid] = item

            # Method 2: Paginated scan (broader but unreliable for temperature)
            offset = 0
            limit = 100
            max_pages = 10  # Don't scan forever
            while offset < max_pages * limit:
                url = (
                    f"{GAMMA_BASE}/markets"
                    f"?active=true&closed=false&limit={limit}&offset={offset}"
                )
                data = await fetch_with_retry(
                    session, url, timeout_sec=20.0, label="Gamma-paginated"
                )
                if not data:
                    break

                for item in data:
                    q = (item.get("question") or "").lower()
                    if "highest" in q and "temperature" in q:
                        mid = str(item.get("id", item.get("conditionId", "")))
                        raw_markets[mid] = item

                if len(data) < limit:
                    break
                offset += limit

            # Method 3: Fetch sibling markets for discovered negRiskMarketIDs.
            # The Gamma API doesn't reliably return all temperature markets in
            # pagination, but once we find one market from a ladder, we can
            # fetch its neighbors by scanning nearby IDs (they're sequential).
            # We also scan a wider range to catch ladders that share NO seed
            # markets with the initial discovery (e.g., different dates for
            # the same city that happen to have nearby IDs).
            discovered_neg_ids = set()
            seed_ids: List[int] = []
            for mid, item in raw_markets.items():
                neg_id = item.get("negRiskMarketID")
                if neg_id:
                    discovered_neg_ids.add(neg_id)
                try:
                    seed_ids.append(int(item.get("id", 0)))
                except (ValueError, TypeError):
                    pass

            if seed_ids:
                # Scan a wide window around each seed ID to find sibling buckets
                # AND adjacent ladders (different dates for same city).
                # Temperature ladders have 8-15 buckets with sequential IDs,
                # and different dates are often 11-15 IDs apart.
                ids_to_check = set()
                for seed in seed_ids:
                    # Wide scan: ±50 covers ~3-4 adjacent ladders
                    for offset_id in range(-50, 51):
                        ids_to_check.add(seed + offset_id)
                # Remove IDs we already have
                existing_ids = set()
                for mid in raw_markets:
                    try:
                        existing_ids.add(int(mid))
                    except (ValueError, TypeError):
                        pass
                ids_to_check -= existing_ids

                # Fetch in batches — individual market endpoints
                logger.info(f"Fetching {len(ids_to_check)} sibling market IDs...")
                fetch_tasks = []
                for check_id in sorted(ids_to_check):
                    url = f"{GAMMA_BASE}/markets/{check_id}"
                    fetch_tasks.append(
                        fetch_with_retry(session, url, timeout_sec=10.0,
                                         label=f"Gamma-sibling-{check_id}")
                    )
                # Run in batches of 20 to avoid overwhelming the API
                for batch_start in range(0, len(fetch_tasks), 20):
                    batch = fetch_tasks[batch_start:batch_start + 20]
                    results = await asyncio.gather(*batch, return_exceptions=True)
                    for result in results:
                        if isinstance(result, dict):
                            q = (result.get("question") or "").lower()
                            if "highest" in q and "temperature" in q:
                                mid = str(result.get("id", result.get("conditionId", "")))
                                if mid not in raw_markets:
                                    raw_markets[mid] = result

        logger.info(f"Discovered {len(raw_markets)} individual temperature markets")

        if not raw_markets:
            logger.info("No active temperature markets found on Polymarket")
            return []

        # Group binary markets by negRiskMarketID to reconstruct ladders
        # Markets sharing the same negRiskMarketID are buckets in the same ladder
        ladders: Dict[str, List[dict]] = {}  # negRiskMarketID -> [market, ...]
        standalone: List[dict] = []  # markets without negRiskMarketID

        for mid, item in raw_markets.items():
            neg_risk_id = item.get("negRiskMarketID")
            if neg_risk_id:
                if neg_risk_id not in ladders:
                    ladders[neg_risk_id] = []
                ladders[neg_risk_id].append(item)
            else:
                standalone.append(item)

        markets: List[TemperatureMarket] = []

        # Process grouped ladders (new format: each bucket = separate binary market)
        for neg_risk_id, siblings in ladders.items():
            # All siblings should share city and date — use the first one
            first = siblings[0]
            question = first.get("question", "")
            city = _match_city(question)
            if not city:
                continue

            market_date = _extract_date(question)
            if not market_date:
                continue

            # Detect unit from question first; if not found, check bucket labels
            unit = _detect_unit(question)
            if unit == "F":
                # Check if any sibling bucket label contains °C
                for sib_check in siblings:
                    gt = sib_check.get("groupItemTitle", "")
                    sq = sib_check.get("question", "")
                    if _detect_unit(gt) == "C" or _detect_unit(sq) == "C":
                        unit = "C"
                        break

            mkt = TemperatureMarket(
                market_id=neg_risk_id,  # Use group ID as market ID
                question=question,
                city=city,
                market_date=market_date,
                resolution_source=first.get("resolutionSource", ""),
                active=True,
                volume=sum(float(s.get("volume", 0) or 0) for s in siblings),
                liquidity=sum(float(s.get("liquidity", 0) or 0) for s in siblings),
            )

            for sib in siblings:
                # Parse bucket from groupItemTitle (preferred) or question
                group_title = sib.get("groupItemTitle", "")
                bucket_source = group_title if group_title else sib.get("question", "")
                lo, hi = _parse_bucket(bucket_source)

                self._parse_stats["total"] += 1
                if lo is None and hi is None:
                    self._parse_stats["failed"] += 1
                    if bucket_source not in self._failed_labels:
                        self._failed_labels.append(bucket_source)
                        logger.warning(f"Unparsed bucket label: '{bucket_source}'")
                else:
                    self._parse_stats["success"] += 1

                # Convert °C to °F if needed (bot forecasts in °F)
                if unit == "C":
                    lo_c, hi_c = lo, hi
                    if lo is not None:
                        lo = _celsius_to_fahrenheit(lo)
                    if hi is not None:
                        hi = _celsius_to_fahrenheit(hi)
                    logger.info(
                        f"Celsius conversion: {bucket_source} "
                        f"({lo_c}-{hi_c}°C → {lo:.0f}-{hi:.0f}°F)"
                        if lo is not None and hi is not None
                        else f"Celsius conversion: {bucket_source} → lo={lo} hi={hi}"
                    )

                # Extract price — in binary markets, outcomePrices[0] is Yes price
                outcome_prices = sib.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    try:
                        outcome_prices = json.loads(outcome_prices)
                    except (json.JSONDecodeError, ValueError):
                        outcome_prices = []

                price_yes = 0.0
                if outcome_prices:
                    try:
                        price_yes = float(outcome_prices[0])
                    except (ValueError, IndexError):
                        price_yes = 0.0

                # The "Yes" token ID for this binary market
                clob_ids = sib.get("clobTokenIds", [])
                if isinstance(clob_ids, str):
                    try:
                        clob_ids = json.loads(clob_ids)
                    except (json.JSONDecodeError, ValueError):
                        clob_ids = []

                token_id = clob_ids[0] if clob_ids else ""

                label = group_title if group_title else f"bucket_{lo}_{hi}"

                mkt.outcomes.append(MarketOutcome(
                    outcome_label=label,
                    token_id=token_id,
                    price_yes=price_yes,
                    price_no=round(1.0 - price_yes, 4),
                    bucket_low=lo,
                    bucket_high=hi,
                ))

            if mkt.outcomes:
                # Sort outcomes by bucket_low for readability
                mkt.outcomes.sort(key=lambda o: (o.bucket_low if o.bucket_low is not None else -999))
                markets.append(mkt)

        # Process standalone markets (old format or single-bucket)
        for item in standalone:
            question = item.get("question", "")
            city = _match_city(question)
            if not city:
                continue
            market_date = _extract_date(question)
            if not market_date:
                continue

            # Detect unit from question or bucket label
            group_title_check = item.get("groupItemTitle", "")
            unit = _detect_unit(question)
            if unit == "F" and group_title_check:
                unit = _detect_unit(group_title_check)

            mkt = TemperatureMarket(
                market_id=str(item.get("id", item.get("conditionId", ""))),
                question=question,
                city=city,
                market_date=market_date,
                resolution_source=item.get("resolutionSource", ""),
                active=item.get("active", True),
                volume=float(item.get("volume", 0) or 0),
                liquidity=float(item.get("liquidity", 0) or 0),
            )

            # For standalone binary markets, parse bucket from groupItemTitle or question
            group_title = item.get("groupItemTitle", "")
            bucket_source = group_title if group_title else question
            lo, hi = _parse_bucket(bucket_source)

            self._parse_stats["total"] += 1
            if lo is None and hi is None:
                self._parse_stats["failed"] += 1
                if bucket_source not in self._failed_labels:
                    self._failed_labels.append(bucket_source)
                    logger.warning(f"Unparsed bucket label: '{bucket_source}'")
            else:
                self._parse_stats["success"] += 1

            if unit == "C":
                if lo is not None:
                    lo = _celsius_to_fahrenheit(lo)
                if hi is not None:
                    hi = _celsius_to_fahrenheit(hi)

            outcome_prices = item.get("outcomePrices", [])
            if isinstance(outcome_prices, str):
                try:
                    outcome_prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, ValueError):
                    outcome_prices = []

            price_yes = float(outcome_prices[0]) if outcome_prices else 0.0
            clob_ids = item.get("clobTokenIds", [])
            if isinstance(clob_ids, str):
                try:
                    clob_ids = json.loads(clob_ids)
                except (json.JSONDecodeError, ValueError):
                    clob_ids = []
            token_id = clob_ids[0] if clob_ids else ""

            label = group_title if group_title else f"bucket_{lo}_{hi}"
            mkt.outcomes.append(MarketOutcome(
                outcome_label=label,
                token_id=token_id,
                price_yes=price_yes,
                price_no=round(1.0 - price_yes, 4),
                bucket_low=lo,
                bucket_high=hi,
            ))

            if mkt.outcomes:
                markets.append(mkt)

        # Log parse stats
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

        logger.info(f"Parsed {len(markets)} active temperature ladders "
                     f"({sum(len(m.outcomes) for m in markets)} total buckets)")
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
