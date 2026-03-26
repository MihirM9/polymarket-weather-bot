"""
analyze_trades.py — Post-hoc trade analysis with actual NWS observations
Reads logs/trades.csv, fetches observed highs, determines win/loss, prints summary.
Usage: python3 analyze_trades.py
"""
import asyncio, csv, logging, re, sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiohttp
from api_utils import fetch_with_retry

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TRADE_LOG = Path(__file__).parent / "logs" / "trades.csv"
NWS_HEADERS = {"User-Agent": "(polymarket-weather-bot, contact@example.com)",
               "Accept": "application/geo+json"}

CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "new york": (40.7789, -73.9692), "chicago": (41.9742, -87.9073),
    "los angeles": (33.9425, -118.4081), "miami": (25.7959, -80.2870),
    "houston": (29.9844, -95.3414), "dallas": (32.8998, -97.0403),
}
_station_cache: Dict[str, str] = {}


@dataclass
class Trade:
    timestamp: str; mode: str; city: str; market_date: date; outcome: str
    side: str; p_true: float; market_price: float; ev: float
    price_limit: float; filled_usd: float
    bucket_lo: Optional[float] = None; bucket_hi: Optional[float] = None
    actual_temp: Optional[float] = None; win: Optional[bool] = None
    pnl: Optional[float] = None


def parse_bucket(outcome: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract (lo, hi) bounds from bucket labels like '70° to 74°F' or '70-74'."""
    m = re.search(r"(\d+)\s*°?\s*F?\s*to\s*(\d+)\s*°?\s*F?", outcome, re.I)
    if m: return float(m.group(1)), float(m.group(2))
    m = re.search(r"(\d+)\s*[-\u2013]\s*(\d+)", outcome)
    if m: return float(m.group(1)), float(m.group(2))
    # Unbounded high: "75°F or higher" / "above 75"
    m = re.search(r"(\d+)\s*°?\s*F?\s*or\s*(higher|above|more)", outcome, re.I)
    if not m:
        m = re.search(r"(above|over|higher than)\s*(\d+)", outcome, re.I)
        if m: return float(m.group(2)), 200.0
    if m and m.group(1).isdigit(): return float(m.group(1)), 200.0
    # Unbounded low: "below 65" / "64°F or lower"
    m = re.search(r"(\d+)\s*°?\s*F?\s*or\s*(lower|below|less)", outcome, re.I)
    if not m:
        m = re.search(r"(below|under|lower than)\s*(\d+)", outcome, re.I)
        if m: return -50.0, float(m.group(2))
    if m and m.group(1).isdigit(): return -50.0, float(m.group(1))
    logger.warning(f"Could not parse bucket: {outcome}")
    return None, None


def load_trades() -> List[Trade]:
    if not TRADE_LOG.exists():
        logger.error(f"Trade log not found: {TRADE_LOG}"); sys.exit(1)
    trades: List[Trade] = []
    with open(TRADE_LOG, newline="") as f:
        for row in csv.DictReader(f):
            filled = float(row.get("filled_usd", 0))
            if filled <= 0: continue
            lo, hi = parse_bucket(row["outcome"])
            trades.append(Trade(
                row["timestamp"], row["mode"], row["city"],
                date.fromisoformat(row["market_date"]), row["outcome"],
                row["side"], float(row["p_true"]), float(row["market_price"]),
                float(row["ev"]), float(row["price_limit"]), filled, lo, hi))
    return trades


def match_city(city: str) -> Optional[Tuple[float, float]]:
    cl = city.lower().strip()
    for name, coords in CITY_COORDS.items():
        if name in cl or cl in name: return coords
    return None


async def get_station(session: aiohttp.ClientSession, city: str) -> Optional[str]:
    """Get the nearest NWS observation station for a city (cached)."""
    if city.lower() in _station_cache: return _station_cache[city.lower()]
    coords = match_city(city)
    if not coords: return None
    data = await fetch_with_retry(
        session, f"https://api.weather.gov/points/{coords[0]:.4f},{coords[1]:.4f}",
        headers=NWS_HEADERS, label="nws-points")
    if not data: return None
    stations_url = data.get("properties", {}).get("observationStations")
    if not stations_url: return None
    sdata = await fetch_with_retry(session, stations_url, headers=NWS_HEADERS, label="nws-stations")
    if not sdata or not sdata.get("features"): return None
    sid = sdata["features"][0]["properties"]["stationIdentifier"]
    _station_cache[city.lower()] = sid
    return sid


async def get_observed_high(session: aiohttp.ClientSession, station: str,
                            obs_date: date) -> Optional[float]:
    """Fetch the observed daily high temperature (°F) from NWS observations."""
    start = datetime(obs_date.year, obs_date.month, obs_date.day, 6, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=24)
    url = (f"https://api.weather.gov/stations/{station}/observations"
           f"?start={start.isoformat()}&end={end.isoformat()}")
    data = await fetch_with_retry(session, url, headers=NWS_HEADERS, label="nws-obs")
    if not data: return None
    max_f: Optional[float] = None
    for feat in data.get("features", []):
        temp = feat.get("properties", {}).get("temperature", {})
        val, unit = temp.get("value"), temp.get("unitCode", "")
        if val is None: continue
        tf = val * 9 / 5 + 32 if "degC" in unit else val
        if max_f is None or tf > max_f: max_f = tf
    return round(max_f, 1) if max_f is not None else None


def evaluate_trade(t: Trade) -> None:
    if t.actual_temp is None or t.bucket_lo is None or t.bucket_hi is None: return
    in_bucket = t.bucket_lo <= t.actual_temp <= t.bucket_hi
    if t.side == "BUY":
        t.win = in_bucket
        t.pnl = (t.filled_usd / t.price_limit - t.filled_usd) if t.win else -t.filled_usd
    else:  # SELL No — win if NOT in bucket
        t.win = not in_bucket
        t.pnl = (t.filled_usd / (1 - t.price_limit) - t.filled_usd) if t.win else -t.filled_usd


async def main() -> None:
    trades = load_trades()
    if not trades:
        print("No filled trades found in log."); return
    today = date.today()
    resolved = [t for t in trades if t.market_date < today]
    unresolved = [t for t in trades if t.market_date >= today]
    print(f"\nLoaded {len(trades)} filled trades "
          f"({len(resolved)} resolved, {len(unresolved)} pending)\n")
    if not resolved:
        print("No resolved trades to analyze."); return

    # Fetch actual observations
    async with aiohttp.ClientSession() as session:
        for t in resolved:
            station = await get_station(session, t.city)
            if not station:
                logger.warning(f"No station for {t.city}"); continue
            t.actual_temp = await get_observed_high(session, station, t.market_date)
            if t.actual_temp is None:
                logger.warning(f"No obs data for {t.city} on {t.market_date}")

    for t in resolved: evaluate_trade(t)

    # Print table
    hdr = (f"{'City':<14} {'Date':<12} {'Bucket':<16} {'Side':<5} "
           f"{'Price':>6} {'Actual':>7} {'Result':>7} {'P&L':>8} {'EV':>6}")
    print(hdr); print("-" * len(hdr))
    wins, losses, total_pnl, total_ev, evaluated = 0, 0, 0.0, 0.0, 0

    for t in resolved:
        bkt = (f"{int(t.bucket_lo)}-{int(t.bucket_hi)}" if t.bucket_lo is not None
               and t.bucket_hi is not None and t.bucket_hi < 200 else t.outcome[:15])
        act = f"{t.actual_temp:.0f}F" if t.actual_temp is not None else "N/A"
        if t.win is not None:
            res, pnl_s = ("WIN" if t.win else "LOSS"), f"${t.pnl:+.2f}"
            evaluated += 1
            if t.win: wins += 1
            else: losses += 1
            total_pnl += t.pnl or 0; total_ev += t.ev * t.filled_usd
        else:
            res, pnl_s = "???", "N/A"
        print(f"{t.city:<14} {t.market_date!s:<12} {bkt:<16} {t.side:<5} "
              f"{t.price_limit:>5.2f}c {act:>7} {res:>7} {pnl_s:>8} {t.ev:>5.1%}")

    # Summary
    print(f"\n{'=' * 60}\nSUMMARY\n{'=' * 60}")
    print(f"  Total trades loaded:    {len(trades)}")
    print(f"  Resolved (past date):   {len(resolved)}")
    print(f"  Successfully evaluated: {evaluated}")
    print(f"  Unresolved (future):    {len(unresolved)}")
    if evaluated > 0:
        cost_basis = sum(t.filled_usd for t in resolved if t.win is not None)
        print(f"  Wins / Losses:          {wins} / {losses}  ({wins/evaluated*100:.1f}% win rate)")
        print(f"  Total P&L:              ${total_pnl:+.2f}")
        print(f"  Avg P&L per trade:      ${total_pnl/evaluated:+.2f}")
        print(f"  Avg expected EV:        {total_ev/cost_basis*100:+.1f}%" if cost_basis else "")
        print(f"  Actual return:          {total_pnl/cost_basis*100:+.1f}%" if cost_basis else "")
    print()


if __name__ == "__main__":
    asyncio.run(main())
