"""
resolution_tracker.py — Market Resolution & P&L Scoring
=========================================================
The missing piece: after a market date passes, fetch the actual high
temperature from NWS station observations, determine which bucket won,
and score each trade as win/loss with real P&L.

Lifecycle:
  1. Trades are logged to trades.csv by execution.py
  2. Each scan cycle, this module checks for unresolved past-date trades
  3. For each city/date needing resolution, fetches the actual observed high
  4. Scores trades: BUY YES wins if actual temp is in the bucket, etc.
  5. Updates resolved_trades.csv with P&L
  6. Feeds cumulative P&L back to the decision engine and Telegram summary
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

from api_utils import fetch_with_retry
from config import cfg

logger = logging.getLogger(__name__)

RESOLVED_LOG = Path("logs") / "resolved_trades.csv"
ACTUAL_TEMPS_LOG = Path("logs") / "actual_temps.csv"

NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {
    "User-Agent": "(polymarket-weather-bot, contact@example.com)",
    "Accept": "application/geo+json",
}


@dataclass
class ResolvedTrade:
    """A trade that has been scored against actual temperature."""
    order_id: str
    city: str
    market_date: date
    bucket_label: str
    bucket_low: Optional[float]
    bucket_high: Optional[float]
    side: str               # "BUY" or "SELL"
    price: float            # entry price
    size_usd: float         # position size
    actual_high_f: float    # observed temperature
    won: bool               # did this trade win?
    pnl: float              # profit/loss in USD


@dataclass
class ResolutionState:
    """Tracks which city/dates have been resolved and cumulative P&L."""
    resolved_dates: Dict[str, float] = field(default_factory=dict)  # "city_date" -> actual_high_f
    resolved_trades: List[ResolvedTrade] = field(default_factory=list)
    total_pnl: float = 0.0
    total_wins: int = 0
    total_losses: int = 0

    @property
    def win_rate(self) -> float:
        total = self.total_wins + self.total_losses
        return self.total_wins / total if total > 0 else 0.0

    @property
    def trade_count(self) -> int:
        return self.total_wins + self.total_losses


class ResolutionTracker:
    """
    Fetches actual temperatures and scores trades after market resolution.

    Uses NWS station observations API to get the actual daily high temperature
    for each city, then compares against each trade's bucket to determine
    win/loss and calculate P&L.
    """

    def __init__(self):
        self.state = ResolutionState()
        self._init_logs()
        self._load_existing_resolutions()

    def _init_logs(self):
        RESOLVED_LOG.parent.mkdir(exist_ok=True)
        if not RESOLVED_LOG.exists():
            with open(RESOLVED_LOG, "w", newline="") as f:
                csv.writer(f).writerow([
                    "resolved_at", "order_id", "city", "market_date",
                    "bucket", "side", "entry_price", "size_usd",
                    "actual_high_f", "won", "pnl",
                ])
        if not ACTUAL_TEMPS_LOG.exists():
            with open(ACTUAL_TEMPS_LOG, "w", newline="") as f:
                csv.writer(f).writerow([
                    "fetched_at", "city", "date", "actual_high_f", "station_id",
                ])

    def _load_existing_resolutions(self):
        """Load previously resolved dates from CSV to avoid re-fetching."""
        if not ACTUAL_TEMPS_LOG.exists():
            return
        try:
            with open(ACTUAL_TEMPS_LOG, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = f"{row['city']}_{row['date']}"
                    self.state.resolved_dates[key] = float(row["actual_high_f"])
        except Exception as e:
            logger.warning(f"Error loading existing resolutions: {e}")

        if not RESOLVED_LOG.exists():
            return
        try:
            with open(RESOLVED_LOG, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    won = row["won"].lower() == "true"
                    pnl = float(row["pnl"])
                    self.state.total_pnl += pnl
                    if won:
                        self.state.total_wins += 1
                    else:
                        self.state.total_losses += 1
        except Exception as e:
            logger.warning(f"Error loading resolved trades: {e}")

        if self.state.trade_count > 0:
            logger.info(
                f"Loaded {self.state.trade_count} resolved trades: "
                f"P&L=${self.state.total_pnl:+.2f}, "
                f"win_rate={self.state.win_rate:.0%}"
            )

    # ── Fetch actual temperatures ─────────────────────────────────────

    async def fetch_actual_high(
        self,
        session: aiohttp.ClientSession,
        city: str,
        target_date: date,
    ) -> Optional[float]:
        """
        Fetch the actual observed daily high temperature for a city/date.

        Uses NWS station observations API to get hourly observations for the
        target date, then takes the maximum temperature as the daily high.
        """
        station_id = cfg.noaa_stations.get(city)
        if not station_id:
            logger.warning(f"No NOAA station configured for {city}")
            return None

        # Fetch observations covering the full local day (midnight to midnight).
        # Convert city local midnight to UTC using the city's offset.
        offset_hours = cfg.CITY_UTC_OFFSETS.get(city, -5)
        # Local midnight = UTC midnight - offset (e.g., EST midnight = 5 AM UTC)
        local_midnight_utc = datetime(
            target_date.year, target_date.month, target_date.day,
            0, 0, tzinfo=timezone.utc
        ) - timedelta(hours=offset_hours)
        start = local_midnight_utc
        end = start + timedelta(hours=24)

        url = (
            f"{NWS_BASE}/stations/{station_id}/observations"
            f"?start={start.isoformat()}&end={end.isoformat()}"
        )

        data = await fetch_with_retry(
            session, url, headers=NWS_HEADERS,
            label=f"NWS-obs-{city}-{target_date}", timeout_sec=20.0,
        )
        if not data:
            return None

        try:
            features = data.get("features", [])
            if not features:
                logger.debug(f"No observations for {city} on {target_date}")
                return None

            # Extract max temperature from all hourly observations
            max_temp_f = None
            for obs in features:
                props = obs.get("properties", {})
                temp_c = props.get("temperature", {}).get("value")
                if temp_c is not None:
                    temp_f = temp_c * 9.0 / 5.0 + 32.0
                    if max_temp_f is None or temp_f > max_temp_f:
                        max_temp_f = temp_f

            if max_temp_f is not None:
                logger.info(f"Actual high for {city} on {target_date}: {max_temp_f:.0f}°F")
                # Log to CSV
                with open(ACTUAL_TEMPS_LOG, "a", newline="") as f:
                    csv.writer(f).writerow([
                        datetime.now(timezone.utc).isoformat(),
                        city, target_date.isoformat(),
                        f"{max_temp_f:.1f}", station_id,
                    ])
            return max_temp_f

        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error parsing observations for {city}: {e}")
            return None

    # ── Trade scoring ─────────────────────────────────────────────────

    @staticmethod
    def _temp_in_bucket(
        temp_f: float,
        bucket_low: Optional[float],
        bucket_high: Optional[float],
    ) -> bool:
        """Check if a temperature falls within a bucket range."""
        if bucket_low is not None and temp_f < bucket_low:
            return False
        if bucket_high is not None and temp_f >= bucket_high:
            return False
        return True

    @staticmethod
    def _calculate_pnl(
        side: str,
        price: float,
        size_usd: float,
        won: bool,
        fee_rate: float = 0.0,  # maker orders = 0% fee
    ) -> float:
        """
        Calculate P&L for a resolved trade.

        BUY YES: pay `price` per share, win → get $1, lose → get $0
        SELL (BUY NO): pay `1-price` per share, win → get $1, lose → get $0

        P&L accounts for partial fills via size_usd (already the filled amount).
        """
        if side == "BUY":
            # Bought YES shares at `price`
            shares = size_usd / price if price > 0 else 0.0
            if won:
                # Each share pays $1, we paid `price` per share
                pnl = shares * (1.0 - price) * (1.0 - fee_rate)
            else:
                # Shares are worthless
                pnl = -size_usd
        else:
            # Bought NO shares at `1 - price`
            price_no = 1.0 - price
            shares = size_usd / price_no if price_no > 0 else 0.0
            if won:
                # NO shares pay $1, we paid `1-price` per share
                pnl = shares * price * (1.0 - fee_rate)
            else:
                pnl = -size_usd

        return round(pnl, 4)

    def score_trade(
        self,
        order_id: str,
        city: str,
        market_date: date,
        bucket_label: str,
        bucket_low: Optional[float],
        bucket_high: Optional[float],
        side: str,
        price: float,
        size_usd: float,
        actual_high_f: float,
    ) -> ResolvedTrade:
        """Score a single trade against the actual temperature."""
        temp_in_bucket = self._temp_in_bucket(actual_high_f, bucket_low, bucket_high)

        # BUY YES wins if temp is in bucket
        # SELL (BUY NO) wins if temp is NOT in bucket
        if side == "BUY":
            won = temp_in_bucket
        else:
            won = not temp_in_bucket

        pnl = self._calculate_pnl(side, price, size_usd, won, fee_rate=cfg.maker_fee_rate)

        resolved = ResolvedTrade(
            order_id=order_id,
            city=city,
            market_date=market_date,
            bucket_label=bucket_label,
            bucket_low=bucket_low,
            bucket_high=bucket_high,
            side=side,
            price=price,
            size_usd=size_usd,
            actual_high_f=actual_high_f,
            won=won,
            pnl=pnl,
        )

        # Log to CSV
        with open(RESOLVED_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(),
                order_id, city, market_date.isoformat(),
                bucket_label, side, f"{price:.4f}", f"{size_usd:.2f}",
                f"{actual_high_f:.1f}", won, f"{pnl:.4f}",
            ])

        # Update state
        self.state.total_pnl += pnl
        if won:
            self.state.total_wins += 1
        else:
            self.state.total_losses += 1
        self.state.resolved_trades.append(resolved)

        outcome = "WIN" if won else "LOSS"
        logger.info(
            f"Resolved: {outcome} {side} {bucket_label} {city} {market_date} | "
            f"actual={actual_high_f:.0f}°F, in_bucket={temp_in_bucket}, "
            f"P&L=${pnl:+.2f} (cumulative=${self.state.total_pnl:+.2f})"
        )

        return resolved

    # ── Batch resolution from trades.csv ──────────────────────────────

    async def resolve_pending_trades(self, trades_csv: Path) -> List[ResolvedTrade]:
        """
        Read trades.csv, find trades on past dates that haven't been resolved,
        fetch actual temps, and score them.

        Returns list of newly resolved trades this cycle.
        """
        if not trades_csv.exists():
            return []

        newly_resolved: List[ResolvedTrade] = []

        # Parse trades.csv to find unresolved trades
        unresolved: List[dict] = []
        resolved_order_ids = {t.order_id for t in self.state.resolved_trades}

        try:
            with open(trades_csv, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    order_id = row.get("order_id", "")
                    if order_id in resolved_order_ids:
                        continue

                    market_date_str = row.get("market_date", "")
                    try:
                        market_date = date.fromisoformat(market_date_str)
                    except ValueError:
                        continue

                    # Only resolve if the market day is fully complete for this city
                    # (past 6 AM local time the next day — ensures all observations are in)
                    city = row.get("city", "")
                    if not cfg.is_market_day_complete(city, market_date):
                        continue

                    unresolved.append(row)
        except Exception as e:
            logger.warning(f"Error reading trades CSV: {e}")
            return []

        if not unresolved:
            return []

        logger.info(f"Found {len(unresolved)} unresolved past-date trades to score")

        # Group by city+date to minimize API calls
        city_dates: Dict[str, date] = {}
        for row in unresolved:
            city = row.get("city", "")
            market_date = date.fromisoformat(row.get("market_date", ""))
            key = f"{city}_{market_date.isoformat()}"
            if key not in self.state.resolved_dates:
                city_dates[key] = market_date

        # Fetch actual temps for unresolved city/dates
        if city_dates:
            async with aiohttp.ClientSession() as session:
                for key, market_date in city_dates.items():
                    city = key.rsplit("_", 1)[0]
                    actual = await self.fetch_actual_high(session, city, market_date)
                    if actual is not None:
                        self.state.resolved_dates[key] = actual

        # Score each unresolved trade
        for row in unresolved:
            city = row.get("city", "")
            market_date = date.fromisoformat(row.get("market_date", ""))
            key = f"{city}_{market_date.isoformat()}"

            actual_high = self.state.resolved_dates.get(key)
            if actual_high is None:
                continue  # couldn't fetch actual temp, skip for now

            # Parse bucket bounds from outcome label
            outcome_label = row.get("outcome", "")
            bucket_low, bucket_high = self._parse_bucket_from_label(outcome_label)

            # Get trade details
            order_id = row.get("order_id", "")
            side = row.get("side", "BUY")
            price = float(row.get("price_limit", row.get("market_price", "0")))

            # Use intended_usd as the position size, NOT filled_usd.
            # filled_usd is written at order placement time and is $0 for maker
            # orders that fill later. The CSV is never updated after fills.
            # For P&L scoring, intended_usd represents the actual risk.
            size_usd = float(row.get("intended_usd", "0"))

            if size_usd <= 0:
                continue  # no position to score

            resolved = self.score_trade(
                order_id=order_id,
                city=city,
                market_date=market_date,
                bucket_label=outcome_label,
                bucket_low=bucket_low,
                bucket_high=bucket_high,
                side=side,
                price=price,
                size_usd=size_usd,
                actual_high_f=actual_high,
            )
            newly_resolved.append(resolved)

        return newly_resolved

    @staticmethod
    def _parse_bucket_from_label(label: str) -> Tuple[Optional[float], Optional[float]]:
        """Parse bucket bounds from an outcome label stored in trades.csv."""
        # Import the parser's bucket parsing logic
        from polymarket_parser import _parse_bucket
        return _parse_bucket(label)

    # ── Summary ───────────────────────────────────────────────────────

    def get_pnl_summary(self) -> str:
        """Human-readable P&L summary for Telegram/logs."""
        s = self.state
        if s.trade_count == 0:
            return "P&L: No resolved trades yet"
        return (
            f"P&L: ${s.total_pnl:+.2f} | "
            f"W/L: {s.total_wins}/{s.total_losses} ({s.win_rate:.0%}) | "
            f"Trades: {s.trade_count}"
        )

    def get_recent_results(self, n: int = 5) -> str:
        """Last N resolved trades for Telegram summary."""
        if not self.state.resolved_trades:
            return "No resolved trades"
        recent = self.state.resolved_trades[-n:]
        lines = []
        for t in recent:
            emoji = "W" if t.won else "L"
            lines.append(
                f"  {emoji} {t.side} {t.bucket_label} {t.city} {t.market_date} "
                f"→ actual {t.actual_high_f:.0f}°F ${t.pnl:+.2f}"
            )
        return "\n".join(lines)
