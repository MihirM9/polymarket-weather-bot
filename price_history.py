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

from infrastructure.http import fetch_with_retry

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
