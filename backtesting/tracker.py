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
        self._active_orders: Set[str] = set()
        self._daily_exposure: float = 0.0

    def can_trade(self, market_id: str, outcome_label: str, size: float, city: str) -> bool:
        """Check if a trade is allowed under all risk caps."""
        key = f"{market_id}:{outcome_label}"
        if key in self._active_orders:
            return False
        if size > MAX_POSITION_USD:
            return False
        city_exp = self._city_exposure.get(city, 0.0)
        if city_exp + size > self._per_market_cap:
            return False
        group_cap = self._per_market_cap * CORRELATED_GROUP_CAP_MULT
        for group_name, group_cities in CORRELATION_GROUPS.items():
            if city in group_cities:
                group_exp = self._group_exposure.get(group_name, 0.0)
                if group_exp + size > group_cap:
                    return False
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
        return False

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
