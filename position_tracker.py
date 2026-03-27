"""
position_tracker.py — v3 Fix #1: Fill Tracking & Exposure Management
=====================================================================
Addresses the critical "flying blind" vulnerability identified in stress testing:
the bot previously assumed all GTC limit orders filled at limit price immediately.
In reality, fills may be partial, delayed, or never happen, causing the risk
engine's exposure tracking to drift from reality.

This module provides:
  1. OpenOrder: tracks each submitted order with its CLOB order_id.
  2. PositionTracker: polls the CLOB for fill status, reconciles realized
     vs pending exposure, and feeds accurate numbers back to the decision engine.
  3. Stale order cancellation: orders unfilled after a configurable TTL are cancelled.

The decision engine now uses realized_exposure (what actually filled) instead of
intended_exposure (what we hoped would fill) for all risk cap calculations.
"""

import asyncio
import csv
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime, date, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from config import cfg

logger = logging.getLogger(__name__)

POSITION_LOG = Path("logs") / "positions.csv"


class OrderStatus(Enum):
    PENDING = "pending"       # Submitted, no fill info yet
    PARTIAL = "partial"       # Partially filled
    FILLED = "filled"         # Fully filled
    CANCELLED = "cancelled"   # Cancelled (by us or expired)
    FAILED = "failed"         # Submission failed


@dataclass
class OpenOrder:
    """Tracks a single submitted order through its lifecycle."""
    order_id: str                     # CLOB order ID returned on submission
    token_id: str
    market_id: str
    city: str
    market_date: date
    outcome_label: str
    side: str                         # "BUY" or "SELL"
    intended_size_usd: float          # What we asked for
    limit_price: float
    submitted_at: datetime
    status: OrderStatus = OrderStatus.PENDING

    # Fill tracking
    filled_size_usd: float = 0.0      # Actual USDC filled
    filled_shares: float = 0.0        # Actual shares received
    avg_fill_price: float = 0.0       # Weighted average fill price
    last_checked: Optional[datetime] = None

    @property
    def unfilled_usd(self) -> float:
        return max(0.0, self.intended_size_usd - self.filled_size_usd)

    @property
    def is_terminal(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED)

    @property
    def age_seconds(self) -> float:
        return (datetime.now(timezone.utc) - self.submitted_at).total_seconds()


class PositionTracker:
    """
    Tracks all open orders and reconciles actual vs intended exposure.

    Key properties exposed to the decision engine:
      - realized_exposure: sum of actually-filled USDC across all orders today
      - pending_exposure: sum of unfilled portions of open orders
      - total_exposure: realized + pending (conservative upper bound)
    """

    # Cancel unfilled orders after this many seconds (default: 10 min)
    ORDER_TTL_SEC = int(600)

    # Number of scan cycles to suppress re-ordering after a stale cancel (§7.1)
    # 3 cycles × 2 min = 6 min cooldown prevents infinite cancel-replace loops
    COOLDOWN_CYCLES = 3

    def __init__(self):
        self._orders: Dict[str, OpenOrder] = {}   # order_id → OpenOrder
        self._today: Optional[date] = None
        self._daily_realized: float = 0.0
        self._daily_pending: float = 0.0
        self._clob_client = None
        # Cooldown tracker: "market_id:outcome_label" → remaining cycles (§7.1)
        # Prevents cancel-replace loops by suppressing re-orders after stale cancels
        self._cancel_cooldowns: Dict[str, int] = {}
        self._fill_speeds: List[float] = []
        self._instant_fill_count: int = 0
        self._total_fills: int = 0

        self._init_log()

    def _init_log(self):
        if not POSITION_LOG.exists():
            with open(POSITION_LOG, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "order_id", "status", "city", "market_date",
                    "outcome", "side", "intended_usd", "filled_usd",
                    "filled_shares", "avg_fill_price", "age_sec",
                ])

    def _log_order(self, order: OpenOrder):
        with open(POSITION_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(),
                order.order_id,
                order.status.value,
                order.city,
                order.market_date.isoformat(),
                order.outcome_label,
                order.side,
                f"{order.intended_size_usd:.2f}",
                f"{order.filled_size_usd:.2f}",
                f"{order.filled_shares:.4f}",
                f"{order.avg_fill_price:.4f}",
                f"{order.age_seconds:.0f}",
            ])

    def _reset_daily_if_needed(self):
        today = date.today()
        if self._today != today:
            # Archive yesterday's terminal orders, keep active ones
            self._orders = {
                oid: o for oid, o in self._orders.items() if not o.is_terminal
            }
            self._daily_realized = 0.0
            self._daily_pending = 0.0
            self._cancel_cooldowns.clear()
            self._fill_speeds.clear()
            self._instant_fill_count = 0
            self._total_fills = 0
            self._today = today

    def set_clob_client(self, client):
        """Set the ClobClient reference for polling order status."""
        self._clob_client = client

    def register_order(self, order: OpenOrder):
        """Called by executor after successfully submitting an order."""
        self._reset_daily_if_needed()
        self._orders[order.order_id] = order
        self._daily_pending += order.intended_size_usd
        logger.info(f"Registered order {order.order_id}: {order.side} {order.outcome_label} "
                     f"${order.intended_size_usd:.2f} @ {order.limit_price:.3f}")

    def register_dry_run_fill(self, order: OpenOrder):
        """For dry-run mode: immediately mark as filled at limit price.

        .. deprecated::
            Use DryRunSimulator + register_order() for realistic dry-run fills.
            This method assumes perfect instant fills at limit price, which
            produces unrealistically optimistic results.
        """
        warnings.warn(
            "register_dry_run_fill() is deprecated — use DryRunSimulator + "
            "register_order() for realistic dry-run simulation",
            DeprecationWarning,
            stacklevel=2,
        )
        self._reset_daily_if_needed()
        order.status = OrderStatus.FILLED
        order.filled_size_usd = order.intended_size_usd
        order.filled_shares = order.intended_size_usd / order.limit_price if order.limit_price > 0 else 0
        order.avg_fill_price = order.limit_price
        self._orders[order.order_id] = order
        self._daily_realized += order.filled_size_usd
        self._log_order(order)

    async def poll_fills(self) -> int:
        """
        Poll CLOB for fill status on all pending/partial orders.
        Returns number of orders whose status changed.

        This is called at the start of each scan cycle so the decision engine
        has accurate exposure numbers before evaluating new trades.
        """
        self._reset_daily_if_needed()
        self.tick_cooldowns()

        if not self._clob_client:
            return 0

        changed = 0
        now = datetime.now(timezone.utc)

        for order_id, order in list(self._orders.items()):
            if order.is_terminal:
                continue

            try:
                # Query CLOB for order status
                resp = self._clob_client.get_order(order_id)
                if not resp:
                    continue

                order.last_checked = now

                # Parse fill data from response
                # py-clob-client returns: status, size_matched, price, etc.
                clob_status = resp.get("status", "").lower()
                size_matched = float(resp.get("size_matched", 0) or 0)
                original_size = float(resp.get("original_size", 0) or 0)
                avg_price = float(resp.get("associate_trades_avg_price",
                                  resp.get("price", order.limit_price)) or order.limit_price)

                prev_filled = order.filled_size_usd
                order.filled_shares = size_matched
                order.avg_fill_price = avg_price
                order.filled_size_usd = size_matched * avg_price

                if clob_status in ("matched", "filled", "closed"):
                    order.status = OrderStatus.FILLED
                    changed += 1
                    fill_time = order.age_seconds
                    self._fill_speeds.append(fill_time)
                    self._total_fills += 1
                    if fill_time < 10:
                        self._instant_fill_count += 1
                        logger.warning(
                            f"Instant fill ({fill_time:.1f}s) for {order.outcome_label} — "
                            f"possible adverse selection"
                        )
                elif size_matched > 0 and clob_status in ("live", "open"):
                    order.status = OrderStatus.PARTIAL
                    changed += 1
                elif clob_status in ("cancelled", "expired"):
                    order.status = OrderStatus.CANCELLED
                    changed += 1

                # Update realized exposure delta
                fill_delta = order.filled_size_usd - prev_filled
                if fill_delta > 0:
                    self._daily_realized += fill_delta
                    self._daily_pending -= fill_delta
                    logger.info(f"Fill update: {order.order_id} → {order.status.value}, "
                                f"filled=${order.filled_size_usd:.2f}/{order.intended_size_usd:.2f}")

                self._log_order(order)

            except Exception as e:
                logger.warning(f"Fill poll error for {order_id}: {e}")

        # Cancel stale unfilled orders
        stale_cancelled = await self._cancel_stale_orders()
        changed += stale_cancelled

        self._recalculate_pending()
        return changed

    async def _cancel_stale_orders(self) -> int:
        """Cancel orders that have been pending/partial too long without filling."""
        cancelled = 0
        for order_id, order in list(self._orders.items()):
            # Cancel both PENDING and PARTIAL orders that are stale.
            # Partial orders have unfilled portions that sit on the book
            # indefinitely if not cleaned up.
            if order.status not in (OrderStatus.PENDING, OrderStatus.PARTIAL):
                continue
            if order.age_seconds > self.ORDER_TTL_SEC:
                try:
                    if self._clob_client:
                        self._clob_client.cancel(order_id)
                    order.status = OrderStatus.CANCELLED
                    self._log_order(order)
                    cancelled += 1
                    # Register cooldown to prevent cancel-replace loop (§7.1)
                    cooldown_key = f"{order.market_id}:{order.outcome_label}"
                    self._cancel_cooldowns[cooldown_key] = self.COOLDOWN_CYCLES
                    logger.info(f"Cancelled stale order {order_id} (age={order.age_seconds:.0f}s), "
                                f"cooldown={self.COOLDOWN_CYCLES} cycles for {cooldown_key}")
                except Exception as e:
                    logger.warning(f"Failed to cancel stale order {order_id}: {e}")
        return cancelled

    def is_cooled_down(self, market_id: str, outcome_label: str) -> bool:
        """
        Returns True if the given market/outcome is still in post-cancel cooldown.

        The decision engine should check this before placing a new order to avoid
        the cancel-replace loop where stale orders are cancelled and immediately
        re-issued at the same price every cycle (§7.1).
        """
        cooldown_key = f"{market_id}:{outcome_label}"
        remaining = self._cancel_cooldowns.get(cooldown_key, 0)
        if remaining > 0:
            logger.debug(f"Cooldown active for {cooldown_key}: {remaining} cycles remaining")
            return True
        return False

    def tick_cooldowns(self):
        """
        Decrement all cooldown counters by 1 and remove expired entries.

        Called once per scan cycle from poll_fills() so cooldowns automatically
        expire after COOLDOWN_CYCLES iterations (§7.1).
        """
        expired_keys: List[str] = []
        for key in self._cancel_cooldowns:
            self._cancel_cooldowns[key] -= 1
            if self._cancel_cooldowns[key] <= 0:
                expired_keys.append(key)
        for key in expired_keys:
            del self._cancel_cooldowns[key]
            logger.debug(f"Cooldown expired for {key}")

    def _recalculate_pending(self):
        """Recalculate pending exposure from active orders."""
        self._daily_pending = sum(
            o.unfilled_usd for o in self._orders.values()
            if not o.is_terminal
        )

    # ── Exposure properties for the decision engine ──

    @property
    def realized_exposure(self) -> float:
        """Actually-filled USDC today. This is ground truth."""
        self._reset_daily_if_needed()
        return self._daily_realized

    @property
    def pending_exposure(self) -> float:
        """Unfilled portions of open orders. May or may not fill."""
        self._reset_daily_if_needed()
        return self._daily_pending

    @property
    def total_exposure(self) -> float:
        """Conservative upper bound: realized + pending."""
        return self.realized_exposure + self.pending_exposure

    @property
    def active_order_count(self) -> int:
        return sum(1 for o in self._orders.values() if not o.is_terminal)

    @property
    def filled_order_count(self) -> int:
        return sum(1 for o in self._orders.values()
                   if o.status == OrderStatus.FILLED)

    @property
    def adverse_selection_rate(self) -> float:
        if self._total_fills == 0:
            return 0.0
        return self._instant_fill_count / self._total_fills

    @property
    def avg_fill_speed(self) -> float:
        if not self._fill_speeds:
            return 0.0
        return sum(self._fill_speeds) / len(self._fill_speeds)

    def has_active_order(self, market_id: str, outcome_label: str) -> bool:
        """Check if there's already ANY order (active OR filled) for this market+bucket.

        Prevents duplicate entries: once we've traded a bucket, don't re-enter it.
        This fixes dry-run mode where filled orders become terminal immediately,
        causing the dedup check to miss them and place duplicate trades.
        """
        for order in self._orders.values():
            if (order.market_id == market_id
                    and order.outcome_label == outcome_label
                    and order.status != OrderStatus.CANCELLED
                    and order.status != OrderStatus.FAILED):
                return True
        return False

    def get_exposure_summary(self) -> str:
        adv_sel = f", adv_sel={self.adverse_selection_rate:.0%}" if self._total_fills > 0 else ""
        fill_spd = f", avg_fill={self.avg_fill_speed:.0f}s" if self._fill_speeds else ""
        return (
            f"Exposure: realized=${self.realized_exposure:.2f}, "
            f"pending=${self.pending_exposure:.2f}, "
            f"total=${self.total_exposure:.2f}, "
            f"active_orders={self.active_order_count}, "
            f"filled={self.filled_order_count}"
            f"{adv_sel}{fill_spd}"
        )
