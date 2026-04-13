"""
dry_run_simulator.py — Realistic Dry-Run Order Matching Engine
===============================================================
Instead of assuming perfect instant fills, this module:
  1. Fetches real L2 orderbook from Polymarket CLOB API
  2. Simulates matching against actual liquidity
  3. Models partial fills, slippage, and time-to-fill
  4. Tracks fill rate metrics for evaluating strategy viability

This makes dry-run results trustworthy for deciding whether to deploy real capital.
"""

import csv
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import aiohttp

from infrastructure.http import fetch_with_retry

from .positions import OpenOrder, OrderStatus

logger = logging.getLogger(__name__)

CLOB_BASE_URL = "https://clob.polymarket.com"
DRY_RUN_FILL_LOG = Path("logs") / "dry_run_fills.csv"


# ── Data classes ───────────────────────────────────────────────────


@dataclass
class OrderbookSnapshot:
    """Point-in-time snapshot of L2 orderbook for a single token."""
    token_id: str
    bids: List[Tuple[float, float]]  # (price, size) sorted best-first (highest bid first)
    asks: List[Tuple[float, float]]  # (price, size) sorted best-first (lowest ask first)
    timestamp: datetime

    @property
    def mid_price(self) -> float:
        """Mid-market price. Returns 0.5 if either side is empty."""
        best_bid = self.bids[0][0] if self.bids else 0.0
        best_ask = self.asks[0][0] if self.asks else 1.0
        if not self.bids and not self.asks:
            return 0.5
        if not self.bids:
            return best_ask
        if not self.asks:
            return best_bid
        return (best_bid + best_ask) / 2.0

    @property
    def spread(self) -> float:
        """Bid-ask spread. Returns 1.0 if either side is empty."""
        best_bid = self.bids[0][0] if self.bids else 0.0
        best_ask = self.asks[0][0] if self.asks else 1.0
        return best_ask - best_bid


@dataclass
class SimulatedFill:
    """Result of simulating an order against the real orderbook."""
    filled_size_usd: float      # USDC actually matched
    filled_shares: float        # shares received
    avg_fill_price: float       # volume-weighted average fill price
    slippage: float             # difference from requested limit price (positive = worse)
    fill_ratio: float           # filled / intended (0.0 to 1.0)
    is_maker: bool              # whether the order would rest on the book
    estimated_fill_cycles: int  # how many 2-min cycles until fill (0 = immediate taker)


@dataclass
class _PendingSimOrder:
    """Internal tracking for a simulated maker order awaiting fill."""
    order_id: str
    sim_fill: SimulatedFill
    order: OpenOrder
    cycles_remaining: int
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ── Simulator ──────────────────────────────────────────────────────


class DryRunSimulator:
    """
    Simulates realistic order matching against the live Polymarket orderbook.
    Even in dry-run mode, we fetch real orderbook data to see what would happen.
    """

    def __init__(self) -> None:
        self._base_url: str = CLOB_BASE_URL
        self._init_fill_log()

    def _init_fill_log(self) -> None:
        DRY_RUN_FILL_LOG.parent.mkdir(exist_ok=True)
        if not DRY_RUN_FILL_LOG.exists():
            with open(DRY_RUN_FILL_LOG, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "token_id", "side", "is_maker",
                    "intended_usd", "filled_usd", "filled_shares",
                    "limit_price", "avg_fill_price", "slippage",
                    "fill_ratio", "estimated_fill_cycles",
                    "book_spread", "book_mid",
                ])

    def _log_fill(
        self,
        token_id: str,
        side: str,
        intended_usd: float,
        limit_price: float,
        sim: SimulatedFill,
        snapshot: Optional["OrderbookSnapshot"],
    ) -> None:
        """Append simulated fill details to CSV log."""
        spread = snapshot.spread if snapshot else 0.0
        mid = snapshot.mid_price if snapshot else 0.0
        try:
            with open(DRY_RUN_FILL_LOG, "a", newline="") as f:
                csv.writer(f).writerow([
                    datetime.now(timezone.utc).isoformat(),
                    token_id,
                    side,
                    sim.is_maker,
                    f"{intended_usd:.2f}",
                    f"{sim.filled_size_usd:.2f}",
                    f"{sim.filled_shares:.4f}",
                    f"{limit_price:.4f}",
                    f"{sim.avg_fill_price:.4f}",
                    f"{sim.slippage:.4f}",
                    f"{sim.fill_ratio:.4f}",
                    sim.estimated_fill_cycles,
                    f"{spread:.4f}",
                    f"{mid:.4f}",
                ])
        except Exception as e:
            logger.warning(f"Failed to log dry-run fill: {e}")

    # ── Orderbook fetch ────────────────────────────────────────────

    async def fetch_orderbook(
        self,
        session: aiohttp.ClientSession,
        token_id: str,
    ) -> Optional[OrderbookSnapshot]:
        """
        Fetch real L2 orderbook from Polymarket CLOB API.
        Returns None if the fetch fails (API down, etc.).
        """
        url = f"{self._base_url}/book?token_id={token_id}"
        data = await fetch_with_retry(
            session, url, label="dry-run-book", timeout_sec=10.0, max_retries=2
        )
        if not data:
            logger.warning(f"Orderbook fetch failed for {token_id[:16]}...")
            return None

        try:
            raw_bids = data.get("bids", [])
            raw_asks = data.get("asks", [])

            # Parse and sort: bids highest-first, asks lowest-first
            bids: List[Tuple[float, float]] = sorted(
                [(float(b["price"]), float(b["size"])) for b in raw_bids],
                key=lambda x: x[0],
                reverse=True,
            )
            asks: List[Tuple[float, float]] = sorted(
                [(float(a["price"]), float(a["size"])) for a in raw_asks],
                key=lambda x: x[0],
            )

            snapshot = OrderbookSnapshot(
                token_id=token_id,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc),
            )
            logger.debug(
                f"Orderbook fetched: {len(bids)} bids, {len(asks)} asks, "
                f"mid={snapshot.mid_price:.3f}, spread={snapshot.spread:.4f}"
            )
            return snapshot

        except (KeyError, ValueError, TypeError) as e:
            logger.warning(f"Orderbook parse error for {token_id[:16]}: {e}")
            return None

    # ── Fill simulation ────────────────────────────────────────────

    def simulate_fill(
        self,
        snapshot: OrderbookSnapshot,
        side: str,
        intended_size_usd: float,
        limit_price: float,
        is_maker: bool = True,
    ) -> SimulatedFill:
        """
        Simulate filling an order against the real orderbook.

        For BUY side:
          - Taker: walks the asks (we lift offers)
          - Maker: rests a bid and waits for someone to hit it
        For SELL side (buying No / selling Yes):
          - Taker: walks the bids (we hit bids)
          - Maker: rests an ask and waits for someone to lift it

        Conservative defaults: when in doubt, assume worse outcomes.
        """
        if intended_size_usd <= 0 or limit_price <= 0:
            return SimulatedFill(
                filled_size_usd=0.0, filled_shares=0.0, avg_fill_price=0.0,
                slippage=0.0, fill_ratio=0.0, is_maker=is_maker,
                estimated_fill_cycles=0,
            )

        # Determine if this order would cross the spread (taker) or rest (maker)
        if side == "BUY":
            best_ask = snapshot.asks[0][0] if snapshot.asks else 1.0
            would_cross = limit_price >= best_ask
        else:
            best_bid = snapshot.bids[0][0] if snapshot.bids else 0.0
            would_cross = limit_price <= best_bid

        # If is_maker=True but the price would cross, treat as taker
        # (the CLOB would match it immediately)
        if is_maker and not would_cross:
            # SELL YES: risk per share = (1 - price), so shares = usd / (1 - price)
            # BUY YES: cost per share = price, so shares = usd / price
            if side == "SELL":
                intended_shares = intended_size_usd / (1.0 - limit_price) if limit_price < 1.0 else 0.0
            else:
                intended_shares = intended_size_usd / limit_price if limit_price > 0 else 0.0
            sim = self._estimate_maker_fill(
                snapshot, side, limit_price, intended_size_usd, intended_shares
            )
        else:
            sim = self._simulate_taker_fill(
                snapshot, side, intended_size_usd, limit_price
            )

        self._log_fill(snapshot.token_id, side, intended_size_usd, limit_price, sim, snapshot)
        return sim

    def _simulate_taker_fill(
        self,
        snapshot: OrderbookSnapshot,
        side: str,
        intended_size_usd: float,
        limit_price: float,
    ) -> SimulatedFill:
        """
        Simulate aggressive (taker) fill by walking the opposing side of the book.
        For BUY: walk asks from lowest to highest.
        For SELL: walk bids from highest to lowest.
        """
        # Select opposing side
        if side == "BUY":
            levels = snapshot.asks  # sorted lowest-first
        else:
            levels = snapshot.bids  # sorted highest-first

        if not levels:
            return SimulatedFill(
                filled_size_usd=0.0, filled_shares=0.0,
                avg_fill_price=limit_price, slippage=0.0,
                fill_ratio=0.0, is_maker=False, estimated_fill_cycles=0,
            )

        remaining_usd = intended_size_usd
        total_shares = 0.0
        total_cost = 0.0

        for level_price, level_size_shares in levels:
            # Check limit: for BUY, don't pay more than limit; for SELL, don't accept less
            if side == "BUY" and level_price > limit_price:
                break
            if side == "SELL" and level_price < limit_price:
                break

            # Available USD at this level
            available_usd = level_size_shares * level_price
            fill_usd = min(remaining_usd, available_usd)
            fill_shares = fill_usd / level_price if level_price > 0 else 0.0

            total_shares += fill_shares
            total_cost += fill_usd
            remaining_usd -= fill_usd

            if remaining_usd <= 0.001:  # effectively zero
                break

        filled_size_usd = total_cost
        avg_price = total_cost / total_shares if total_shares > 0 else limit_price
        fill_ratio = filled_size_usd / intended_size_usd if intended_size_usd > 0 else 0.0

        # Slippage: how much worse than our limit price
        if side == "BUY":
            slippage = max(0.0, avg_price - limit_price)
        else:
            slippage = max(0.0, limit_price - avg_price)

        return SimulatedFill(
            filled_size_usd=filled_size_usd,
            filled_shares=total_shares,
            avg_fill_price=avg_price,
            slippage=slippage,
            fill_ratio=min(1.0, fill_ratio),
            is_maker=False,
            estimated_fill_cycles=0,  # taker fills are immediate
        )

    def _estimate_maker_fill(
        self,
        snapshot: OrderbookSnapshot,
        side: str,
        limit_price: float,
        intended_size_usd: float,
        intended_shares: float,
    ) -> SimulatedFill:
        """
        Estimate fill probability and timing for a passive limit (maker) order.

        Conservative model:
          - Queue position: how much volume is ahead of us at our price level
          - Fill probability: based on how competitive our price is vs best bid/ask
          - Time estimate: number of 2-min cycles until expected fill

        We intentionally err on the pessimistic side — better to underestimate
        dry-run performance than to overestimate it.
        """
        if side == "BUY":
            # Our bid sits among other bids. We need asks to come down to us.
            best_bid = snapshot.bids[0][0] if snapshot.bids else 0.0
            best_ask = snapshot.asks[0][0] if snapshot.asks else 1.0
            spread = best_ask - best_bid

            # How competitive is our bid? (distance from best ask as fraction of spread)
            if spread > 0:
                competitiveness = max(0.0, 1.0 - (best_ask - limit_price) / spread)
            else:
                competitiveness = 0.5

            # Volume ahead of us in the queue at our price level
            queue_volume = sum(
                size for price, size in snapshot.bids if price >= limit_price
            )
        else:
            # Our ask sits among other asks. We need bids to come up to us.
            best_bid = snapshot.bids[0][0] if snapshot.bids else 0.0
            best_ask = snapshot.asks[0][0] if snapshot.asks else 1.0
            spread = best_ask - best_bid

            if spread > 0:
                competitiveness = max(0.0, 1.0 - (limit_price - best_bid) / spread)
            else:
                competitiveness = 0.5

            queue_volume = sum(
                size for price, size in snapshot.asks if price <= limit_price
            )

        # Conservative fill probability model:
        #   - Base probability from competitiveness (how close to crossing)
        #   - Penalized by queue depth (more volume ahead = lower probability)
        #   - Capped conservatively — never assume > 70% fill for maker orders
        base_fill_prob = competitiveness * 0.6  # max 60% from price alone

        # Queue penalty: more volume ahead of us means less chance we get filled
        if queue_volume > 0 and intended_shares > 0:
            queue_factor = min(1.0, intended_shares / (queue_volume + intended_shares))
        else:
            queue_factor = 0.5  # conservative default

        fill_prob = min(0.70, base_fill_prob * queue_factor)

        # No artificial floor — broken orders should show 0% fill, not 5%

        # Estimate fill cycles (inverse of probability, with floor)
        # More competitive orders fill faster
        if fill_prob > 0.5:
            estimated_cycles = 1
        elif fill_prob > 0.3:
            estimated_cycles = 3
        elif fill_prob > 0.15:
            estimated_cycles = 5
        else:
            estimated_cycles = 8  # ~16 minutes — might not fill at all

        # Simulated fill amount (probabilistic partial fill)
        filled_shares = intended_shares * fill_prob
        # SELL YES: collateral per share = (1 - price), BUY YES: cost = price
        if side == "SELL":
            filled_usd = filled_shares * (1.0 - limit_price)
        else:
            filled_usd = filled_shares * limit_price

        # Maker orders fill at limit price (no slippage), but we add a tiny
        # conservative buffer for price uncertainty
        slippage = 0.001  # 0.1 cent conservative buffer

        fill_ratio = fill_prob

        logger.debug(
            f"Maker fill estimate: side={side}, price={limit_price:.3f}, "
            f"competitiveness={competitiveness:.2f}, queue={queue_volume:.1f}, "
            f"fill_prob={fill_prob:.2f}, cycles={estimated_cycles}"
        )

        return SimulatedFill(
            filled_size_usd=filled_usd,
            filled_shares=filled_shares,
            avg_fill_price=limit_price + (slippage if side == "BUY" else -slippage),
            slippage=slippage,
            fill_ratio=min(1.0, fill_ratio),
            is_maker=True,
            estimated_fill_cycles=estimated_cycles,
        )


# ── Fill Tracker (pending maker orders) ───────────────────────────


class DryRunFillTracker:
    """
    Tracks pending simulated maker orders and resolves them over time.

    Each cycle, pending orders tick down their estimated_fill_cycles counter.
    When it reaches zero, the order is considered "filled" and returned to
    the caller for position tracking updates.
    """

    def __init__(self) -> None:
        self._pending: Dict[str, _PendingSimOrder] = {}
        self._metrics: Dict[str, int] = {
            "total_orders": 0,
            "fully_filled": 0,
            "partially_filled": 0,
            "unfilled": 0,
        }
        self._slippages: List[float] = []
        self._fill_ratios: List[float] = []

    def register_pending(
        self,
        order_id: str,
        sim_fill: SimulatedFill,
        order: OpenOrder,
    ) -> None:
        """Store a pending maker order for future fill simulation."""
        self._pending[order_id] = _PendingSimOrder(
            order_id=order_id,
            sim_fill=sim_fill,
            order=order,
            cycles_remaining=max(1, sim_fill.estimated_fill_cycles),
        )
        self._metrics["total_orders"] += 1
        logger.info(
            f"[DRY-RUN] Registered pending maker order {order_id[:16]}: "
            f"~{sim_fill.estimated_fill_cycles} cycles to fill, "
            f"fill_ratio={sim_fill.fill_ratio:.2f}"
        )

    def record_immediate(self, sim_fill: SimulatedFill) -> None:
        """Record metrics for an immediately resolved order (taker or instant maker)."""
        self._metrics["total_orders"] += 1
        self._slippages.append(sim_fill.slippage)
        self._fill_ratios.append(sim_fill.fill_ratio)
        if sim_fill.fill_ratio >= 0.95:
            self._metrics["fully_filled"] += 1
        elif sim_fill.fill_ratio > 0:
            self._metrics["partially_filled"] += 1
        else:
            self._metrics["unfilled"] += 1

    def tick(self, cycle_count: int) -> List[OpenOrder]:
        """
        Advance pending maker orders by one cycle.
        Returns list of orders that have now "filled".
        """
        newly_filled: List[OpenOrder] = []
        completed_ids: List[str] = []

        for order_id, pending in self._pending.items():
            pending.cycles_remaining -= 1

            if pending.cycles_remaining <= 0:
                # Order has "filled" — apply the simulated fill to the order
                order = pending.order
                sim = pending.sim_fill

                order.filled_size_usd = sim.filled_size_usd
                order.filled_shares = sim.filled_shares
                order.avg_fill_price = sim.avg_fill_price

                if sim.fill_ratio >= 0.95:
                    order.status = OrderStatus.FILLED
                    self._metrics["fully_filled"] += 1
                elif sim.fill_ratio > 0:
                    order.status = OrderStatus.PARTIAL
                    self._metrics["partially_filled"] += 1
                else:
                    order.status = OrderStatus.CANCELLED
                    self._metrics["unfilled"] += 1

                self._slippages.append(sim.slippage)
                self._fill_ratios.append(sim.fill_ratio)
                newly_filled.append(order)
                completed_ids.append(order_id)

        # Remove completed orders from pending
        for oid in completed_ids:
            del self._pending[oid]

        return newly_filled

    def get_metrics(self) -> Dict[str, float]:
        """Return fill rate statistics for evaluating strategy viability."""
        avg_slippage = (
            sum(self._slippages) / len(self._slippages)
            if self._slippages else 0.0
        )
        avg_fill_ratio = (
            sum(self._fill_ratios) / len(self._fill_ratios)
            if self._fill_ratios else 0.0
        )
        return {
            "total_orders": self._metrics["total_orders"],
            "fully_filled": self._metrics["fully_filled"],
            "partially_filled": self._metrics["partially_filled"],
            "unfilled": self._metrics["unfilled"],
            "pending": len(self._pending),
            "avg_slippage": avg_slippage,
            "avg_fill_ratio": avg_fill_ratio,
        }

    def get_summary(self) -> str:
        """Human-readable summary of dry-run fill metrics."""
        m = self.get_metrics()
        return (
            f"DryRunFills: total={m['total_orders']}, "
            f"full={m['fully_filled']}, partial={m['partially_filled']}, "
            f"unfilled={m['unfilled']}, pending={m['pending']}, "
            f"avg_slip={m['avg_slippage']:.4f}, avg_fill={m['avg_fill_ratio']:.2f}"
        )
