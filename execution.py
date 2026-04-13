"""
execution.py — Module 4: Execution, Logging & Reporting (v3)
=============================================================
v3 changes:
  - Integrates with PositionTracker for real fill tracking.
  - OrderExecutor returns OpenOrder objects instead of bare fill prices.
  - execute_batch feeds actual order IDs to the tracker.
  - Dry-run mode creates synthetic OpenOrders with immediate fills.

Ref: Gemini stress test Fix #1 (fill tracking & exposure management).
"""

import asyncio
import csv
import logging
import math
import os
import uuid
from datetime import datetime, date, timezone
from pathlib import Path
from typing import List, Optional

import aiohttp

from config import cfg
from decision_engine import TradeSignal
from dry_run_simulator import DryRunSimulator, DryRunFillTracker
from position_tracker import OpenOrder, OrderStatus, PositionTracker

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
TRADE_LOG = LOG_DIR / "trades.csv"
SCAN_LOG = LOG_DIR / "scans.csv"


# ── Telegram Alerter ────────────────────────────────────────────────

class TelegramAlerter:
    """Sends alerts via Telegram bot API."""

    def __init__(self):
        self.token = cfg.telegram_token
        self.chat_id = cfg.telegram_chat_id
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            logger.info("Telegram alerting disabled (no token/chat_id)")

    async def send(self, message: str):
        if not self.enabled:
            return
        import aiohttp
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning(f"Telegram send failed: {resp.status}")
        except Exception as e:
            logger.warning(f"Telegram exception: {e}")

    async def trade_alert(self, signal: TradeSignal, order: OpenOrder, dry_run: bool):
        mode = "DRY-RUN" if dry_run else "LIVE"
        fill_info = (
            f"${order.filled_size_usd:.2f} @ {order.avg_fill_price:.3f}"
            if order.status == OrderStatus.FILLED
            else f"pending (order {order.order_id[:12]})"
        )
        msg = (
            f"*{mode}* | *{signal.side}* | {signal.city} {signal.market_date}\n"
            f"Bucket: {signal.outcome_label}\n"
            f"p_true: {signal.p_true:.3f} | mkt: {signal.market_price:.3f}\n"
            f"EV: {signal.ev:.3f} | Edge: {signal.edge:.3f}\n"
            f"Size: ${signal.position_size_usd:.2f} @ limit {signal.price_limit:.3f}\n"
            f"Fill: {fill_info}\n"
            f"_{signal.rationale}_"
        )
        await self.send(msg)

    async def daily_summary(self, daily_pnl: float, trade_count: int,
                             tracker: PositionTracker,
                             resolution_summary: str = "",
                             recent_results: str = ""):
        msg = (
            f"*Daily Summary* ({date.today().isoformat()})\n"
            f"{resolution_summary}\n"
            f"Today's orders: {trade_count}\n"
            f"{tracker.get_exposure_summary()}\n"
            f"Mode: {'LIVE' if cfg.is_live else 'DRY-RUN'}"
        )
        if recent_results:
            msg += f"\n\n*Recent Results:*\n{recent_results}"
        await self.send(msg)

    async def resolution_alert(self, resolved_trades):
        """Send alert when trades are resolved with actual results."""
        if not resolved_trades:
            return
        lines = []
        for t in resolved_trades:
            icon = "+" if t.won else "-"
            lines.append(
                f"  [{icon}] {t.side} {t.bucket_label} {t.city} "
                f"→ {t.actual_high_f:.0f}°F ${t.pnl:+.2f}"
            )
        msg = (
            f"*Trades Resolved* ({len(resolved_trades)})\n"
            + "\n".join(lines)
        )
        await self.send(msg)

    async def fill_update_alert(self, order: OpenOrder):
        msg = (
            f"*Fill Update*: {order.side} {order.outcome_label}\n"
            f"Status: {order.status.value}\n"
            f"Filled: ${order.filled_size_usd:.2f} / ${order.intended_size_usd:.2f}\n"
            f"Avg price: {order.avg_fill_price:.3f}"
        )
        await self.send(msg)

    async def hourly_summary(self, hourly_trades: int, total_trades_today: int,
                              tracker: PositionTracker,
                              resolution_summary: str = "",
                              cycle_count: int = 0):
        """Send hourly trade summary."""
        now = datetime.now(timezone.utc)
        msg = (
            f"*Hourly Update* ({now.strftime('%H:%M')} UTC)\n"
            f"Trades this hour: {hourly_trades}\n"
            f"Trades today: {total_trades_today}\n"
            f"{tracker.get_exposure_summary()}\n"
            f"{resolution_summary}\n"
            f"Cycles: {cycle_count} | Mode: {'LIVE' if cfg.is_live else 'DRY-RUN'}"
        )
        await self.send(msg)

    async def error_alert(self, error_msg: str):
        await self.send(f"*Error*: {error_msg}")

    async def shutdown_alert(self, reason: str):
        await self.send(f"*Bot Shutdown*: {reason}")


# ── Trade Logger ────────────────────────────────────────────────────

class TradeLogger:
    """CSV trade logger — v5: includes slippage, fill_ratio, is_maker, book_depth columns."""

    def __init__(self):
        if not TRADE_LOG.exists():
            with open(TRADE_LOG, "w", newline="") as f:
                csv.writer(f).writerow([
                    "timestamp", "mode", "order_id", "city", "market_date",
                    "outcome", "side", "p_true", "market_price", "ev", "edge",
                    "kelly_frac", "intended_usd", "price_limit",
                    "fill_status", "filled_usd", "avg_fill_price",
                    "slippage", "fill_ratio", "is_maker", "book_depth",
                    "token_id", "market_id", "rationale",
                ])

    def log_trade(
        self,
        signal: TradeSignal,
        order: OpenOrder,
        dry_run: bool,
        slippage: float = 0.0,
        fill_ratio: float = 1.0,
        is_maker: bool = False,
        book_depth: float = 0.0,
    ):
        with open(TRADE_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(),
                "dry-run" if dry_run else "live",
                order.order_id,
                signal.city,
                signal.market_date.isoformat(),
                signal.outcome_label,
                signal.side,
                f"{signal.p_true:.4f}",
                f"{signal.market_price:.4f}",
                f"{signal.ev:.4f}",
                f"{signal.edge:.4f}",
                f"{signal.kelly_fraction:.4f}",
                f"{signal.position_size_usd:.2f}",
                f"{signal.price_limit:.4f}",
                order.status.value,
                f"{order.filled_size_usd:.2f}",
                f"{order.avg_fill_price:.4f}",
                f"{slippage:.4f}",
                f"{fill_ratio:.4f}",
                is_maker,
                f"{book_depth:.2f}",
                signal.token_id,
                signal.market_id,
                signal.rationale,
            ])

    def log_scan(self, num_markets: int, num_matches: int, num_signals: int):
        if not SCAN_LOG.exists():
            with open(SCAN_LOG, "w", newline="") as f:
                csv.writer(f).writerow(["timestamp", "markets_found", "matches", "signals"])
        with open(SCAN_LOG, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now(timezone.utc).isoformat(),
                num_markets, num_matches, num_signals,
            ])


# ── Performance Tracker ─────────────────────────────────────────────

class PerformanceTracker:
    """Tracks risk-adjusted performance metrics (Paper Appendix A)."""

    def __init__(self):
        self._daily_returns: List[float] = []
        self._trade_returns: List[float] = []

    def record_daily_pnl(self, pnl: float, bankroll: float):
        if bankroll > 0:
            self._daily_returns.append(pnl / bankroll)

    def record_trade_return(self, pnl: float, size: float):
        if size > 0:
            self._trade_returns.append(pnl / size)

    @property
    def sharpe_ratio(self) -> float:
        if len(self._daily_returns) < 2:
            return 0.0
        mean_r = sum(self._daily_returns) / len(self._daily_returns)
        variance = sum((r - mean_r) ** 2 for r in self._daily_returns) / (len(self._daily_returns) - 1)
        std_r = math.sqrt(variance) if variance > 0 else 0.001
        return (mean_r / std_r) * math.sqrt(365)

    @property
    def win_rate(self) -> float:
        if not self._trade_returns:
            return 0.0
        return sum(1 for r in self._trade_returns if r > 0) / len(self._trade_returns)

    @property
    def avg_return(self) -> float:
        if not self._trade_returns:
            return 0.0
        return sum(self._trade_returns) / len(self._trade_returns)

    def get_summary(self) -> str:
        return (
            f"Sharpe={self.sharpe_ratio:.2f}, "
            f"WinRate={self.win_rate:.0%}, "
            f"AvgReturn={self.avg_return:.1%}, "
            f"Days={len(self._daily_returns)}, "
            f"Trades={len(self._trade_returns)}"
        )


# ── Order Executor (v3: fill-tracking aware) ──────────────────────

class OrderExecutor:
    """
    Places orders on Polymarket and registers them with the PositionTracker.
    v3: Returns OpenOrder objects, tracks order_ids, supports fill polling.
    v3.1: Pre-flight orderbook depth check before placing orders.
    """

    # Minimum orderbook depth (in shares) required at our price level
    # to avoid placing orders into thin books where we'd wait indefinitely.
    MIN_BOOK_DEPTH = float(os.getenv("MIN_BOOK_DEPTH", "5.0"))

    def __init__(self, tracker: PositionTracker):
        self.client = None
        self.dry_run = not cfg.is_live
        self.tracker = tracker
        self.simulator = DryRunSimulator()
        self.fill_tracker = DryRunFillTracker()

        if cfg.is_live:
            try:
                from py_clob_client.client import ClobClient

                self.client = ClobClient(
                    host=cfg.polymarket_host,
                    key=cfg.private_key,
                    chain_id=cfg.chain_id,
                    funder=cfg.funder if cfg.funder else None,
                )
                self.client.set_api_creds(self.client.create_or_derive_api_creds())
                tracker.set_clob_client(self.client)
                logger.info("ClobClient initialized in LIVE mode")
            except ImportError:
                logger.error("py-clob-client not installed — falling back to dry-run")
                self.dry_run = True
            except Exception as e:
                logger.error(f"ClobClient init failed: {e} — falling back to dry-run")
                self.dry_run = True
        else:
            logger.info("OrderExecutor running in DRY-RUN mode")

    def _check_orderbook_depth(self, signal: TradeSignal) -> bool:
        """
        Pre-flight check: is there sufficient liquidity at our price level?
        Returns True if depth is acceptable or if check is unavailable.
        Ref: Known limitation #4 — order book depth check before placing orders.
        """
        if not self.client or self.dry_run:
            return True  # can't check in dry-run, assume ok

        try:
            book = self.client.get_order_book(signal.token_id)
            if not book:
                logger.debug(f"No orderbook data for {signal.token_id}")
                return True  # no data — proceed cautiously

            # For BUY (Yes), check asks; for SELL (buy No), check bids
            if signal.side == "BUY":
                levels = book.get("asks", [])
            else:
                levels = book.get("bids", [])

            # Sum available size at or near our limit price (within 2¢)
            available = 0.0
            for level in levels:
                level_price = float(level.get("price", 0))
                level_size = float(level.get("size", 0))
                if abs(level_price - signal.price_limit) <= 0.02:
                    available += level_size

            if available < self.MIN_BOOK_DEPTH:
                logger.info(
                    f"Thin orderbook for {signal.outcome_label}: "
                    f"{available:.1f} shares available (min={self.MIN_BOOK_DEPTH}), skipping"
                )
                return False

            return True

        except Exception as e:
            logger.debug(f"Orderbook depth check failed: {e}")
            return True  # fail open — don't block trades on check errors

    def _get_maker_price(self, signal: TradeSignal) -> Optional[float]:
        """
        Attempt to place a passive limit order just inside the spread.
        Returns an adjusted maker price, or None if unavailable.
        Makers pay 0% fees on Polymarket, so pricing at/near the spread
        improves effective EV significantly.
        """
        if self.dry_run or not self.client:
            return None

        try:
            book = self.client.get_order_book(signal.token_id)
            if not book:
                return None

            if signal.side == "BUY":
                bids = book.get("bids", [])
                if not bids:
                    return None
                best_bid = max(float(b.get("price", 0)) for b in bids)
                return min(best_bid + cfg.maker_spread_offset, signal.price_limit)
            else:
                asks = book.get("asks", [])
                if not asks:
                    return None
                best_ask = min(float(a.get("price", 0)) for a in asks)
                return max(best_ask - cfg.maker_spread_offset, signal.price_limit)

        except Exception as e:
            logger.debug(f"Maker price fetch failed: {e}")
            return None

    def _build_open_order(self, signal: TradeSignal, order_id: str) -> OpenOrder:
        return OpenOrder(
            order_id=order_id,
            token_id=signal.token_id,
            market_id=signal.market_id,
            city=signal.city,
            market_date=signal.market_date,
            outcome_label=signal.outcome_label,
            side=signal.side,
            intended_size_usd=signal.position_size_usd,
            limit_price=signal.price_limit,
            submitted_at=datetime.now(timezone.utc),
        )

    async def execute_signal(self, signal: TradeSignal) -> Optional[OpenOrder]:
        """Execute a single trade signal. Returns OpenOrder registered with tracker."""
        if self.dry_run:
            order_id = f"dry-{uuid.uuid4().hex[:12]}"
            order = self._build_open_order(signal, order_id)

            # Fetch real orderbook and simulate realistic fill
            sim_fill = None
            snapshot = None
            try:
                async with aiohttp.ClientSession() as session:
                    snapshot = await self.simulator.fetch_orderbook(session, signal.token_id)
            except Exception as e:
                logger.warning(f"[DRY-RUN] Orderbook fetch error: {e}")

            if snapshot:
                sim_fill = self.simulator.simulate_fill(
                    snapshot, signal.side, signal.position_size_usd,
                    signal.price_limit, is_maker=True
                )
                # Apply simulated fill results
                order.filled_size_usd = sim_fill.filled_size_usd
                order.filled_shares = sim_fill.filled_shares
                order.avg_fill_price = sim_fill.avg_fill_price

                if sim_fill.fill_ratio >= 0.95:
                    order.status = OrderStatus.FILLED
                elif sim_fill.fill_ratio > 0:
                    order.status = OrderStatus.PARTIAL
                else:
                    order.status = OrderStatus.PENDING

                # If maker order with delayed fill, register as pending
                if sim_fill.is_maker and sim_fill.estimated_fill_cycles > 0:
                    self.fill_tracker.register_pending(order_id, sim_fill, order)
                else:
                    self.fill_tracker.record_immediate(sim_fill)
            else:
                # Fallback: conservative estimate (50% fill, 1 cent slippage)
                order.filled_size_usd = signal.position_size_usd * 0.5
                order.filled_shares = (
                    order.filled_size_usd / (signal.price_limit + 0.01)
                    if signal.price_limit > 0 else 0.0
                )
                order.avg_fill_price = signal.price_limit + 0.01
                order.status = OrderStatus.PARTIAL

                # Fix: Register fallback as pending maker so fill_tracker.tick()
                # can eventually clear the remaining 50%. Without this, PARTIAL
                # orders sit in limbo until stale timeout kills them.
                from dry_run_simulator import SimulatedFill
                fallback_fill = SimulatedFill(
                    filled_size_usd=order.filled_size_usd,
                    filled_shares=order.filled_shares,
                    avg_fill_price=order.avg_fill_price,
                    slippage=0.01,
                    fill_ratio=0.5,
                    is_maker=True,
                    estimated_fill_cycles=3,
                )
                self.fill_tracker.register_pending(order_id, fallback_fill, order)

            order.simulated_slippage = sim_fill.slippage if sim_fill else 0.01
            order.simulated_fill_ratio = sim_fill.fill_ratio if sim_fill else 0.5
            order.simulated_is_maker = sim_fill.is_maker if sim_fill else True
            order.simulated_book_depth = (
                sum(s for _, s in snapshot.asks) + sum(s for _, s in snapshot.bids)
                if snapshot else 0.0
            )
            self.tracker.register_order(order)

            # Build descriptive log message with fill quality info
            slippage_str = f"{sim_fill.slippage:.4f}" if sim_fill else "n/a"
            fill_ratio_str = f"{sim_fill.fill_ratio:.2f}" if sim_fill else "0.50"
            maker_str = "maker" if (sim_fill and sim_fill.is_maker) else "taker"
            logger.info(
                f"[DRY-RUN] {signal.side} {signal.outcome_label} "
                f"${signal.position_size_usd:.2f} @ {signal.price_limit:.3f} "
                f"| fill={fill_ratio_str} slip={slippage_str} ({maker_str}) "
                f"| {signal.city} {signal.market_date} (id={order_id})"
            )
            return order

        # Pre-flight: check orderbook depth (v3.1)
        if not self._check_orderbook_depth(signal):
            return None

        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL

            side = BUY if signal.side == "BUY" else SELL
            maker_price = self._get_maker_price(signal)
            price = maker_price if maker_price is not None else signal.price_limit

            # Size calculation differs by side:
            # BUY YES: cost per share = price → shares = usd / price
            # SELL YES (bet NO): risk per share = (1 - price) → shares = usd / (1 - price)
            if signal.side == "SELL":
                size = signal.position_size_usd / (1.0 - price) if price < 1.0 else 0
            else:
                size = signal.position_size_usd / price if price > 0 else 0

            order_args = OrderArgs(
                token_id=signal.token_id,
                price=price,
                size=size,
                side=side,
            )

            signed_order = self.client.create_and_sign_order(order_args)
            resp = self.client.post_order(signed_order, OrderType.GTC)

            if resp and resp.get("success"):
                order_id = resp.get("orderID", resp.get("order_id",
                                    f"live-{uuid.uuid4().hex[:12]}"))
                order = self._build_open_order(signal, order_id)
                order.status = OrderStatus.PENDING
                self.tracker.register_order(order)
                logger.info(
                    f"[LIVE] Order submitted: {signal.side} {signal.outcome_label} "
                    f"${signal.position_size_usd:.2f} @ {price:.3f} "
                    f"| {signal.city} {signal.market_date} (id={order_id})"
                )
                return order
            else:
                logger.warning(f"Order rejected: {resp}")
                return None

        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None

    async def execute_batch(
        self,
        signals: List[TradeSignal],
        telegram: TelegramAlerter,
        trade_logger: TradeLogger,
    ) -> int:
        """Execute all signals, register with tracker, log, and alert."""
        executed = 0
        for signal in signals:
            order = await self.execute_signal(signal)
            if order:
                trade_logger.log_trade(
                    signal, order, self.dry_run,
                    slippage=order.simulated_slippage,
                    fill_ratio=order.simulated_fill_ratio,
                    is_maker=order.simulated_is_maker,
                    book_depth=order.simulated_book_depth,
                )
                # Only send Telegram alerts for high-conviction trades (§ noise filter)
                if signal.ev >= cfg.telegram_min_ev:
                    await telegram.trade_alert(signal, order, self.dry_run)
                executed += 1
            await asyncio.sleep(0.5)
        return executed
