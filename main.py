"""
main.py — Polymarket Weather Trading Bot v5
=============================================
v5 changes:
  - PositionTracker: polls CLOB for fill status, reconciles real exposure.
  - EnsembleBlender: blends NWS + OpenWeatherMap, dynamic sigma from model spread.
  - Decision engine uses tracker.total_exposure for accurate risk caps.
  - Resolution tracking and dashboard export are integrated into the main loop.
  - Scan cycle now: poll_fills → forecast → ensemble_blend → parse → decide → execute.

Architecture (v5):
  ┌─────────────────┐
  │ Position Tracker │◄──── poll CLOB for fill status (start of each cycle)
  └────────┬────────┘
           │ realized_exposure
  ┌────────▼────────┐     ┌──────────────────┐
  │ Forecast Scanner│────▶│ Ensemble Blender  │──── OWM API (supplemental)
  │ (NWS/NOAA)      │     │ (NWS + OWM blend) │
  └─────────────────┘     └────────┬─────────┘
                                   │ blended_high, ensemble_sigma
                          ┌────────▼─────────┐
                          │ Polymarket Parser │
                          │ (Gamma API)       │
                          └────────┬─────────┘
                                   │ matched (market, forecast) pairs
                          ┌────────▼─────────┐
                          │ Decision Engine   │◄── tracker.total_exposure
                          │ (EV/Kelly/Risk)   │
                          └────────┬─────────┘
                                   │ [TradeSignal]
                          ┌────────▼─────────┐
                          │ Execution         │───▶ PositionTracker.register()
                          │ (CLOB/Telegram)   │
                          └──────────────────┘
"""

import asyncio
import logging
import signal
from dataclasses import dataclass
from datetime import date as date_type
from datetime import datetime, timezone
from typing import Optional

from config import cfg
from forecasting import EnsembleBlender, ForecastingService, ForecastScanner, MetarFetcher
from infrastructure.health import HealthMonitor
from infrastructure.io import default_io_manager
from infrastructure.logging import configure_logging
from trading.decision import DecisionEngine
from trading.execution import OrderExecutor, PerformanceTracker, TelegramAlerter, TradeLogger
from trading.markets import PolymarketParser
from trading.positions import OrderStatus, PositionTracker
from trading.resolution import ResolutionTracker

# ── Logging ──

logger = logging.getLogger("main")
log_listener = None

# ── Graceful shutdown ──

shutdown_event = asyncio.Event()

def _handle_signal(sig, frame):
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


@dataclass
class ScanCycleResult:
    executed: int
    market_count: int = 0
    match_count: int = 0
    signal_count: int = 0
    parse_failure_rate: float = 0.0


def _tick_order_state(executor: OrderExecutor, tracker: PositionTracker, cycle_count: int) -> None:
    """Advance open-order state before evaluating fresh signals."""
    if not executor.dry_run:
        return

    newly_filled = executor.fill_tracker.tick(cycle_count)
    applied_fills = tracker.apply_dry_run_fill_tick(newly_filled)
    for order in newly_filled:
        logger.info(
            f"[DRY-RUN] Simulated maker fill: {order.outcome_label} "
            f"${order.filled_size_usd:.2f} @ {order.avg_fill_price:.3f}"
        )
    if applied_fills:
        logger.info(
            f"Dry-run fill tick: {len(newly_filled)} order(s) filled. "
            f"{executor.fill_tracker.get_summary()}"
        )


def _build_dashboard_state(
    tracker: PositionTracker,
    engine: DecisionEngine,
    resolution_tracker: ResolutionTracker,
    trade_logger: TradeLogger,
    cycle_count: int,
    start_time: datetime,
) -> dict:
    orders = list(tracker._orders.values())
    positions = [
        {
            "order_id": o.order_id,
            "city": o.city,
            "date": o.market_date.isoformat(),
            "bucket": o.outcome_label,
            "side": o.side,
            "size": o.intended_size_usd,
            "entry_price": o.limit_price,
            "current_price": o.avg_fill_price or o.limit_price,
            "filled_usd": o.filled_size_usd,
            "status": o.status.value,
        }
        for o in orders if o.status.value not in ("cancelled", "failed")
    ]
    pending = [
        {
            "order_id": o.order_id,
            "city": o.city,
            "date": o.market_date.isoformat(),
            "bucket": o.outcome_label,
            "side": o.side,
            "size": o.intended_size_usd,
            "limit_price": o.limit_price,
            "age_seconds": o.age_seconds,
            "status": o.status.value,
        }
        for o in orders if o.status == OrderStatus.PENDING
    ]

    city_exp: dict[str, float] = {}
    for order in orders:
        if not order.is_terminal:
            city_exp[order.city] = city_exp.get(order.city, 0.0) + order.filled_size_usd + order.unfilled_usd

    today = date_type.today()
    total_deployed = sum(
        order.filled_size_usd for order in orders
        if order.filled_size_usd > 0
        and order.status not in (OrderStatus.CANCELLED, OrderStatus.FAILED)
        and order.market_date >= today
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "live" if cfg.is_live else "dry-run",
        "cycle": cycle_count,
        "uptime_seconds": (datetime.now(timezone.utc) - start_time).total_seconds(),
        "bankroll": cfg.bankroll,
        "daily_pnl": resolution_tracker.state.total_pnl,
        "daily_loss_cap_used": abs(engine.daily_pnl) / cfg.daily_loss_cap if cfg.daily_loss_cap > 0 else 0,
        "total_exposure": tracker.total_exposure,
        "total_deployed": round(total_deployed, 2),
        "realized_exposure": tracker.realized_exposure,
        "pending_exposure": tracker.pending_exposure,
        "positions": positions,
        "pending_orders": pending,
        "recent_trades": trade_logger.recent_trades,
        "exposure_by_city": city_exp,
        "resolution": {
            "total_pnl": resolution_tracker.state.total_pnl,
            "wins": resolution_tracker.state.total_wins,
            "losses": resolution_tracker.state.total_losses,
            "win_rate": resolution_tracker.state.win_rate,
            "trade_count": resolution_tracker.state.trade_count,
        },
        "adverse_selection": {
            "instant_fill_rate": tracker.adverse_selection_rate,
            "avg_fill_speed": tracker.avg_fill_speed,
        },
        "fill_stats": {
            "total": tracker._total_fills,
            "active": tracker.active_order_count,
            "filled": tracker.filled_order_count,
        },
    }


def export_dashboard_state(
    tracker: PositionTracker,
    engine: DecisionEngine,
    resolution_tracker: ResolutionTracker,
    trade_logger: TradeLogger,
    cycle_count: int,
    start_time: datetime,
) -> bool:
    """Queue a dashboard snapshot write for the HTTP dashboard."""
    try:
        state = _build_dashboard_state(
            tracker,
            engine,
            resolution_tracker,
            trade_logger,
            cycle_count,
            start_time,
        )
        default_io_manager.write_json_atomic(STATE_FILE, state)
        return True
    except Exception as e:
        logger.warning(f"Dashboard state export failed: {e}")
        return False

async def run_scan_cycle(
    scanner: ForecastScanner,
    forecasting_service: ForecastingService,
    parser: PolymarketParser,
    engine: DecisionEngine,
    executor: OrderExecutor,
    tracker: PositionTracker,
    telegram: TelegramAlerter,
    trade_logger: TradeLogger,
    resolution_tracker: ResolutionTracker,
    cycle_count: int = 0,
) -> ScanCycleResult:
    cycle_start = datetime.now(timezone.utc)
    logger.info(f"=== Scan cycle start: {cycle_start.isoformat()} (cycle #{cycle_count}) ===")

    # Step 0: Poll fills on open orders (v3 Fix #1)
    if not executor.dry_run:
        fill_changes = await tracker.poll_fills()
        if fill_changes:
            logger.info(f"Fill poll: {fill_changes} order(s) updated. {tracker.get_exposure_summary()}")
    else:
        _tick_order_state(executor, tracker, cycle_count)
    logger.info(f"Exposure state: {tracker.get_exposure_summary()}")

    # Step 1: Check daily loss cap
    if engine.is_shutdown():
        await telegram.shutdown_alert(f"Daily loss cap breached: PnL={engine.daily_pnl:+.2f}")
        return ScanCycleResult(executed=0)

    # Step 2: Fetch NWS forecasts
    logger.info("Step 1/5: Fetching NWS forecasts...")
    forecasts = await scanner.scan_all()
    if not forecasts:
        logger.warning("No forecasts retrieved — skipping cycle")
        return ScanCycleResult(executed=0)

    # Step 3: Fetch supplemental forecasts and blend (v3 Fix #2)
    logger.info("Step 2/5: Blending with ensemble forecasts...")
    await forecasting_service.enrich_forecasts(forecasts)

    # Step 4: Discover & parse markets
    logger.info("Step 3/5: Discovering temperature markets...")
    markets = await parser.fetch_temperature_markets()
    if not markets:
        logger.info("No active temperature markets found")
        trade_logger.log_scan(0, 0, 0)
        return ScanCycleResult(executed=0, parse_failure_rate=parser.parse_failure_rate)

    matches = parser.match_forecasts(markets, forecasts)
    if not matches:
        logger.info("No market-forecast matches")
        trade_logger.log_scan(len(markets), 0, 0)
        return ScanCycleResult(
            executed=0,
            market_count=len(markets),
            parse_failure_rate=parser.parse_failure_rate,
        )

    # Step 5: Decision engine (v3: with tracker for accurate exposure)
    logger.info("Step 4/5: Evaluating EV and sizing trades...")
    signals = engine.evaluate(matches, tracker=tracker)
    trade_logger.log_scan(len(markets), len(matches), len(signals))

    if not signals:
        logger.info("No actionable signals this cycle")
        return ScanCycleResult(
            executed=0,
            market_count=len(markets),
            match_count=len(matches),
            signal_count=0,
            parse_failure_rate=parser.parse_failure_rate,
        )

    logger.info(f"Generated {len(signals)} trade signals:")
    for i, sig in enumerate(signals[:10]):
        logger.info(f"  #{i+1}: {sig.rationale}")

    # Step 6: Execute
    logger.info("Step 5/5: Executing trades...")
    executed = await executor.execute_batch(signals, telegram, trade_logger)

    elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
    logger.info(f"=== Cycle complete: {executed}/{len(signals)} trades in {elapsed:.1f}s ===")
    return ScanCycleResult(
        executed=executed,
        market_count=len(markets),
        match_count=len(matches),
        signal_count=len(signals),
        parse_failure_rate=parser.parse_failure_rate,
    )


STATE_FILE = "/tmp/bot_state.json"


# ── Main ──

async def main():
    global log_listener
    if log_listener is None:
        log_listener = configure_logging("logs/bot.log")
    logger.info("=" * 60)
    logger.info("  Polymarket Weather Trading Bot v5")
    logger.info(f"  Mode: {'LIVE' if cfg.is_live else 'DRY-RUN'}")
    logger.info(f"  Bankroll: ${cfg.bankroll:.0f}")
    logger.info(f"  Cities: {', '.join(cfg.cities)}")
    logger.info(f"  Scan interval: {cfg.scan_interval_sec}s")
    logger.info(f"  Ensemble blending: {'enabled' if EnsembleBlender().enabled else 'disabled (no OWM key)'}")
    logger.info("  Fill tracking: enabled")
    logger.info("  Resolution tracking: enabled")
    logger.info("=" * 60)

    # Initialize all components
    scanner = ForecastScanner()
    blender = EnsembleBlender()
    metar_fetcher = MetarFetcher()
    forecasting_service = ForecastingService(scanner, blender, metar_fetcher)
    parser = PolymarketParser()
    engine = DecisionEngine()
    tracker = PositionTracker()
    executor = OrderExecutor(tracker)
    telegram = TelegramAlerter()
    trade_logger = TradeLogger()
    perf_tracker = PerformanceTracker()
    resolution_tracker = ResolutionTracker()
    health_monitor = HealthMonitor()

    await telegram.send(
        f"*Bot Started v5* ({'LIVE' if cfg.is_live else 'DRY-RUN'})\n"
        f"Bankroll: ${cfg.bankroll:.0f}\n"
        f"Ensemble: {'NWS+OWM+METAR' if blender.enabled else 'NWS+METAR'}\n"
        f"Resolution tracking: active\n"
        f"{resolution_tracker.get_pnl_summary()}\n"
        f"Scan every {cfg.scan_interval_sec}s"
    )

    total_trades = 0
    hourly_trades = 0
    cycle_count = 0
    last_summary_date: Optional[date_type] = None
    last_hourly_summary: datetime = datetime.now(timezone.utc)
    start_time = datetime.now(timezone.utc)

    while not shutdown_event.is_set():
        try:
            cycle_count += 1
            cycle_result = await run_scan_cycle(
                scanner, forecasting_service, parser, engine, executor,
                tracker, telegram, trade_logger,
                resolution_tracker,
                cycle_count=cycle_count,
            )
            total_trades += cycle_result.executed
            hourly_trades += cycle_result.executed
            health_monitor.record_cycle_success()

            if cycle_result.parse_failure_rate > 0.35:
                logger.warning(
                    f"High bucket parse failure rate this cycle: {cycle_result.parse_failure_rate:.0%}"
                )

            # Hourly summary: every 60 minutes
            now = datetime.now(timezone.utc)
            if (now - last_hourly_summary).total_seconds() >= 3600:
                await telegram.hourly_summary(
                    hourly_trades, total_trades, tracker,
                    resolution_summary=resolution_tracker.get_pnl_summary(),
                    cycle_count=cycle_count,
                )
                logger.info(f"Hourly summary sent: {hourly_trades} trades this hour")
                hourly_trades = 0
                last_hourly_summary = now

            # Resolution check: every 10 cycles (~20 min), score past-date trades
            if cycle_count % 10 == 0:
                try:
                    from trading.execution import TRADE_LOG
                    newly_resolved = await resolution_tracker.resolve_pending_trades(TRADE_LOG)
                    if newly_resolved:
                        # Feed real P&L into decision engine
                        for t in newly_resolved:
                            engine.update_pnl(t.pnl)
                            perf_tracker.record_trade_return(t.pnl, t.size_usd)
                        # Send Telegram alert for resolved trades
                        await telegram.resolution_alert(newly_resolved)
                        logger.info(
                            f"Resolved {len(newly_resolved)} trades this cycle. "
                            f"{resolution_tracker.get_pnl_summary()}"
                        )
                except Exception as e:
                    logger.warning(f"Resolution check error: {e}")

            # Daily summary: once per day (when the date changes)
            today = date_type.today()
            if last_summary_date != today and cycle_count > 1:
                last_summary_date = today
                perf_tracker.record_daily_pnl(
                    resolution_tracker.state.total_pnl, cfg.bankroll
                )
                await telegram.daily_summary(
                    resolution_tracker.state.total_pnl,
                    total_trades,
                    tracker,
                    resolution_summary=resolution_tracker.get_pnl_summary(),
                    recent_results=resolution_tracker.get_recent_results(5),
                )
                logger.info(f"Performance: {perf_tracker.get_summary()}")
                # Reset daily trade counter
                total_trades = 0

            dashboard_ok = export_dashboard_state(
                tracker, engine, resolution_tracker,
                trade_logger, cycle_count, start_time,
            )
            health_monitor.record_dashboard_export(dashboard_ok)

        except Exception as e:
            logger.exception(f"Scan cycle error: {e}")
            health_monitor.record_cycle_failure(str(e))
            await telegram.error_alert(str(e))

        fail_safe_reason = health_monitor.fail_safe_reason()
        if fail_safe_reason:
            logger.error(f"Fail-safe shutdown triggered: {fail_safe_reason}")
            await telegram.shutdown_alert(fail_safe_reason)
            shutdown_event.set()
            break

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=cfg.scan_interval_sec)
            break
        except asyncio.TimeoutError:
            pass

    logger.info("Bot shutting down...")
    default_io_manager.flush()
    await telegram.send(
        f"*Bot Stopped v5*\n"
        f"Session trades: {total_trades}\n"
        f"{resolution_tracker.get_pnl_summary()}\n"
        f"{tracker.get_exposure_summary()}\n"
        f"{perf_tracker.get_summary()}"
    )
    default_io_manager.flush()
    if log_listener is not None:
        log_listener.stop()

if __name__ == "__main__":
    asyncio.run(main())
