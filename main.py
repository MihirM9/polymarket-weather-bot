"""
main.py — Polymarket Weather Trading Bot v3
=============================================
v3 changes:
  - PositionTracker: polls CLOB for fill status, reconciles real exposure.
  - EnsembleBlender: blends NWS + OpenWeatherMap, dynamic sigma from model spread.
  - Decision engine uses tracker.total_exposure for accurate risk caps.
  - Scan cycle now: poll_fills → forecast → ensemble_blend → parse → decide → execute.

Architecture (v3):
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
import csv
import json
import logging
import os
import signal
import sys
from datetime import datetime, date as date_type, timezone
from pathlib import Path

import aiohttp

from config import cfg
from forecast_scanner import ForecastScanner, CityForecast, compute_confidence
from polymarket_parser import PolymarketParser
from decision_engine import DecisionEngine
from ensemble_blender import EnsembleBlender, ForecastPoint
from position_tracker import PositionTracker, OrderStatus
from execution import OrderExecutor, TelegramAlerter, TradeLogger, PerformanceTracker
from metar_fetcher import MetarFetcher
from resolution_tracker import ResolutionTracker

# ── Logging ──

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/bot.log", mode="a"),
    ],
)
logger = logging.getLogger("main")

# ── Graceful shutdown ──

shutdown_event = asyncio.Event()

def _handle_signal(sig, frame):
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    shutdown_event.set()

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ── Scan cycle (v3) ──

async def run_scan_cycle(
    scanner: ForecastScanner,
    blender: EnsembleBlender,
    parser: PolymarketParser,
    engine: DecisionEngine,
    executor: OrderExecutor,
    tracker: PositionTracker,
    telegram: TelegramAlerter,
    trade_logger: TradeLogger,
    metar_fetcher: MetarFetcher,
    resolution_tracker: ResolutionTracker,
    cycle_count: int = 0,
) -> int:
    cycle_start = datetime.now(timezone.utc)
    logger.info(f"=== Scan cycle start: {cycle_start.isoformat()} (cycle #{cycle_count}) ===")

    # Step 0: Poll fills on open orders (v3 Fix #1)
    if not executor.dry_run:
        fill_changes = await tracker.poll_fills()
        if fill_changes:
            logger.info(f"Fill poll: {fill_changes} order(s) updated. {tracker.get_exposure_summary()}")
    else:
        # Dry-run: tick simulated maker fills forward by one cycle
        newly_filled = executor.fill_tracker.tick(cycle_count)
        for order in newly_filled:
            tracker._daily_realized += order.filled_size_usd
            logger.info(
                f"[DRY-RUN] Simulated maker fill: {order.outcome_label} "
                f"${order.filled_size_usd:.2f} @ {order.avg_fill_price:.3f}"
            )
        tracker._recalculate_pending()
        if newly_filled:
            tracker._save_state()
            logger.info(f"Dry-run fill tick: {len(newly_filled)} order(s) filled. "
                        f"{executor.fill_tracker.get_summary()}")
    logger.info(f"Exposure state: {tracker.get_exposure_summary()}")

    # Step 1: Check daily loss cap
    if engine.is_shutdown():
        await telegram.shutdown_alert(f"Daily loss cap breached: PnL={engine.daily_pnl:+.2f}")
        return 0

    # Step 2: Fetch NWS forecasts
    logger.info("Step 1/5: Fetching NWS forecasts...")
    forecasts = await scanner.scan_all()
    if not forecasts:
        logger.warning("No forecasts retrieved — skipping cycle")
        return 0

    # Step 3: Fetch supplemental forecasts and blend (v3 Fix #2)
    logger.info("Step 2/5: Blending with ensemble forecasts...")
    cities_dates = [(fc.city, fc.forecast_date) for fc in forecasts]
    owm_forecasts = await blender.fetch_all_supplemental(cities_dates)

    # Fetch METAR observations (v4: third ensemble source)
    metar_observations = await metar_fetcher.fetch_all()

    # Fetch NOAA station observations for same-day markets (v4: station-specific)
    # Use city-local date, not UTC date, to avoid "night blindness" where
    # the UTC date advances at 8 PM EST / 4 PM PST
    station_observations: dict = {}
    same_day_cities = set(
        fc.city for fc in forecasts
        if fc.forecast_date == cfg.city_local_date(fc.city)
    )
    if same_day_cities:
        async with aiohttp.ClientSession() as session:
            for city in same_day_cities:
                station_id = cfg.noaa_stations.get(city)
                if station_id:
                    temp_f = await scanner.fetch_station_observation(session, station_id)
                    if temp_f is not None:
                        station_observations[city] = temp_f

    # Apply ensemble blending to each NWS forecast
    for fc in forecasts:
        key = f"{fc.city}_{fc.forecast_date.isoformat()}"
        owm_point = owm_forecasts.get(key)
        supplemental = [owm_point] if owm_point else []

        # For same-day forecasts, use METAR and station observations as a floor
        # and sigma signal — NOT as daily high forecasts to average.
        # Current temp is an instantaneous reading (e.g., 60°F at 8 AM),
        # not a prediction of the daily high (e.g., 88°F). Averaging them
        # would systematically drag the forecast toward current conditions,
        # producing catastrophically wrong predictions during cool mornings
        # and evenings.
        if fc.forecast_date == cfg.city_local_date(fc.city):
            # Collect current observations
            current_temps = []
            metar_obs = metar_observations.get(fc.city)
            if metar_obs:
                current_temps.append(metar_obs.temp_f)
            station_temp = station_observations.get(fc.city)
            if station_temp is not None:
                current_temps.append(station_temp)

            if current_temps:
                max_observed = max(current_temps)

                # Floor: if current temp already exceeds forecast, raise it
                if max_observed > fc.high_f:
                    logger.info(
                        f"Observed temp {max_observed:.0f}°F > forecast "
                        f"{fc.high_f:.0f}°F for {fc.city} — raising floor"
                    )
                    fc.high_f = max_observed
                    fc.sigma = max(fc.sigma, 1.5)  # widen sigma, forecast was wrong

                # Sigma signal: large divergence means more uncertainty
                divergence = abs(fc.high_f - max_observed)
                if divergence > 10:
                    sigma_boost = 1.0 + (divergence - 10) * 0.05
                    fc.sigma *= sigma_boost
                    logger.info(
                        f"Large obs/forecast divergence ({divergence:.0f}°F) "
                        f"for {fc.city} — σ boosted to {fc.sigma:.1f}"
                    )

        ensemble = blender.blend(fc.high_f, fc.sigma, supplemental)

        # Update forecast with blended values
        fc.high_f = ensemble.blended_high
        fc.sigma = ensemble.ensemble_sigma
        # Recompute confidence with new sigma
        fc.confidence = compute_confidence(fc.sigma, fc.is_stable, fc.regime_multiplier)

    # Step 4: Discover & parse markets
    logger.info("Step 3/5: Discovering temperature markets...")
    markets = await parser.fetch_temperature_markets()
    if not markets:
        logger.info("No active temperature markets found")
        trade_logger.log_scan(0, 0, 0)
        return 0

    matches = parser.match_forecasts(markets, forecasts)
    if not matches:
        logger.info("No market-forecast matches")
        trade_logger.log_scan(len(markets), 0, 0)
        return 0

    # Step 5: Decision engine (v3: with tracker for accurate exposure)
    logger.info("Step 4/5: Evaluating EV and sizing trades...")
    signals = engine.evaluate(matches, tracker=tracker)
    trade_logger.log_scan(len(markets), len(matches), len(signals))

    if not signals:
        logger.info("No actionable signals this cycle")
        return 0

    logger.info(f"Generated {len(signals)} trade signals:")
    for i, sig in enumerate(signals[:10]):
        logger.info(f"  #{i+1}: {sig.rationale}")

    # Step 6: Execute
    logger.info("Step 5/5: Executing trades...")
    executed = await executor.execute_batch(signals, telegram, trade_logger)

    elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
    logger.info(f"=== Cycle complete: {executed}/{len(signals)} trades in {elapsed:.1f}s ===")
    return executed


STATE_FILE = "/tmp/bot_state.json"

def export_dashboard_state(
    tracker: PositionTracker,
    engine: DecisionEngine,
    resolution_tracker: ResolutionTracker,
    cycle_count: int,
    start_time: datetime,
):
    """Write JSON snapshot for the dashboard to read."""
    try:
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

        # Read recent trades from CSV
        recent_trades = []
        trade_log = Path("logs/trades.csv")
        if trade_log.exists():
            with open(trade_log) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                for row in rows[-20:]:
                    recent_trades.append({
                        "timestamp": row.get("timestamp", ""),
                        "city": row.get("city", ""),
                        "date": row.get("market_date", ""),
                        "bucket": row.get("outcome", ""),
                        "side": row.get("side", ""),
                        "price": row.get("price_limit", ""),
                        "size": row.get("intended_usd", ""),
                        "ev": row.get("ev", ""),
                        "edge": row.get("edge", ""),
                        "p_true": row.get("p_true", ""),
                        "market_price": row.get("market_price", ""),
                        "status": row.get("fill_status", ""),
                        "is_maker": row.get("is_maker", ""),
                    })

        # Compute per-city exposure
        city_exp = {}
        for o in orders:
            if not o.is_terminal:
                city_exp[o.city] = city_exp.get(o.city, 0) + o.filled_size_usd + o.unfilled_usd

        # Total deployed: sum of filled USD on UNRESOLVED positions only.
        # Resolved markets (past market_date) no longer tie up capital.
        today = date_type.today()
        total_deployed = sum(
            o.filled_size_usd for o in orders
            if o.filled_size_usd > 0
            and o.status != OrderStatus.CANCELLED
            and o.status != OrderStatus.FAILED
            and o.market_date >= today  # exclude resolved/expired markets
        )

        state = {
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
            "recent_trades": recent_trades,
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

        # Atomic write: write to tmp then rename
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE_FILE)

    except Exception as e:
        logger.warning(f"Dashboard state export failed: {e}")


# ── Main ──

async def main():
    logger.info("=" * 60)
    logger.info("  Polymarket Weather Trading Bot v5")
    logger.info(f"  Mode: {'LIVE' if cfg.is_live else 'DRY-RUN'}")
    logger.info(f"  Bankroll: ${cfg.bankroll:.0f}")
    logger.info(f"  Cities: {', '.join(cfg.cities)}")
    logger.info(f"  Scan interval: {cfg.scan_interval_sec}s")
    logger.info(f"  Ensemble blending: {'enabled' if EnsembleBlender().enabled else 'disabled (no OWM key)'}")
    logger.info(f"  Fill tracking: enabled")
    logger.info(f"  Resolution tracking: enabled")
    logger.info("=" * 60)

    # Initialize all components
    scanner = ForecastScanner()
    blender = EnsembleBlender()
    parser = PolymarketParser()
    engine = DecisionEngine()
    tracker = PositionTracker()
    executor = OrderExecutor(tracker)
    telegram = TelegramAlerter()
    trade_logger = TradeLogger()
    perf_tracker = PerformanceTracker()
    metar_fetcher = MetarFetcher()
    resolution_tracker = ResolutionTracker()

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
            executed = await run_scan_cycle(
                scanner, blender, parser, engine, executor,
                tracker, telegram, trade_logger, metar_fetcher,
                resolution_tracker,
                cycle_count=cycle_count,
            )
            total_trades += executed
            hourly_trades += executed

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
                    from execution import TRADE_LOG
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

            export_dashboard_state(
                tracker, engine, resolution_tracker,
                cycle_count, start_time,
            )

        except Exception as e:
            logger.exception(f"Scan cycle error: {e}")
            await telegram.error_alert(str(e))

        try:
            await asyncio.wait_for(shutdown_event.wait(), timeout=cfg.scan_interval_sec)
            break
        except asyncio.TimeoutError:
            pass

    logger.info("Bot shutting down...")
    await telegram.send(
        f"*Bot Stopped v5*\n"
        f"Session trades: {total_trades}\n"
        f"{resolution_tracker.get_pnl_summary()}\n"
        f"{tracker.get_exposure_summary()}\n"
        f"{perf_tracker.get_summary()}"
    )

if __name__ == "__main__":
    asyncio.run(main())
