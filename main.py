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
import logging
import signal
import sys
from datetime import datetime, date as date_type, timezone

import aiohttp

from config import cfg
from forecast_scanner import ForecastScanner, CityForecast, compute_confidence
from polymarket_parser import PolymarketParser
from decision_engine import DecisionEngine
from ensemble_blender import EnsembleBlender, ForecastPoint
from position_tracker import PositionTracker
from execution import OrderExecutor, TelegramAlerter, TradeLogger, PerformanceTracker
from metar_fetcher import MetarFetcher

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
) -> int:
    cycle_start = datetime.now(timezone.utc)
    logger.info(f"=== Scan cycle start: {cycle_start.isoformat()} ===")

    # Step 0: Poll fills on open orders (v3 Fix #1)
    if not executor.dry_run:
        fill_changes = await tracker.poll_fills()
        if fill_changes:
            logger.info(f"Fill poll: {fill_changes} order(s) updated. {tracker.get_exposure_summary()}")
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
    station_observations: dict = {}
    today = date_type.today()
    same_day_cities = set(fc.city for fc in forecasts if fc.forecast_date == today)
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

        # For same-day forecasts, add METAR and station observations as ensemble sources
        if fc.forecast_date == today:
            # METAR observation (v4: aviation weather ground truth)
            metar_obs = metar_observations.get(fc.city)
            if metar_obs:
                supplemental.append(ForecastPoint(
                    source="metar",
                    high_f=metar_obs.temp_f,
                    sigma=0.5,
                    weight=1.5,
                ))

            # NOAA station observation (v4: station-specific ground truth)
            station_temp = station_observations.get(fc.city)
            if station_temp is not None:
                supplemental.append(ForecastPoint(
                    source="noaa_station",
                    high_f=station_temp,
                    sigma=0.5,
                    weight=1.5,
                ))

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


# ── Main ──

async def main():
    logger.info("=" * 60)
    logger.info("  Polymarket Weather Trading Bot v3")
    logger.info(f"  Mode: {'LIVE' if cfg.is_live else 'DRY-RUN'}")
    logger.info(f"  Bankroll: ${cfg.bankroll:.0f}")
    logger.info(f"  Cities: {', '.join(cfg.cities)}")
    logger.info(f"  Scan interval: {cfg.scan_interval_sec}s")
    logger.info(f"  Ensemble blending: {'enabled' if EnsembleBlender().enabled else 'disabled (no OWM key)'}")
    logger.info(f"  Fill tracking: enabled")
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

    await telegram.send(
        f"*Bot Started v3* ({'LIVE' if cfg.is_live else 'DRY-RUN'})\n"
        f"Bankroll: ${cfg.bankroll:.0f}\n"
        f"Ensemble: {'NWS+OWM' if blender.enabled else 'NWS only'}\n"
        f"Fill tracking: active\n"
        f"Scan every {cfg.scan_interval_sec}s"
    )

    total_trades = 0
    cycle_count = 0

    while not shutdown_event.is_set():
        try:
            cycle_count += 1
            executed = await run_scan_cycle(
                scanner, blender, parser, engine, executor,
                tracker, telegram, trade_logger, metar_fetcher
            )
            total_trades += executed

            if cycle_count % 50 == 0:
                perf_tracker.record_daily_pnl(engine.daily_pnl, cfg.bankroll)
                await telegram.daily_summary(engine.daily_pnl, total_trades, tracker)
                logger.info(f"Performance: {perf_tracker.get_summary()}")

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
        f"*Bot Stopped v3*\n"
        f"Total trades: {total_trades}\n"
        f"PnL: ${engine.daily_pnl:+.2f}\n"
        f"{tracker.get_exposure_summary()}\n"
        f"{perf_tracker.get_summary()}"
    )

if __name__ == "__main__":
    asyncio.run(main())
