# CLAUDE.md — Polymarket Weather Trading Bot v4

## What This Is
Autonomous Python trading bot for Polymarket US city daily high-temperature bucket markets. Trades on Polygon (chain_id=137) via py-clob-client. Exploits systematic mispricing between NOAA/NWS forecast accuracy (~90% at 5-day, 1-2°F short-term) and retail market pricing.

## Architecture
```
Position Tracker ◄── poll CLOB for fill status (start of each cycle)
       │
Forecast Scanner ──► Ensemble Blender ──► OWM API (supplemental)
(NWS/NOAA)           (NWS + OWM +       ◄── METAR Fetcher (aviation weather)
                      METAR + Station)   ◄── NOAA Station Obs (resolution source)
                           │
                    Polymarket Parser
                    (Gamma API)
                           │
                    Decision Engine ◄── tracker.total_exposure
                    (EV / Kelly / Risk)
                           │
                    Execution ──► PositionTracker.register()
                    (CLOB / Telegram / CSV)
```

Main loop runs every 2 minutes: poll_fills → forecast → blend → parse → decide → execute.

## File Map
| File | Purpose |
|------|---------|
| `main.py` | Async main loop, scan orchestration, graceful shutdown |
| `config.py` | Loads .env, typed Config singleton |
| `api_utils.py` | Shared HTTP retry/backoff for all external API calls |
| `forecast_scanner.py` | NWS API → Gaussian bucket probabilities, negation-aware regime detection, seasonal σ, station observations |
| `ensemble_blender.py` | NWS + OWM + METAR + Station blending, dynamic σ, city-specific peak hours, cache pruning |
| `metar_fetcher.py` | METAR aviation weather real-time observations (aviationweather.gov), temp parsing |
| `polymarket_parser.py` | Gamma API market discovery, bucket parsing (regex) with parse metrics, city/date matching |
| `decision_engine.py` | EV calc, capped Kelly, time-decay sizing, correlated exposure caps, maker fee EV |
| `position_tracker.py` | CLOB fill polling, exposure tracking, stale cancel cooldown, adverse selection detection |
| `execution.py` | ClobClient orders, maker/taker pricing, orderbook depth, Sharpe tracking, CSV/Telegram |

## Key Math
- **Bucket probabilities**: Gaussian CDF integration. `P(bucket) = Φ((hi - μ) / σ) - Φ((lo - μ) / σ)` where μ = forecast high, σ = horizon + regime + seasonal adjusted.
- **EV Yes**: `p_true × (1 - price) × (1 - fee) - (1 - p_true) × price` (fee=0 for maker orders)
- **EV No**: `(1 - p_true) × price_yes × (1 - fee) - p_true × (1 - price_yes)`
- **Kelly**: `f* = (b×p - q) / b` where `b = (1-price)/price`, tempered by `base × min(0.25 + 1.25 × confidence, MAX_KELLY_MULT)`
- **Dynamic edge threshold**: `base_edge + (1 - confidence) × 0.05 + max(0, σ - 1.5) × 0.02`
- **Time-decay EV threshold**: `min_ev × sqrt(days_to_resolution)` — demands higher EV for longer-duration trades
- **Seasonal σ**: Base σ × monthly multiplier (Apr=1.35x, Jul=0.9x) — spring more volatile than summer
- **Ensemble σ**: `sqrt(σ_base² + σ_spread²)` where σ_spread = std dev across model forecasts (NWS + OWM + METAR + Station).
- **Correlated exposure**: Cities in same weather system (e.g., NYC+Chicago) share a group cap of 1.5× single-city cap
- **Adverse selection**: Fill speed < 10s flagged; high instant-fill rate indicates informed counter-trading

## Version History
- **v1**: Basic Gaussian model, flat Kelly (0.15x), flat edge threshold (8¢), fire-and-forget orders.
- **v2**: Weather regime detection (inflates σ for storms/fronts), confidence-adaptive Kelly tempering, dynamic edge thresholds.
- **v3**: Fill tracking via PositionTracker (realized vs pending exposure), ensemble blending (NWS + OWM), stale order cancellation.
- **v3.1**: 8 robustness fixes — cancel-replace cooldown, negation-aware regime detection, API retry/backoff, parse metrics, city-specific peak hours, OWM cache pruning, configurable fees, orderbook depth check.
- **v4**: 9 optimizations from "151 Trading Strategies" paper + microstructure analysis — station-specific NOAA modeling, METAR aviation weather as 3rd ensemble source, seasonal σ adjustment, time-decay capital allocation, correlated exposure caps, maker/taker fee optimization, adverse selection detection, Sharpe ratio tracking, capped adaptive Kelly.

## Risk Controls (13 layers)
1. Capped confidence-adaptive Kelly (0.25x–1.25x base, hard cap prevents overconfident sizing)
2. Dynamic edge threshold (8–21¢ depending on uncertainty)
3. Time-decay EV threshold (scales with √days — longer trades need higher EV)
4. Seasonal σ adjustment (spring/fall inflated, summer/winter tighter)
5. Per-market cap (3% of bankroll)
6. Absolute position cap ($10 default)
7. Correlated exposure caps (NYC+Chicago share 1.5x single-city cap)
8. Daily exposure cap (30% of bankroll)
9. Daily loss cap ($50 default, auto-shutdown)
10. Cancel-replace cooldown (3 cycles after stale order cancel)
11. Orderbook depth check (minimum liquidity before placing orders)
12. Adverse selection detection (flags instant fills as possible informed counter-trading)
13. Maker/taker optimization (passive limit orders for 0% fees)

## External APIs
- **NWS/NOAA**: `api.weather.gov` — grid forecasts + station observations (free, no auth, needs User-Agent header)
- **METAR/Aviation**: `aviationweather.gov` — real-time airport weather observations (free, no auth, updates every 30-60 min)
- **OpenWeatherMap**: One Call 3.0 — supplemental forecasts (free tier 1000 calls/day, needs API key)
- **Polymarket Gamma**: `gamma-api.polymarket.com` — market discovery (free, no auth)
- **Polymarket CLOB**: `clob.polymarket.com` — order placement + orderbook data (needs Polygon private key)
- **Telegram**: `api.telegram.org` — alerts (needs bot token + chat ID)

## Config
All params in `.env`. Key ones: `PRIVATE_KEY`, `BANKROLL`, `KELLY_FRACTION=0.15`, `MAX_KELLY_MULT=1.25`, `MIN_EDGE=0.08`, `DAILY_LOSS_CAP=50`, `MODE=dry-run`, `OPENWEATHER_API_KEY`, `FEE_RATE=0.02`, `MAKER_FEE_RATE=0.0`, `MAKER_SPREAD_OFFSET=0.005`, `MIN_BOOK_DEPTH=5.0`, `NOAA_STATIONS`.

## Known Limitations / TODO
1. Gaussian assumption doesn't capture fat tails — skew-normal would be better for extreme buckets
2. No position lifecycle management (can't exit/hedge after fill if forecast reverses)
3. No historical backtester (need NOAA obs + archived Polymarket prices)
4. ~~No order book depth check before placing orders~~ (v3.1: basic depth check added)
5. No seasonal σ adjustment (spring more volatile than mid-summer)
6. City matching is string-based regex — could mismatch edge cases (v3.1: parse metrics added for visibility)
7. PnL tracking is approximate in live mode (fill tracking helps but doesn't track market resolution)

## Style Preferences
- Python 3.10+, async/await with aiohttp
- Type hints everywhere
- Dataclasses for data objects
- Logging via stdlib `logging`
- No scipy dependency — math.erf for Gaussian CDF
- Comments reference research sections (§5.1, §6.2, etc.)
- Conservative defaults — dry-run mode, small positions, multiple safety layers
