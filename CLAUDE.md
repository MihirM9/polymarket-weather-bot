# CLAUDE.md — Polymarket Weather Trading Bot v3.1

## What This Is
Autonomous Python trading bot for Polymarket US city daily high-temperature bucket markets. Trades on Polygon (chain_id=137) via py-clob-client. Exploits systematic mispricing between NOAA/NWS forecast accuracy (~90% at 5-day, 1-2°F short-term) and retail market pricing.

## Architecture
```
Position Tracker ◄── poll CLOB for fill status (start of each cycle)
       │
Forecast Scanner ──► Ensemble Blender ──► OWM API (supplemental)
(NWS/NOAA)           (NWS + OWM blend)
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
| `forecast_scanner.py` | NWS API → Gaussian bucket probabilities, negation-aware weather regime detection, confidence scoring |
| `ensemble_blender.py` | NWS + OpenWeatherMap blending, dynamic σ from model disagreement, city-specific peak hours, cache pruning |
| `polymarket_parser.py` | Gamma API market discovery, bucket parsing (regex) with parse metrics, city/date matching |
| `decision_engine.py` | EV calculation, Kelly criterion, confidence-adaptive tempering, dynamic edge thresholds, cooldown-aware |
| `position_tracker.py` | CLOB fill polling, realized vs pending exposure, stale order cancellation, cancel-replace cooldown |
| `execution.py` | ClobClient orders, orderbook depth check, dry-run simulation, CSV logging, Telegram alerts |

## Key Math
- **Bucket probabilities**: Gaussian CDF integration. `P(bucket) = Φ((hi - μ) / σ) - Φ((lo - μ) / σ)` where μ = forecast high, σ = horizon + regime adjusted.
- **EV Yes**: `p_true × (1 - price) × (1 - fee) - (1 - p_true) × price`
- **EV No**: `(1 - p_true) × price_yes × (1 - fee) - p_true × (1 - price_yes)`
- **Kelly**: `f* = (b×p - q) / b` where `b = (1-price)/price`, tempered by `base × (0.25 + 1.25 × confidence)`
- **Dynamic edge threshold**: `base_edge + (1 - confidence) × 0.05 + max(0, σ - 1.5) × 0.02`
- **Ensemble σ**: `sqrt(σ_base² + σ_spread²)` where σ_spread = std dev across model forecasts.

## Version History
- **v1**: Basic Gaussian model, flat Kelly (0.15x), flat edge threshold (8¢), fire-and-forget orders.
- **v2**: Weather regime detection (inflates σ for storms/fronts), confidence-adaptive Kelly tempering, dynamic edge thresholds.
- **v3**: Fill tracking via PositionTracker (realized vs pending exposure), ensemble blending (NWS + OWM), stale order cancellation.
- **v3.1**: 8 robustness fixes — cancel-replace cooldown, negation-aware regime detection, API retry/backoff, parse metrics, city-specific peak hours, OWM cache pruning, configurable fees, orderbook depth check.

## Risk Controls (9 layers)
1. Confidence-adaptive Kelly (0.25x–1.5x base depending on forecast quality)
2. Dynamic edge threshold (8–21¢ depending on uncertainty)
3. Min EV threshold (3%)
4. Per-market cap (3% of bankroll)
5. Absolute position cap ($10 default)
6. Daily exposure cap (30% of bankroll)
7. Daily loss cap ($50 default, auto-shutdown)
8. Cancel-replace cooldown (3 cycles after stale order cancel, prevents spam loops)
9. Orderbook depth check (minimum liquidity required before placing orders)

## External APIs
- **NWS/NOAA**: `api.weather.gov` — forecasts (free, no auth, needs User-Agent header)
- **OpenWeatherMap**: One Call 3.0 — supplemental forecasts (free tier 1000 calls/day, needs API key)
- **Polymarket Gamma**: `gamma-api.polymarket.com` — market discovery (free, no auth)
- **Polymarket CLOB**: `clob.polymarket.com` — order placement (needs Polygon private key)
- **Telegram**: `api.telegram.org` — alerts (needs bot token + chat ID)

## Config
All params in `.env`. Key ones: `PRIVATE_KEY`, `BANKROLL`, `KELLY_FRACTION=0.15`, `MIN_EDGE=0.08`, `DAILY_LOSS_CAP=50`, `MODE=dry-run`, `OPENWEATHER_API_KEY`, `FEE_RATE=0.02`, `MIN_BOOK_DEPTH=5.0`.

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
