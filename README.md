# Polymarket Weather Trading Bot v5

Autonomous Python trading bot for Polymarket US city daily high-temperature bucket markets. Trades on Polygon (chain_id=137) via py-clob-client. Exploits systematic mispricing between NOAA/NWS forecast accuracy (~90% at 5-day, 1-2F short-term) and retail market pricing.

## How It Works

Polymarket hosts daily prediction markets like: *"Will the highest temperature in Chicago be between 70-71F on March 26?"* Each city/date has ~11 temperature buckets (e.g., 60-61F, 62-63F, ... 80F+), each priced by the market as a probability.

The bot:
1. **Fetches weather forecasts** from NWS/NOAA, METAR (aviation weather), and NOAA station observations
2. **Blends multiple data sources** into a single forecast with uncertainty (ensemble blending)
3. **Converts the forecast into bucket probabilities** using a Gaussian CDF model
4. **Compares those probabilities against market prices** on Polymarket
5. **Trades when it finds mispricing** -- if the bot thinks a bucket has a 25% chance but the market prices it at 7%, that's an edge
6. **Manages risk** with 13 layers of controls (Kelly sizing, exposure caps, loss limits, etc.)

The live loop currently runs every 2 minutes: poll fills -> forecast -> blend -> parse markets -> decide -> execute. That cadence is configurable and should be treated as an operational default, not a permanent truth.

## Architecture

```
Position Tracker <-- poll CLOB for fill status (start of each cycle)
       |
Forecast Scanner --> Ensemble Blender --> OWM API (supplemental)
(NWS/NOAA)           (NWS + OWM +     <-- METAR Fetcher (aviation weather)
                      METAR + Station)  <-- NOAA Station Obs (resolution source)
                           |
                    Polymarket Parser
                    (Gamma API)
                           |
                    Decision Engine <-- tracker.total_exposure
                    (EV / Kelly / Risk)
                           |
                    Execution --> PositionTracker.register()
                    (CLOB / Telegram / CSV)
```

## Runtime Modules

| File | Purpose |
|------|---------|
| `main.py` | Async main loop, scan orchestration, graceful shutdown |
| `config.py` | Loads .env, typed Config singleton |
| `forecasting/` | Forecast subsystem entrypoint and orchestration package |
| `forecast_scanner.py` | NWS API -> Gaussian bucket probabilities, negation-aware regime detection, seasonal sigma, station observations |
| `ensemble_blender.py` | NWS + OWM + METAR + Station blending, dynamic sigma, city-specific peak hours, cache pruning |
| `metar_fetcher.py` | METAR aviation weather real-time observations (aviationweather.gov), temp parsing |
| `polymarket_parser.py` | Gamma API market discovery, bucket parsing (regex) with parse metrics, city/date matching |
| `decision_engine.py` | EV calc, capped Kelly, time-decay sizing, correlated exposure caps, maker fee EV, seasonal sigma |
| `position_tracker.py` | CLOB fill polling, exposure tracking, stale cancel cooldown, adverse selection detection |
| `execution.py` | ClobClient orders, maker/taker pricing, orderbook depth, Sharpe tracking, CSV/Telegram |
| `dry_run_simulator.py` | Realistic paper trading -- fetches live orderbooks, simulates partial fills and slippage |

## Infrastructure

| File | Purpose |
|------|---------|
| `api_utils.py` | Shared HTTP retry/backoff for external API calls |
| `api_models.py` | Response validation models for external APIs |
| `background_io.py` | Background persistence so disk writes do not block the event loop |
| `runtime_logging.py` | Queue-backed logging setup |
| `health_monitor.py` | Runtime health checks and fail-safe shutdown rules |

## Research & Backtesting

The repo still contains several `backtest_*` modules. That is intentional for now, but the target state is a single `backtesting/` package once the live path settles enough to justify a more aggressive collapse.

## Key Math

- **Bucket probabilities**: Gaussian CDF integration. `P(bucket) = CDF((hi - mu) / sigma) - CDF((lo - mu) / sigma)` where mu = forecast high, sigma = horizon + regime + seasonal adjusted
- **EV Yes**: `p_true * (1 - price) * (1 - fee) - (1 - p_true) * price`
- **EV No**: `(1 - p_true) * price_yes * (1 - fee) - p_true * (1 - price_yes)`
- **Kelly**: `f* = (b*p - q) / b` where `b = (1-price)/price`, tempered by confidence and capped
- **Ensemble sigma**: `sqrt(sigma_base^2 + sigma_spread^2)` where sigma_spread = std dev across model forecasts
- **Seasonal sigma**: Base sigma * monthly multiplier (Apr=1.35x more volatile, Jul=0.9x tighter)
- **Time-decay**: `min_ev * sqrt(days_to_resolution)` -- demands higher EV for longer-duration trades

## Risk Controls (13 layers)

| # | Control | Default |
|---|---------|---------|
| 1 | Capped confidence-adaptive Kelly | 0.25x-1.25x base, hard cap |
| 2 | Dynamic edge threshold | 8-21 cents depending on uncertainty |
| 3 | Time-decay EV threshold | Scales with sqrt(days) |
| 4 | Seasonal sigma adjustment | Spring/fall inflated, summer tighter |
| 5 | Per-market cap | 3% of bankroll |
| 6 | Absolute position cap | $10 default |
| 7 | Correlated exposure caps | NYC+Chicago share 1.5x single-city cap |
| 8 | Daily exposure cap | 30% of bankroll |
| 9 | Daily loss cap | $50 default, auto-shutdown |
| 10 | Cancel-replace cooldown | 3 cycles after stale order cancel |
| 11 | Orderbook depth check | Min liquidity before placing orders |
| 12 | Adverse selection detection | Flags instant fills as informed counter-trading |
| 13 | Maker/taker optimization | Passive limit orders for 0% fees |

## Quick Start

### 1. Install
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional developer tooling:

```bash
pip install -r requirements-dev.txt
```

If you prefer `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

### 2. Configure
```bash
cp .env.example .env
nano .env
```

Key settings:
- `MODE=dry-run` (default, no real money)
- `PRIVATE_KEY` -- leave as dummy for dry-run
- `BANKROLL=500` -- simulated bankroll in USDC
- `TELEGRAM_TOKEN` + `TELEGRAM_CHAT_ID` -- optional alerts
- `TELEGRAM_MIN_EV=0.15` -- only alert on 15%+ EV trades

### 3. Run (paper trading)
```bash
python3 main.py
```

The bot will:
- Fetch real NWS forecasts and real Polymarket prices
- Compute real probabilities and edge
- Simulate trades against live orderbook depth (partial fills, slippage)
- Log everything to `logs/trades.csv` and `logs/dry_run_fills.csv`
- Send Telegram alerts for high-EV trades (if configured)

**No real money is used in dry-run mode.**

### 4. Check results
```bash
# Trade decisions
cat logs/trades.csv

# Dry-run fill simulation
cat logs/dry_run_fills.csv

# Check actual temperatures the next day at weather.gov
# Compare against what the bot traded
```

### 5. Go live (only after validating dry-run)
```bash
# In .env:
MODE=live
PRIVATE_KEY=0xYOUR_REAL_POLYGON_PRIVATE_KEY
```

Your wallet needs USDC + small amount of MATIC on Polygon network.

## VPS Deployment (24/7)

For uninterrupted trading, deploy on a VPS (DigitalOcean, Oracle Cloud free tier, etc.):

```bash
# Clone/copy code to VPS
cd ~/polymarket-weather-bot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env with your config
cp .env.example .env
nano .env

# Set up systemd for auto-restart
cat > /etc/systemd/system/weatherbot.service << 'EOF'
[Unit]
Description=Polymarket Weather Trading Bot
After=network.target

[Service]
WorkingDirectory=/root/polymarket-weather-bot
ExecStart=/root/polymarket-weather-bot/venv/bin/python3 main.py
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable weatherbot
systemctl start weatherbot

# Check logs anytime
journalctl -u weatherbot -f
```

## External APIs

| API | Purpose | Auth |
|-----|---------|------|
| NWS/NOAA (`api.weather.gov`) | Grid forecasts + station observations | Free, no key (needs User-Agent) |
| METAR (`aviationweather.gov`) | Real-time airport weather, updates every 30-60 min | Free, no key |
| OpenWeatherMap | Supplemental forecasts (optional) | Free tier 1000 calls/day, needs API key |
| Polymarket Gamma API | Market discovery + prices | Free, no key |
| Polymarket CLOB | Order placement + orderbook | Needs Polygon private key |
| Telegram | Trade alerts | Needs bot token + chat ID |

## How the Bot Makes Money

The bot's edge comes from **knowing the weather better than the market**:

1. **Most Polymarket traders** use intuition, current weather, or basic forecasts
2. **The bot** uses NWS grid forecasts, METAR aviation data, NOAA station observations, and ensemble blending
3. **When the market prices a bucket at 7% but the bot calculates 25%**, it buys
4. **When the market prices an extreme bucket at 15% but the bot calculates 0%**, it sells (buys NO)
5. **Only one bucket wins per city per day** -- the SELL/NO bets on unlikely buckets win most often (10 out of 11 buckets lose each day)

## Version History

- **v1**: Basic Gaussian model, flat Kelly (0.15x), flat edge threshold (8 cents), fire-and-forget orders
- **v2**: Weather regime detection (inflates sigma for storms/fronts), confidence-adaptive Kelly, dynamic edge thresholds
- **v3**: Fill tracking via PositionTracker, ensemble blending (NWS + OWM), stale order cancellation
- **v3.1**: 8 robustness fixes -- cancel-replace cooldown, negation-aware regime detection, API retry/backoff, parse metrics, city-specific peak hours, OWM cache pruning, configurable fees, orderbook depth check
- **v4**: 9 optimizations -- station-specific NOAA modeling, METAR aviation weather as 3rd ensemble source, seasonal sigma adjustment, time-decay capital allocation, correlated exposure caps, maker/taker fee optimization, adverse selection detection, Sharpe ratio tracking, capped adaptive Kelly
- **v4.1**: Realistic dry-run simulator -- live orderbook matching, partial fills, slippage tracking, fill rate metrics
- **v5**: Runtime hardening -- response validation, background persistence, queue-backed logging, health-monitor fail-safes, stronger live-path tests

## Developer Notes

- See [ARCHITECTURE.md](./ARCHITECTURE.md) for the current and target module layout.
- `requirements.txt` is the runtime install surface.
- `requirements-dev.txt` is for testing, linting, property-based testing, and recorded-response integration work.

## License

Private use only. Not financial advice. Trading involves risk of loss. Use at your own risk.
