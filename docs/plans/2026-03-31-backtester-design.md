# Backtester Design — Polymarket Weather Trading Bot

**Date:** 2026-03-31
**Status:** Ready for implementation
**Goal:** Validate whether the DecisionEngine's edge detection is real by replaying 12–18 months of historical data through the exact same code path the live bot uses.

## Pass/Fail Criteria

| Metric | Threshold | Verdict |
|--------|-----------|---------|
| Sharpe (annualized) | < 1.5 | Edge likely fake or too small |
| Win rate | < 62% | Insufficient signal quality |
| Max drawdown | > 15% | Risk controls inadequate |

If all three pass, the strategy is validated for continued live trading.
If any fail, the strategy needs surgery before scaling capital.

## Architecture

```
┌──────────────────┐     ┌──────────────────┐
│ NOAA Historical  │     │ Gamma API        │
│ Daily Highs      │     │ Closed Markets   │
│ (api.weather.gov)│     │ (gamma-api)      │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
    ┌────▼────────────────────────▼────┐
    │     HistoricalDataLoader         │
    │  - fetch_noaa_daily_highs()      │
    │  - fetch_gamma_closed_markets()  │
    │  - load_climatology_normals()    │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │  HistoricalForecastApproximator │
    │  - Realistic: clim + bias +    │
    │    noise (no look-ahead)       │
    │  - Optimistic: actual + noise  │
    │    (upper bound / sanity)      │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │     MispricingModel             │
    │  - Calibrated from real Gamma   │
    │  - Tail overpricing, mode       │
    │    underpricing, convergence    │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │     BacktestEngine              │
    │  - Daily snapshots (5d → 0d)   │
    │  - Dual variant (real/optim)   │
    │  - DecisionEngine.evaluate()   │
    │  - MockTracker for risk caps   │
    │  - Score vs actual high        │
    └────────────────┬────────────────┘
                     │
    ┌────────────────▼────────────────┐
    │     BacktestScorecard           │
    │  - Core metrics + breakdowns   │
    │  - Variant comparison          │
    │  - SensitivityAnalyzer sweeps  │
    │  - Fragility notes             │
    │  - CSV export                  │
    └─────────────────────────────────┘
```

## File Map

| File | Purpose |
|------|---------|
| `backtester.py` | Main entry point, BacktestEngine loop, CLI |
| `backtest_data.py` | HistoricalDataLoader, NOAA fetching, Gamma closed market fetching |
| `backtest_forecast.py` | HistoricalForecastApproximator, climatology, bias model |
| `backtest_pricing.py` | MispricingModel, calibration from real Gamma data |
| `backtest_scorecard.py` | BacktestScorecard, SensitivityAnalyzer, terminal output, CSV export |
| `backtest_tracker.py` | MockTracker (lightweight PositionTracker for backtest risk caps) |

## Component Details

### 1. HistoricalDataLoader (`backtest_data.py`)

**NOAA Daily Highs:**
- Source: NWS station observations API (`api.weather.gov/stations/{id}/observations`)
- Same endpoint as `resolution_tracker.fetch_actual_high()` — reuse that logic
- Fetch in bulk: for each city/station, query date ranges (30 days at a time)
- Cache to `data/noaa_daily_highs.csv` (city, date, actual_high_f, station_id)
- ~6 cities × 365 days × 1.5 years = ~3,285 data points

**NOAA Climate Normals:**
- Source: NCEI Climate Normals (1991-2020), freely downloadable CSV
- Daily normals per station: expected high for each day-of-year
- Cache to `data/climate_normals.csv`
- Used as the anchor for realistic forecast generation

**Gamma Closed Markets:**
- Source: `gamma-api.polymarket.com/markets?active=false&closed=true`
- Filter for temperature markets (same regex as live parser)
- Extract: market_id, city, date, bucket labels, final prices, outcomes
- Cache to `data/gamma_closed_markets.json`
- Used to calibrate the mispricing model AND as real price data where available

### 2. HistoricalForecastApproximator (`backtest_forecast.py`)

Generates plausible CityForecast objects WITHOUT seeing the actual high.

**Realistic variant** (primary — no look-ahead):
```
base_high = climatology_normal(city, day_of_year)
bias = horizon_bias(days_out) + regime_bias(regime) + seasonal_bias(month)
noise = random.gauss(0, sigma_horizon * seasonal_mult * regime_mult)
forecast_high = base_high + bias + noise
```

**Optimistic variant** (upper bound — uses actual):
```
noise = random.gauss(0, sigma_horizon * 0.5)
forecast_high = actual_high + noise
```

**Horizon σ model** (from NWS verification data):

| Days out | MAE (°F) | σ used |
|----------|----------|--------|
| 0 | ~1 | 1.5 |
| 1 | ~2 | 2.0 |
| 2 | ~2.5 | 2.5 |
| 3 | ~3 | 3.0 |
| 5 | ~4 | 4.5 |
| 7 | ~5.5 | 6.0 |

**Horizon bias** (NWS systematic under-forecasting of extremes):

| Days out | Bias (°F) |
|----------|-----------|
| 0 | 0.0 |
| 1 | -0.3 |
| 2 | -0.5 |
| 3 | -0.7 |
| 5 | -1.0 |
| 7 | -1.2 |

**Regime inference** from actual conditions:
- actual >> climatology + 8°F → "heat" / possible warm front
- actual << climatology - 8°F → "cold" / possible cold front
- high humidity months (Jun-Sep) in southern cities → "convective"
- large day-to-day swings → "frontal"
- otherwise → "normal" or "stable"

**Seasonal σ multipliers** (same as live bot):
- Apr: 1.35x, May: 1.2x, Jun: 1.1x
- Jul: 0.9x, Aug: 0.95x, Sep: 1.1x
- Oct: 1.15x, Nov: 1.2x, Dec: 1.0x
- Jan: 1.0x, Feb: 1.05x, Mar: 1.25x

### 3. MispricingModel (`backtest_pricing.py`)

Generates synthetic market prices calibrated from real Polymarket behavior.

**Calibration step** (run once from Gamma closed markets):
1. For each closed temperature market:
   - Get final bucket prices and actual outcome (1.0 for winner, 0.0 for losers)
   - Compute `bias = market_price - true_outcome` per bucket
2. Bin biases by:
   - Bucket position (tail vs central) → `tail_overpricing` parameter
   - Days-to-resolution snapshot (if available) → `convergence_rate`
   - Overall noise level → `noise_sigma`
3. Fitted constants (expected ranges):
   - `tail_overpricing`: 0.05–0.10 (tails priced 5-10¢ too high)
   - `mode_underpricing`: 0.03–0.05 (most-likely bucket underpriced)
   - `noise_sigma`: 0.02–0.04
   - `convergence_rate`: 0.3–0.5 (prices sharpen near resolution)

**Price generation:**
```
For each bucket i:
    tail_distance = distance from mode bucket (normalized 0-1)
    tail_bias = tail_overpricing * tail_distance
    mode_bias = mode_underpricing if bucket_i is mode else 0
    convergence = 1.0 - (days_out / 7.0) * convergence_rate
    noise = random.gauss(0, noise_sigma * convergence)
    price = true_prob + (tail_bias + mode_bias) * convergence + noise
    price = clamp(price, 0.02, 0.98)
```

### 4. BacktestEngine (`backtester.py`)

**Core loop:**
```python
for city in cities:
    for target_date in daterange(start, end):
        actual_high = loader.get_actual_high(city, target_date)
        if actual_high is None:
            continue

        for days_out in range(5, -1, -1):  # 5 down to 0
            forecast_r, forecast_o = approximator.generate(
                city, target_date, days_out, actual_high
            )
            market = build_market(city, target_date, days_out, actual_high)

            for variant, fc in [("realistic", forecast_r), ("optimistic", forecast_o)]:
                city_forecast = CityForecast(...)
                signals = engine.evaluate([(market, city_forecast)], tracker=mock_tracker)
                for sig in signals:
                    won = score_signal(sig, actual_high)
                    pnl = calculate_pnl(sig, won)
                    result.record(BacktestTrade(..., variant=variant))
```

**MockTracker** enforces:
- Per-market caps (3% of bankroll)
- Absolute position cap ($10)
- Correlated exposure caps (NYC+Chicago share 1.5x single-city cap)
- Daily exposure cap (30% of bankroll)
- `has_active_order()` dedup across horizons
- Resets daily

**Market construction priority:**
1. Real Gamma closed market with prices → use directly
2. No real data → generate synthetic via MispricingModel

**Scoring:** Reuses `ResolutionTracker._temp_in_bucket()` and `._calculate_pnl()`.

### 5. BacktestScorecard (`backtest_scorecard.py`)

**Core metrics:**
- Sharpe ratio (annualized, risk-free = 0)
- Sortino ratio (downside deviation only)
- Win rate
- Profit factor (gross wins / gross losses)
- Max drawdown (peak-to-trough cumulative PnL)
- Average drawdown + recovery time (days to new high)
- Calmar ratio (annualized PnL / max DD)
- Average EV per trade
- Total PnL / total deployed

**Breakdowns:**
- By city: Sharpe, win rate, PnL, trade count
- By month: Sharpe, win rate, PnL
- By horizon (days_out): win rate, avg EV, trade count
- By regime: win rate, PnL
- By side (BUY vs SELL): win rate, PnL

**Variant comparison:**
- Side-by-side realistic vs optimistic
- Look-ahead leakage = optimistic_sharpe - realistic_sharpe

**Sensitivity sweeps** (one-at-a-time):

| Parameter | Values swept |
|-----------|-------------|
| min_edge | 0.05, 0.06, 0.08, 0.10, 0.12, 0.15 |
| kelly_fraction | 0.05, 0.10, 0.15, 0.20, 0.25 |
| sigma_mult | 0.8, 0.9, 1.0, 1.1, 1.2, 1.3 |
| max_kelly_mult | 0.75, 1.0, 1.25, 1.5 |
| tail_overpricing | 0.03, 0.05, 0.08, 0.10, 0.12 |

For each: show Sharpe, win rate, max drawdown, PnL, trade count.

**Fragility notes** (auto-generated):
- Flag any parameter where Sharpe drops below 1.0
- Flag any city or regime where win rate < 55%
- Note parameter ranges where edge is robust (✓) vs fragile (⚠)

**Output formats:**
- Terminal (formatted table with sections)
- CSV (`logs/backtest_YYYYMMDD_HHMMSS.csv`) with every trade
- Summary JSON for programmatic consumption

### 6. CLI Interface

```bash
# Full backtest with defaults
python backtester.py

# Custom date range
python backtester.py --start 2025-01-01 --end 2026-03-31

# Specific cities
python backtester.py --cities "New York,Miami"

# With sensitivity analysis (slower — runs ~30 backtests)
python backtester.py --sensitivity

# Data fetch only (populate caches)
python backtester.py --fetch-only

# Quick mode (realistic variant only, no sensitivity)
python backtester.py --quick
```

## Key Design Decisions

1. **Same code path as live** — `DecisionEngine.evaluate()` is called directly, not reimplemented
2. **No look-ahead bias** — Realistic forecast uses climatology + bias + noise, not actual high
3. **Dual variants** — Gap between realistic and optimistic quantifies methodology risk
4. **Calibrated synthetic prices** — Mispricing model trained on real Gamma data, not efficient markets
5. **No execution simulation** — Tests signal quality, not fill mechanics (v2 concern)
6. **Deterministic with seed** — `random.seed(42)` for reproducible results; vary seed for robustness

## Methodology Disclosure (included in every scorecard)

> Forecasts are synthetically generated from NOAA 30-year climate normals
> with calibrated horizon-dependent bias and Gaussian noise. This is a
> conservative approximation — real NWS forecasts may contain additional
> systematic biases not modeled here. Market prices use real Gamma API
> data where available; synthetic prices elsewhere are calibrated from
> observed retail mispricing patterns in closed Polymarket temperature
> markets. No order execution, partial fills, or slippage are simulated.
> Results represent expected signal quality, not realized trading P&L.

## Implementation Order

1. `backtest_data.py` — Data fetching + caching (NOAA highs, climatology, Gamma)
2. `backtest_forecast.py` — Forecast approximator (realistic + optimistic)
3. `backtest_pricing.py` — Mispricing model + calibration
4. `backtest_tracker.py` — MockTracker
5. `backtester.py` — Core loop + CLI
6. `backtest_scorecard.py` — Metrics, breakdowns, sensitivity, output
