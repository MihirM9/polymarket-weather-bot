# Architecture

This repo has grown into three practical layers:

1. **Trading runtime**
   Forecast ingestion, market parsing, decisioning, execution, position state, health checks.
2. **Research and backtesting**
   Historical data loading, synthetic pricing, replay, scorecards.
3. **Operational tooling**
   Dashboard, logging, alerts, deployment helpers.

## Current Runtime Shape

```text
main.py
  -> forecasting/
     -> forecast_scanner.py
     -> ensemble_blender.py
     -> metar_fetcher.py
  -> polymarket_parser.py
  -> decision_engine.py
  -> execution.py
  -> position_tracker.py
  -> resolution_tracker.py
  -> infrastructure helpers
     -> background_io.py
     -> runtime_logging.py
     -> health_monitor.py
     -> api_utils.py
     -> api_models.py
```

## Current Research Shape

```text
backtester.py
  -> backtest_data.py
  -> backtest_forecast.py
  -> backtest_pricing.py
  -> backtest_scorecard.py
  -> backtest_tracker.py
  -> price_history.py
```

## Target Shape

The long-term goal is a smaller, more boring repo:

```text
app/
  main.py
  config.py
  forecasting/
  markets/
  decision/
  execution/
  state/
  infrastructure/

backtesting/
  loader.py
  replay.py
  pricing.py
  scorecard.py
```

Principles for getting there:

- keep the **live trading path** obvious and easy to test
- keep **plumbing** in one place instead of many top-level files
- consolidate research code into a **single backtesting package**
- prefer deleting indirection over preserving one-file-per-idea sprawl
