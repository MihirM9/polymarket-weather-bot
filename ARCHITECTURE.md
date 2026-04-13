# Architecture

This repo is now organized into four practical layers:

1. **Trading runtime**
   Forecast ingestion, market parsing, decisioning, execution, position state, health checks.
2. **Research and backtesting**
   Historical data loading, synthetic pricing, replay, scorecards.
3. **Operational tooling**
   Dashboard, logging, alerts, deployment helpers.
4. **Configuration**
   Thin root entrypoints and typed config shared across the runtime.

## Current Runtime Shape

```text
main.py
  -> forecasting/
     -> scanner.py
     -> blender.py
     -> metar.py
     -> service.py
  -> trading/
     -> markets.py
     -> decision.py
     -> execution.py
     -> positions.py
     -> resolution.py
     -> dry_run.py
  -> infrastructure/
     -> http.py
     -> models.py
     -> io.py
     -> logging.py
     -> health.py
```

## Current Research Shape

```text
backtesting/
  -> data.py
  -> forecast.py
  -> pricing.py
  -> replay.py
  -> scorecard.py
  -> tracker.py
  -> price_history.py
```

## Tooling Shape

```text
dashboarding/
  -> app.py
  -> simulate.py

tools/
  -> analyze_trades.py
```

## Root Shape

The root is intentionally thin:

```text
main.py
config.py
amm_config.py
dashboard.html
README.md
ARCHITECTURE.md
```

## Principles

Principles that matter going forward:

- keep the **live trading path** obvious and easy to test
- keep **plumbing** in one place instead of many top-level files
- consolidate research code into a **single backtesting package**
- keep operational helpers out of the runtime path
- prefer deleting indirection over preserving one-file-per-idea sprawl
- enforce quality checks automatically so the repo does not regress
