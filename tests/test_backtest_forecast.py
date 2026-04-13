# tests/test_backtest_forecast.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import date, timedelta


def test_realistic_forecast_uses_climatology_not_actual():
    """Core anti-leakage test: realistic forecast must NOT center on actual."""
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    random.seed(42)
    results = []
    for _ in range(1000):
        r, o = approx.generate("Miami", date(2025, 7, 15), days_out=3, actual_high=98.0)
        results.append(r.high_f)

    mean_forecast = sum(results) / len(results)
    assert abs(mean_forecast - 91.0) < 3.0, f"Mean {mean_forecast:.1f} too close to actual 98"
    assert abs(mean_forecast - 98.0) > 3.0, f"Mean {mean_forecast:.1f} leaked actual high"

def test_optimistic_forecast_centers_on_actual():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    random.seed(42)
    results = []
    for _ in range(1000):
        r, o = approx.generate("Miami", date(2025, 7, 15), days_out=3, actual_high=98.0)
        results.append(o.high_f)

    mean_forecast = sum(results) / len(results)
    assert abs(mean_forecast - 98.0) < 2.0

def test_sigma_increases_with_horizon():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"New York": {1: 38.0}})

    random.seed(42)
    r0, _ = approx.generate("New York", date(2025, 1, 1), days_out=0, actual_high=40.0)
    r5, _ = approx.generate("New York", date(2025, 1, 1), days_out=5, actual_high=40.0)
    assert r5.sigma > r0.sigma

def test_seasonal_sigma_multiplier():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {105: 85.0, 196: 91.0}})

    random.seed(42)
    r_apr, _ = approx.generate("Miami", date(2025, 4, 15), days_out=3, actual_high=85.0)
    r_jul, _ = approx.generate("Miami", date(2025, 7, 15), days_out=3, actual_high=91.0)
    assert r_apr.sigma > r_jul.sigma

def test_regime_inference():
    """Regime should be inferred from lagged actuals, not the target day's actual."""
    from backtest_forecast import HistoricalForecastApproximator

    target = date(2025, 7, 15)
    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    # Lagged days all hot → heat/extreme regime
    approx.set_lagged_actuals({"Miami": {
        target - timedelta(days=1): 101.0,
        target - timedelta(days=2): 102.0,
        target - timedelta(days=3): 100.0,
    }})
    r, _ = approx.generate("Miami", target, days_out=1, actual_high=101.0)
    assert r.regime in ("heat", "frontal", "extreme")

    # Lagged days near climatology → normal/stable regime
    approx.set_lagged_actuals({"Miami": {
        target - timedelta(days=1): 91.0,
        target - timedelta(days=2): 92.0,
        target - timedelta(days=3): 90.0,
    }})
    r, _ = approx.generate("Miami", target, days_out=1, actual_high=91.0)
    assert r.regime in ("normal", "stable")

    # No lagged data → default to "normal"
    approx.set_lagged_actuals({})
    r, _ = approx.generate("Miami", target, days_out=1, actual_high=105.0)
    assert r.regime == "normal"

def test_confidence_from_sigma():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    r0, _ = approx.generate("Miami", date(2025, 7, 15), days_out=0, actual_high=91.0)
    r5, _ = approx.generate("Miami", date(2025, 7, 15), days_out=5, actual_high=91.0)
    assert r0.confidence > r5.confidence
    assert 0.0 <= r0.confidence <= 1.0
    assert 0.0 <= r5.confidence <= 1.0
