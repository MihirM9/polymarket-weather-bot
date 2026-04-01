# tests/test_backtest_forecast.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from datetime import date

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
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    r, _ = approx.generate("Miami", date(2025, 7, 15), days_out=1, actual_high=101.0)
    assert r.regime in ("heat", "frontal", "extreme")

    r, _ = approx.generate("Miami", date(2025, 7, 15), days_out=1, actual_high=91.0)
    assert r.regime in ("normal", "stable")

def test_confidence_from_sigma():
    from backtest_forecast import HistoricalForecastApproximator

    approx = HistoricalForecastApproximator()
    approx.set_climatology({"Miami": {196: 91.0}})

    r0, _ = approx.generate("Miami", date(2025, 7, 15), days_out=0, actual_high=91.0)
    r5, _ = approx.generate("Miami", date(2025, 7, 15), days_out=5, actual_high=91.0)
    assert r0.confidence > r5.confidence
    assert 0.0 <= r0.confidence <= 1.0
    assert 0.0 <= r5.confidence <= 1.0
