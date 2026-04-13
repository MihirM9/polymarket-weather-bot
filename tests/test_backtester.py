# tests/test_backtester.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date


def test_backtest_engine_produces_trades():
    """Integration test: engine should produce trades from synthetic data."""
    from backtester import BacktestEngine

    engine = BacktestEngine(bankroll=500.0, seed=42)

    engine.loader._highs = {
        ("Miami", date(2025, 7, 15)): 91.0,
        ("Miami", date(2025, 7, 16)): 89.0,
        ("Miami", date(2025, 7, 17)): 93.0,
    }
    engine.loader._climatology = {
        "Miami": {196: 91.0, 197: 91.2, 198: 91.3},
    }
    engine.approximator.set_climatology(engine.loader._climatology)

    result = engine.run(
        cities=["Miami"],
        start_date=date(2025, 7, 15),
        end_date=date(2025, 7, 17),
    )

    assert len(result.trades) > 0
    variants = set(t.variant for t in result.trades)
    assert "realistic" in variants
    assert "optimistic" in variants

def test_backtest_deduplication():
    """Engine should not re-enter same market+bucket across horizons."""
    from backtester import BacktestEngine

    engine = BacktestEngine(bankroll=500.0, seed=42)

    engine.loader._highs = {
        ("Miami", date(2025, 7, 15)): 91.0,
    }
    engine.loader._climatology = {
        "Miami": {196: 91.0},
    }
    engine.approximator.set_climatology(engine.loader._climatology)

    result = engine.run(
        cities=["Miami"],
        start_date=date(2025, 7, 15),
        end_date=date(2025, 7, 15),
    )

    for variant in ["realistic", "optimistic"]:
        seen = set()
        for t in result.trades:
            if t.variant != variant:
                continue
            key = f"{t.city}_{t.target_date}_{t.outcome_label}_{t.variant}"
            assert key not in seen, f"Duplicate trade: {key}"
            seen.add(key)

def test_backtest_scoring_correct():
    """Verify BUY YES wins when actual temp is in bucket."""
    from backtester import BacktestEngine

    engine = BacktestEngine(bankroll=500.0, seed=42)

    engine.loader._highs = {
        ("Miami", date(2025, 7, 15)): 91.0,
    }
    engine.loader._climatology = {"Miami": {196: 91.0}}
    engine.approximator.set_climatology(engine.loader._climatology)

    result = engine.run(
        cities=["Miami"],
        start_date=date(2025, 7, 15),
        end_date=date(2025, 7, 15),
    )

    for t in result.trades:
        if t.variant != "realistic":
            continue
        if t.bucket_low is not None and t.bucket_high is not None:
            in_bucket = t.bucket_low <= 91.0 < t.bucket_high
            if t.side == "BUY":
                assert t.won == in_bucket
            else:
                assert t.won == (not in_bucket)
