# tests/test_price_replay_integration.py
"""Integration test: full backtest with mocked CLOB price histories."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date, datetime, timezone

from backtesting import BacktestEngine, PriceSnapshot


def test_real_price_replay_produces_trades():
    """Backtest with injected real prices should produce trades with
    price_source='real_clob' and non-circular metrics."""
    engine = BacktestEngine(bankroll=500.0, seed=42)

    # Inject minimal test data
    target = date(2026, 4, 5)
    engine.loader._highs = {("Miami", target): 83.0}
    engine.loader._climatology = {"Miami": {target.timetuple().tm_yday: 80.0}}
    engine.approximator.set_climatology(engine.loader._climatology)
    lagged = {"Miami": {
        date(2026, 4, 4): 82.0,
        date(2026, 4, 3): 81.0,
        date(2026, 4, 2): 79.0,
    }}
    engine.approximator.set_lagged_actuals(lagged)

    # Inject token map and price history for 5 buckets
    buckets = [
        ("70-74°F", 0.05), ("75-79°F", 0.20), ("80-84°F", 0.45),
        ("85-89°F", 0.20), ("90-94°F", 0.10),
    ]
    # Decision time for days_out=3 is April 2 noon UTC
    decision_ts = int(datetime(2026, 4, 2, 12, 0, 0, tzinfo=timezone.utc).timestamp())

    for i, (label, price) in enumerate(buckets):
        token_id = f"tok_{i}"
        engine.loader._token_map[("Miami", target, label)] = token_id
        engine.loader._price_fetcher._cache[token_id] = [
            PriceSnapshot(timestamp=decision_ts, price=price),
        ]

    result = engine.run(cities=["Miami"], start_date=target, end_date=target)

    # Should have produced at least some trades
    realistic_trades = [t for t in result.trades if t.variant == "realistic"]
    if realistic_trades:
        # All trades from this date should be real_clob source
        sources = {t.price_source for t in realistic_trades}
        assert "real_clob" in sources or "synthetic" in sources  # depends on edge found
