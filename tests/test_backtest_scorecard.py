# tests/test_backtest_scorecard.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date


def _make_trade(city, target_date, days_out, side, price, size, won, pnl, variant="realistic"):
    from backtest_scorecard import BacktestTrade
    return BacktestTrade(
        city=city, target_date=target_date, days_out=days_out,
        side=side, outcome_label="72-73°F", bucket_low=72, bucket_high=74,
        p_true=0.30, market_price=price, ev=0.15, edge=0.10,
        kelly_fraction=0.05, position_size_usd=size, price_limit=price,
        actual_high=73.0, won=won, pnl=pnl, variant=variant,
        regime="normal",
    )

def test_win_rate():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("Miami", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 5.0),
        _make_trade("Miami", date(2025, 1, 2), 3, "BUY", 0.30, 10, True, 4.0),
        _make_trade("Miami", date(2025, 1, 3), 3, "BUY", 0.30, 10, False, -10.0),
    ]
    sc = BacktestScorecard(trades)
    assert abs(sc.win_rate("realistic") - 2/3) < 0.01

def test_max_drawdown():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("NY", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 10.0),
        _make_trade("NY", date(2025, 1, 2), 3, "BUY", 0.30, 10, True, 10.0),
        _make_trade("NY", date(2025, 1, 3), 3, "BUY", 0.30, 10, False, -15.0),
        _make_trade("NY", date(2025, 1, 4), 3, "BUY", 0.30, 10, False, -10.0),
        _make_trade("NY", date(2025, 1, 5), 3, "BUY", 0.30, 10, True, 8.0),
    ]
    sc = BacktestScorecard(trades)
    assert sc.max_drawdown("realistic") == 25.0

def test_profit_factor():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("LA", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 20.0),
        _make_trade("LA", date(2025, 1, 2), 3, "BUY", 0.30, 10, False, -10.0),
    ]
    sc = BacktestScorecard(trades)
    assert abs(sc.profit_factor("realistic") - 2.0) < 0.01

def test_breakdown_by_city():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("Miami", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 5.0),
        _make_trade("Miami", date(2025, 1, 2), 3, "BUY", 0.30, 10, True, 3.0),
        _make_trade("New York", date(2025, 1, 1), 3, "BUY", 0.30, 10, False, -10.0),
    ]
    sc = BacktestScorecard(trades)
    by_city = sc.breakdown_by_city("realistic")
    assert by_city["Miami"]["win_rate"] == 1.0
    assert by_city["Miami"]["pnl"] == 8.0
    assert by_city["New York"]["win_rate"] == 0.0

def test_breakdown_by_horizon():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("Miami", date(2025, 1, 1), 0, "BUY", 0.30, 10, True, 5.0),
        _make_trade("Miami", date(2025, 1, 2), 0, "BUY", 0.30, 10, True, 5.0),
        _make_trade("Miami", date(2025, 1, 3), 5, "BUY", 0.30, 10, False, -10.0),
    ]
    sc = BacktestScorecard(trades)
    by_horizon = sc.breakdown_by_horizon("realistic")
    assert by_horizon[0]["win_rate"] == 1.0
    assert by_horizon[5]["win_rate"] == 0.0

def test_variant_comparison():
    from backtest_scorecard import BacktestScorecard

    trades = [
        _make_trade("Miami", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 5.0, "realistic"),
        _make_trade("Miami", date(2025, 1, 1), 3, "BUY", 0.30, 10, True, 8.0, "optimistic"),
    ]
    sc = BacktestScorecard(trades)
    assert sc.win_rate("realistic") == 1.0
    assert sc.win_rate("optimistic") == 1.0
