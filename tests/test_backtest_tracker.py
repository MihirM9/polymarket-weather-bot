# tests/test_backtest_tracker.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date

def test_mock_tracker_enforces_per_market_cap():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    assert t.can_trade("mkt1", "bucket_a", 10.0, "New York") is True
    t.record_trade("mkt1", "bucket_a", 10.0, "New York")
    assert t.can_trade("mkt1", "bucket_b", 10.0, "New York") is False

def test_mock_tracker_dedup():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    t.record_trade("mkt1", "72-73°F", 5.0, "Miami")
    assert t.has_active_order("mkt1", "72-73°F") is True
    assert t.has_active_order("mkt1", "74-75°F") is False

def test_mock_tracker_daily_reset():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    t.record_trade("mkt1", "72-73°F", 10.0, "Miami")
    t.reset_day()
    assert t.has_active_order("mkt1", "72-73°F") is False
    assert t.daily_exposure == 0.0

def test_mock_tracker_correlated_group_cap():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    t.record_trade("mkt1", "a", 10.0, "New York")
    t.record_trade("mkt2", "b", 10.0, "Chicago")
    assert t.can_trade("mkt3", "c", 5.0, "New York") is False

def test_mock_tracker_daily_cap():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    for i in range(15):
        t.record_trade(f"mkt{i}", f"b{i}", 10.0, "Miami")
    assert t.can_trade("mkt99", "x", 5.0, "Houston") is False

def test_mock_tracker_exposure_properties():
    from backtest_tracker import MockTracker
    t = MockTracker(bankroll=500.0)
    t.record_trade("mkt1", "a", 8.0, "New York")
    t.record_trade("mkt2", "b", 5.0, "Miami")
    assert t.total_exposure == 13.0
    assert t.daily_exposure == 13.0
