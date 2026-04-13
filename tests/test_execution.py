import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from datetime import date, datetime, timezone

from decision_engine import TradeSignal
from dry_run_simulator import OrderbookSnapshot, SimulatedFill
from execution import OrderExecutor, TradeLogger
from position_tracker import PositionTracker


def test_trade_logger_keeps_recent_trades():
    logger = TradeLogger()
    signal = TradeSignal(
        market_id="mkt1",
        city="Miami",
        market_date=date(2026, 4, 12),
        outcome_label="80-81°F",
        token_id="tok1",
        side="BUY",
        p_true=0.55,
        market_price=0.40,
        ev=0.10,
        edge=0.15,
        kelly_fraction=0.03,
        position_size_usd=5.0,
        price_limit=0.41,
        rationale="test signal",
    )

    class DummyOrder:
        order_id = "ord1"
        status = type("Status", (), {"value": "filled"})()
        filled_size_usd = 5.0
        avg_fill_price = 0.41

    logger.log_trade(signal, DummyOrder(), True, slippage=0.0, fill_ratio=1.0, is_maker=True, book_depth=10.0)

    recent = logger.recent_trades
    assert len(recent) == 1
    assert recent[0]["bucket"] == "80-81°F"


def test_order_executor_uses_simulated_fill_metadata(monkeypatch):
    monkeypatch.setattr(PositionTracker, "_load_state", lambda self: None)
    tracker = PositionTracker()
    executor = OrderExecutor(tracker)

    async def fake_fetch_orderbook(session, token_id):
        return OrderbookSnapshot(
            token_id=token_id,
            bids=[(0.40, 100.0)],
            asks=[(0.42, 100.0)],
            timestamp=datetime.now(timezone.utc),
        )

    def fake_simulate_fill(snapshot, side, intended_size_usd, limit_price, is_maker=True):
        return SimulatedFill(
            filled_size_usd=intended_size_usd,
            filled_shares=intended_size_usd / limit_price,
            avg_fill_price=limit_price,
            slippage=0.0,
            fill_ratio=1.0,
            is_maker=True,
            estimated_fill_cycles=0,
        )

    monkeypatch.setattr(executor.simulator, "fetch_orderbook", fake_fetch_orderbook)
    monkeypatch.setattr(executor.simulator, "simulate_fill", fake_simulate_fill)

    signal = TradeSignal(
        market_id="mkt1",
        city="Miami",
        market_date=date(2026, 4, 12),
        outcome_label="80-81°F",
        token_id="tok1",
        side="BUY",
        p_true=0.55,
        market_price=0.40,
        ev=0.10,
        edge=0.15,
        kelly_fraction=0.03,
        position_size_usd=5.0,
        price_limit=0.41,
        rationale="test signal",
    )

    order = asyncio.run(executor.execute_signal(signal))

    assert order is not None
    assert order.simulated_is_maker is True
    assert order.simulated_fill_ratio == 1.0
    assert tracker.total_exposure >= 5.0
