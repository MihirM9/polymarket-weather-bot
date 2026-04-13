import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta

from backtesting import MockTracker
from forecasting import CityForecast
from trading.decision import DecisionEngine
from trading.markets import MarketOutcome, TemperatureMarket


def test_decision_engine_generates_signal_for_clear_edge():
    engine = DecisionEngine()
    tracker = MockTracker(bankroll=500.0)
    market_date = date.today() + timedelta(days=1)
    market = TemperatureMarket(
        market_id="mkt1",
        question="What will the highest temperature be in Miami on tomorrow?",
        city="Miami",
        market_date=market_date,
        resolution_source="NWS",
        outcomes=[
            MarketOutcome(
                outcome_label="89-90°F",
                token_id="tok1",
                price_yes=0.20,
                price_no=0.80,
                bucket_low=89.0,
                bucket_high=91.0,
            )
        ],
    )
    forecast = CityForecast(
        city="Miami",
        forecast_date=market_date,
        high_f=90.0,
        sigma=1.2,
        is_stable=True,
        confidence=0.95,
        weather_regime="stable",
        regime_multiplier=1.0,
    )

    signals = engine.evaluate([(market, forecast)], tracker=tracker)

    assert len(signals) == 1
    assert signals[0].side == "BUY"


def test_decision_engine_skips_duplicate_active_order():
    engine = DecisionEngine()
    tracker = MockTracker(bankroll=500.0)
    market_date = date.today() + timedelta(days=1)
    market = TemperatureMarket(
        market_id="mkt1",
        question="What will the highest temperature be in Miami on tomorrow?",
        city="Miami",
        market_date=market_date,
        resolution_source="NWS",
        outcomes=[
            MarketOutcome(
                outcome_label="89-90°F",
                token_id="tok1",
                price_yes=0.20,
                price_no=0.80,
                bucket_low=89.0,
                bucket_high=91.0,
            )
        ],
    )
    forecast = CityForecast(
        city="Miami",
        forecast_date=market_date,
        high_f=90.0,
        sigma=1.2,
        is_stable=True,
        confidence=0.95,
        weather_regime="stable",
        regime_multiplier=1.0,
    )
    tracker.record_trade("mkt1", "89-90°F", 5.0, "Miami")

    signals = engine.evaluate([(market, forecast)], tracker=tracker)

    assert signals == []


def test_decision_engine_shuts_down_after_daily_loss_cap_breach():
    engine = DecisionEngine()
    engine.update_pnl(-(engine.config.daily_loss_cap + 0.01))

    assert engine.is_shutdown() is True
