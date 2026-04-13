# tests/test_backtest_data.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date


def test_parse_noaa_observation_response():
    """Test parsing of NWS observations API response format."""
    from backtesting import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir="/tmp/backtest_test_data")

    mock_response = {
        "features": [
            {"properties": {"temperature": {"value": 28.3}}},
            {"properties": {"temperature": {"value": 30.0}}},
            {"properties": {"temperature": {"value": 27.1}}},
            {"properties": {"temperature": {"value": None}}},
        ]
    }
    result = loader._extract_max_temp(mock_response)
    assert result is not None
    assert abs(result - 86.0) < 0.1

def test_parse_noaa_empty_response():
    from backtesting import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir="/tmp/backtest_test_data")
    assert loader._extract_max_temp({"features": []}) is None
    assert loader._extract_max_temp({}) is None

def test_cache_write_and_read(tmp_path):
    from backtesting import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir=str(tmp_path))

    loader._cache_daily_high("Miami", date(2025, 7, 15), 91.2)
    loader._cache_daily_high("Miami", date(2025, 7, 16), 89.5)

    cached = loader._load_cached_highs()
    assert cached[("Miami", date(2025, 7, 15))] == 91.2
    assert cached[("Miami", date(2025, 7, 16))] == 89.5

def test_climatology_lookup():
    from backtesting import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir="/tmp/backtest_test_data")

    loader._climatology = {
        "Miami": {1: 76.5, 196: 91.3},
        "New York": {1: 38.2, 196: 84.1},
    }
    assert loader.get_climatology("Miami", date(2025, 7, 15)) == 91.3
    assert loader.get_climatology("Miami", date(2025, 1, 1)) == 76.5
    assert loader.get_climatology("Dallas", date(2025, 1, 1)) is None

def test_gamma_market_parsing():
    """Test parsing of closed Gamma API temperature market."""
    from backtesting import HistoricalDataLoader
    loader = HistoricalDataLoader(data_dir="/tmp/backtest_test_data")

    mock_market = {
        "id": "12345",
        "question": "What will the highest temperature be in Miami on July 15?",
        "groupItemTitle": "82-83°F",
        "outcomePrices": "[0.35, 0.65]",
        "clobTokenIds": "[\"token_abc\", \"token_def\"]",
        "negRiskMarketID": "group_xyz",
        "active": False,
        "closed": True,
        "volume": "5000",
    }
    city, mkt_date, bucket_lo, bucket_hi, price_yes = loader._parse_gamma_market(mock_market)
    assert city == "Miami"
    assert mkt_date == date(2026, 7, 15)
    assert bucket_lo == 82.0
    assert bucket_hi == 84.0
    assert abs(price_yes - 0.35) < 0.01
