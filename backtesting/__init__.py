"""
backtesting package
===================
Organizes the historical research and replay stack under a single namespace.
"""

from .data import HistoricalDataLoader
from .forecast import HistoricalForecastApproximator, SyntheticForecast
from .pricing import MispricingModel
from .replay import BacktestEngine, BacktestResult
from .scorecard import BacktestScorecard, BacktestTrade, SensitivityAnalyzer
from .tracker import MockTracker

__all__ = [
    "HistoricalDataLoader",
    "HistoricalForecastApproximator",
    "SyntheticForecast",
    "MispricingModel",
    "BacktestEngine",
    "BacktestResult",
    "BacktestScorecard",
    "BacktestTrade",
    "SensitivityAnalyzer",
    "MockTracker",
]
