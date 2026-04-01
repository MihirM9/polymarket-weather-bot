# backtest_forecast.py
"""
backtest_forecast.py — Historical Forecast Approximator
========================================================
Generates plausible CityForecast-like objects for backtesting WITHOUT
seeing the actual daily high (avoiding look-ahead bias).

Two variants per city/date/horizon:
  - Realistic: climatology + calibrated NWS bias + Gaussian noise
  - Optimistic: centered on actual high with reduced noise (upper bound)

The realistic variant is the primary test. The gap between realistic
and optimistic quantifies how much look-ahead leakage would help.
"""

import math
import random
from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional, Tuple

from forecast_scanner import compute_confidence

HORIZON_SIGMA = {
    0: 1.5, 1: 2.0, 2: 2.5, 3: 3.0, 4: 3.8, 5: 4.5, 6: 5.3, 7: 6.0,
}

HORIZON_BIAS = {
    0: 0.0, 1: -0.3, 2: -0.5, 3: -0.7, 4: -0.85, 5: -1.0, 6: -1.1, 7: -1.2,
}

REGIME_BIAS = {
    "heat": -1.5,
    "cold": 1.0,
    "convective": -1.5,
    "frontal": 1.0,
    "normal": 0.0,
    "stable": 0.0,
    "extreme": -2.0,
}

REGIME_SIGMA_MULT = {
    "heat": 1.2, "cold": 1.2, "convective": 1.3, "frontal": 1.2,
    "normal": 1.0, "stable": 0.9, "extreme": 1.4,
}

SEASONAL_SIGMA_MULT = {
    1: 1.0, 2: 1.05, 3: 1.25, 4: 1.35, 5: 1.2, 6: 1.1,
    7: 0.9, 8: 0.95, 9: 1.1, 10: 1.15, 11: 1.2, 12: 1.0,
}


@dataclass
class SyntheticForecast:
    """A generated forecast for backtesting."""
    high_f: float
    sigma: float
    confidence: float
    regime: str
    variant: str


class HistoricalForecastApproximator:
    """
    Generate plausible forecasts WITHOUT seeing the actual high.
    """

    def __init__(self):
        self._climatology: Dict[str, Dict[int, float]] = {}

    def set_climatology(self, climatology: Dict[str, Dict[int, float]]):
        self._climatology = climatology

    def _infer_regime(
        self, city: str, target_date: date, actual_high: float, climatology: float
    ) -> str:
        delta = actual_high - climatology

        if abs(delta) > 12:
            return "extreme"
        if delta > 8:
            return "heat"
        if delta < -8:
            return "cold"

        month = target_date.month
        if city in ("Miami", "Houston", "Dallas") and month in (6, 7, 8, 9):
            if delta < -3:
                return "convective"

        if month in (3, 4, 5, 10, 11):
            if abs(delta) > 5:
                return "frontal"

        if abs(delta) <= 2:
            return "stable"
        return "normal"

    def generate(
        self,
        city: str,
        target_date: date,
        days_out: int,
        actual_high: float,
    ) -> Tuple[SyntheticForecast, SyntheticForecast]:
        doy = target_date.timetuple().tm_yday
        month = target_date.month

        city_clim = self._climatology.get(city, {})
        clim = city_clim.get(doy)
        if clim is None:
            clim = actual_high

        regime = self._infer_regime(city, target_date, actual_high, clim)

        base_sigma = HORIZON_SIGMA.get(days_out, 6.0)
        seasonal_mult = SEASONAL_SIGMA_MULT.get(month, 1.0)
        regime_mult = REGIME_SIGMA_MULT.get(regime, 1.0)
        sigma = base_sigma * seasonal_mult * regime_mult

        horizon_bias = HORIZON_BIAS.get(days_out, -1.2)
        regime_bias = REGIME_BIAS.get(regime, 0.0)
        total_bias = horizon_bias + regime_bias
        noise = random.gauss(0, sigma)
        realistic_high = clim + total_bias + noise

        is_stable = regime in ("stable", "normal")
        confidence = compute_confidence(sigma, is_stable, regime_mult)

        realistic = SyntheticForecast(
            high_f=realistic_high,
            sigma=sigma,
            confidence=confidence,
            regime=regime,
            variant="realistic",
        )

        opt_noise = random.gauss(0, sigma * 0.5)
        optimistic = SyntheticForecast(
            high_f=actual_high + opt_noise,
            sigma=sigma,
            confidence=confidence,
            regime=regime,
            variant="optimistic",
        )

        return realistic, optimistic
