"""
forecast_scanner.py — Module 1: NWS Forecast Scanner
=====================================================
Ref: Research §4.2 (NOAA/NWS ~90% 5-day accuracy, 1-2°F short-term),
     §5.1 (Translating forecasts into bucket probabilities),
     §2.2 (Data availability and automation friendliness).

Queries api.weather.gov for each configured city:
  1. /points/{lat},{lon}          → grid office + grid coordinates
  2. /gridpoints/{office}/{x},{y} → detailed forecast with temperature data

Produces a probability distribution over temperature buckets using a
Gaussian model centered on the NWS point forecast with configurable
uncertainty (σ ≈ 1.5°F for 0-24h, scaling up for longer horizons).
"""

import asyncio
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Dict, List, Optional, Tuple, cast

import aiohttp

from config import Config, cfg
from infrastructure.http import fetch_with_retry
from infrastructure.models import (
    NWSForecastResponse,
    NWSLatestObservationResponse,
    NWSPointsResponse,
    validate_model,
)

logger = logging.getLogger(__name__)

# NWS API base
NWS_BASE = "https://api.weather.gov"
NWS_HEADERS = {"User-Agent": "(polymarket-weather-bot, contact@example.com)", "Accept": "application/geo+json"}

# Uncertainty model (§4.2: 1-2°F short-term).
# σ in °F by forecast horizon bucket (hours out).
SIGMA_BY_HORIZON: List[Tuple[int, float]] = [
    (24, 1.5),    # 0-24h: very tight
    (48, 2.0),    # 24-48h
    (72, 2.5),    # 48-72h
    (120, 3.5),   # 3-5 days
    (168, 5.0),   # 5-7 days
]


# Seasonal sigma multipliers — spring is more volatile, mid-summer is stable.
# Ref: §5.1 known limitation #5 (seasonal σ adjustment).
SEASONAL_SIGMA_MULT: Dict[int, float] = {
    1: 1.15,   # January — winter storms
    2: 1.2,    # February — late winter variability
    3: 1.3,    # March — spring transition, high volatility
    4: 1.35,   # April — peak spring volatility
    5: 1.2,    # May — settling into summer pattern
    6: 1.0,    # June — stable
    7: 0.9,    # July — mid-summer, very stable
    8: 0.9,    # August — mid-summer, very stable
    9: 1.0,    # September — early fall
    10: 1.15,  # October — fall transition
    11: 1.2,   # November — late fall variability
    12: 1.15,  # December — winter
}


def _sigma_for_horizon(hours_out: float, forecast_date: Optional[date] = None) -> float:
    """Return forecast uncertainty σ (°F) given hours until resolution.

    If forecast_date is provided, applies seasonal sigma multiplier
    (spring months are more volatile than mid-summer).
    """
    base_sigma = 6.0  # beyond 7 days — very uncertain
    for threshold, sigma in SIGMA_BY_HORIZON:
        if hours_out <= threshold:
            base_sigma = sigma
            break

    if forecast_date is not None:
        seasonal_mult = SEASONAL_SIGMA_MULT.get(forecast_date.month, 1.0)
        base_sigma *= seasonal_mult

    return base_sigma


def _normal_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal CDF via math.erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def bucket_probabilities(
    forecast_high: float,
    sigma: float,
    buckets: List[Tuple[Optional[float], Optional[float]]],
) -> Dict[int, float]:
    """
    Compute P(temp in bucket) for each bucket using Gaussian model.
    Ref: §5.1 — Integrate forecast distribution over bucket ranges.

    buckets: list of (low_bound, high_bound) in °F.
             None means unbounded on that side.
    Returns: {bucket_index: probability}
    """
    probs = {}
    for i, (lo, hi) in enumerate(buckets):
        cdf_lo = _normal_cdf(lo, forecast_high, sigma) if lo is not None else 0.0
        cdf_hi = _normal_cdf(hi, forecast_high, sigma) if hi is not None else 1.0
        probs[i] = max(0.0, cdf_hi - cdf_lo)

    # Do NOT normalize. Raw Gaussian CDF probabilities are correct.
    # Normalization caused a critical bug: when Polymarket only lists partial
    # buckets (e.g., upper tail only), normalizing inflates probabilities.
    # Example: forecast 68°F, only buckets 70°F+ listed → raw P(70-72)=7.5%,
    # but after normalizing over just those buckets → 97%! (fake edge)
    return probs


def _has_negation_before(text: str, keyword: str) -> bool:
    """Check if keyword in text is preceded by negation within ~30 chars."""
    NEGATIONS = {"no ", "not ", "no longer ", "without ", "unlikely ", "not expecting "}
    idx = text.find(keyword)
    if idx < 0:
        return False
    # Check the 30 characters before the keyword for negation
    prefix = text[max(0, idx - 30):idx].lower()
    return any(neg in prefix for neg in NEGATIONS)


def _detect_weather_regime(short_forecast: str) -> Tuple[str, float]:
    """
    Detect weather regime from NWS detailed forecast text.
    Returns (regime_name, sigma_multiplier).

    Certain weather patterns make temperature forecasts less reliable:
    - Frontal passages: cold/warm fronts can shift timing ±hours → ±3-5°F
    - Tropical systems: highly unpredictable temperature impacts
    - Thunderstorms: convective cooling can drop temps 10°F+ suddenly
    - Clear/stable: high confidence, no sigma inflation needed

    Negation-aware: phrases like "No hurricane expected" won't trigger
    the tropical regime. Checks for negation words within ~30 chars
    before each keyword match.
    """
    text = short_forecast.lower() if short_forecast else ""

    # Most volatile → least volatile (negation-aware — §4.2)
    if any(w in text and not _has_negation_before(text, w) for w in ["hurricane", "tropical storm", "tropical depression"]):
        return ("tropical", 2.5)
    if any(w in text and not _has_negation_before(text, w) for w in ["severe thunderstorm", "tornado", "severe"]):
        return ("severe", 2.0)
    if any(w in text and not _has_negation_before(text, w) for w in ["thunderstorm", "tstorm", "t-storm"]):
        return ("convective", 1.5)
    if any(w in text and not _has_negation_before(text, w) for w in ["cold front", "warm front", "frontal", "front passage"]):
        return ("frontal", 1.4)
    if any(w in text and not _has_negation_before(text, w) for w in ["wind advisory", "high wind"]):
        return ("windy", 1.2)
    if any(w in text and not _has_negation_before(text, w) for w in ["fog", "patchy fog"]):
        return ("fog", 1.1)  # minimal temp impact
    if any(w in text for w in ["sunny", "clear", "mostly sunny", "mostly clear"]):
        return ("stable", 0.9)  # slightly tighter than default (no negation check needed)

    # Log if any weather keywords were present but all negated
    _volatile_keywords = [
        "hurricane", "tropical storm", "tropical depression",
        "severe thunderstorm", "tornado", "severe",
        "thunderstorm", "tstorm", "t-storm",
        "cold front", "warm front", "frontal", "front passage",
        "wind advisory", "high wind", "fog", "patchy fog",
    ]
    if any(w in text and _has_negation_before(text, w) for w in _volatile_keywords):
        logger.debug(f"Negated weather keyword detected in: {text[:80]}")

    return ("normal", 1.0)


def compute_confidence(sigma: float, is_stable: bool, regime_multiplier: float) -> float:
    """
    Compute a 0-1 confidence score for a forecast.
    Used downstream by the decision engine to scale Kelly tempering and edge thresholds.

    High confidence (→1.0): tight sigma, stable forecast, benign weather regime.
    Low confidence (→0.0): wide sigma, unstable, volatile regime.
    """
    # Base confidence from sigma (1.5°F → ~0.95, 5.0°F → ~0.5)
    sigma_conf = max(0.0, min(1.0, 1.0 - (sigma - 1.0) / 8.0))

    # Stability penalty
    stability_factor = 1.0 if is_stable else 0.6

    # Regime factor (inverted: multiplier 2.5 → low confidence)
    regime_conf = max(0.0, min(1.0, 1.0 / regime_multiplier))

    confidence = sigma_conf * stability_factor * regime_conf
    return round(max(0.0, min(1.0, confidence)), 4)


@dataclass
class CityForecast:
    """Forecast result for one city on one date."""
    city: str
    forecast_date: date
    high_f: float                # NWS point forecast high (°F)
    sigma: float                 # uncertainty σ (°F), includes regime inflation
    sigma_base: float = 0.0      # base σ before regime inflation
    hours_out: float = 0.0       # hours until resolution date
    is_stable: bool = True       # False if rapid change detected (§ skip volatile)
    confidence: float = 1.0      # 0-1 confidence score for downstream Kelly/edge scaling
    weather_regime: str = "normal"
    regime_multiplier: float = 1.0
    raw_periods: list = field(default_factory=list)


class ForecastScanner:
    """Async scanner that fetches NWS forecasts for all configured cities."""

    def __init__(self, config: Config = cfg):
        self.config = config
        self._grid_cache: Dict[str, Tuple[str, int, int]] = {}  # city → (office, gridX, gridY)
        self._last_forecasts: Dict[str, float] = {}  # city → last high for stability check

    async def _get_grid(self, session: aiohttp.ClientSession, city: str) -> Optional[Tuple[str, int, int]]:
        """Resolve city lat/lon → NWS grid office and coordinates. Cached."""
        if city in self._grid_cache:
            return self._grid_cache[city]

        coords = self.config.nws_points.get(city)
        if not coords:
            logger.warning(f"No NWS point configured for {city}")
            return None

        lat, lon = coords
        url = f"{NWS_BASE}/points/{lat:.4f},{lon:.4f}"
        data = await fetch_with_retry(session, url, headers=NWS_HEADERS, label=f"NWS-points-{city}")
        if not data:
            return None
        parsed = validate_model(data, NWSPointsResponse, label=f"NWS-points-{city}")
        if parsed is None:
            return None
        office = parsed.properties.gridId
        gx, gy = parsed.properties.gridX, parsed.properties.gridY
        self._grid_cache[city] = (office, gx, gy)
        return (office, gx, gy)

    async def _get_forecast(self, session: aiohttp.ClientSession, city: str) -> List[CityForecast]:
        """Fetch NWS gridpoint forecast and extract daily highs."""
        grid = await self._get_grid(session, city)
        if not grid:
            return []

        office, gx, gy = grid
        url = f"{NWS_BASE}/gridpoints/{office}/{gx},{gy}/forecast"
        data = await fetch_with_retry(session, url, headers=NWS_HEADERS, label=f"NWS-forecast-{city}")
        if not data:
            return []
        parsed = validate_model(data, NWSForecastResponse, label=f"NWS-forecast-{city}")
        if parsed is None:
            return []

        periods = parsed.properties.periods
        results: List[CityForecast] = []
        now = datetime.now(timezone.utc)

        for p in periods:
            if not p.isDaytime:
                continue
            temp = p.temperature
            unit = p.temperatureUnit
            if temp is None:
                continue
            if unit == "C":
                temp = temp * 9.0 / 5.0 + 32.0

            # Parse start time to get the forecast date
            start_str = p.startTime
            try:
                start_dt = datetime.fromisoformat(start_str)
                forecast_date = start_dt.date()
            except (ValueError, TypeError):
                continue

            hours_out = max(0.0, (start_dt - now).total_seconds() / 3600.0)
            sigma_base = _sigma_for_horizon(hours_out, forecast_date=forecast_date)

            # Weather regime detection — inflate σ for volatile patterns
            detail_text = p.detailedForecast or p.shortForecast
            regime, regime_mult = _detect_weather_regime(detail_text)
            sigma = sigma_base * regime_mult

            # Stability check: flag if forecast swung >4°F from last scan
            cache_key = f"{city}_{forecast_date}"
            is_stable = True
            if cache_key in self._last_forecasts:
                delta = abs(temp - self._last_forecasts[cache_key])
                if delta > 4.0:
                    is_stable = False
                    logger.warning(f"Instability detected for {city} on {forecast_date}: "
                                   f"Δ={delta:.1f}°F (prev={self._last_forecasts[cache_key]:.0f}, now={temp:.0f})")
            self._last_forecasts[cache_key] = temp

            # Compute confidence score for downstream Kelly/edge scaling
            confidence = compute_confidence(sigma, is_stable, regime_mult)

            results.append(CityForecast(
                city=city,
                forecast_date=forecast_date,
                high_f=temp,
                sigma=sigma,
                sigma_base=sigma_base,
                hours_out=hours_out,
                is_stable=is_stable,
                confidence=confidence,
                weather_regime=regime,
                regime_multiplier=regime_mult,
                raw_periods=[p.model_dump()],
            ))

        return results

    async def scan_all(self) -> List[CityForecast]:
        """
        Scan all configured cities, return list of CityForecast objects.
        Ref: §7 — NOAA pipeline, scan every 2-5 min.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [self._get_forecast(session, city) for city in self.config.cities]
            all_results = await asyncio.gather(*tasks, return_exceptions=True)

        forecasts: List[CityForecast] = []
        for res in all_results:
            if isinstance(res, Exception):
                logger.error(f"Forecast scan exception: {res}")
                continue
            forecasts.extend(cast(List[CityForecast], res))

        logger.info(f"Forecast scan complete: {len(forecasts)} city-date pairs across {len(self.config.cities)} cities")
        return forecasts

    async def fetch_station_observation(
        self, session: aiohttp.ClientSession, station_id: str
    ) -> Optional[float]:
        """
        Fetch the latest observation from a specific NOAA station.
        Returns observed temperature in °F, or None on failure.

        Uses the NWS stations API: /stations/{station_id}/observations/latest
        Temperature is returned in °C by the API and converted to °F.
        """
        url = f"{NWS_BASE}/stations/{station_id}/observations/latest"
        data = await fetch_with_retry(
            session, url, headers=NWS_HEADERS, label=f"NWS-station-{station_id}"
        )
        if not data:
            return None
        parsed = validate_model(
            data,
            NWSLatestObservationResponse,
            label=f"NWS-station-{station_id}",
        )
        if parsed is None:
            return None
        temp_c = parsed.properties.temperature.value
        if temp_c is None:
            logger.debug(f"Station {station_id}: temperature value is null")
            return None
        temp_f = temp_c * 9.0 / 5.0 + 32.0
        logger.debug(f"Station {station_id}: observed {temp_f:.1f}°F ({temp_c:.1f}°C)")
        return temp_f
