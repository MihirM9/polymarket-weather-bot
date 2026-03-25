"""
ensemble_blender.py — v3 Fix #2: Multi-Source Forecast Blending
================================================================
Addresses the single-source dependency vulnerability: the bot previously
relied exclusively on NWS for forecasts. If NWS has a model bias or
experiences downtime, the bot has no cross-validation.

This module:
  1. Fetches forecasts from OpenWeatherMap (free tier: 1000 calls/day).
  2. Blends NWS + OWM forecasts using inverse-variance weighting.
  3. Computes dynamic sigma from inter-model disagreement rather than
     a static horizon lookup table.

The key insight (from Gemini's stress test): if NWS predicts 82°F and
OWM predicts 86°F, sigma should naturally widen to reflect the 4°F
disagreement — regardless of forecast horizon. This automatically triggers
the dynamic edge threshold, requiring higher profit margins when
meteorologists disagree.

Requires: OPENWEATHER_API_KEY in .env (free at openweathermap.org/api).
If not configured, the blender falls through to NWS-only mode gracefully.
"""

import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Dict, List, Optional, Tuple

import aiohttp

from api_utils import fetch_with_retry
from config import cfg

logger = logging.getLogger(__name__)

OWM_BASE = "https://api.openweathermap.org/data/3.0/onecall"
OWM_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")

# Minimum sigma floor — even if all models agree perfectly, we never
# assume zero uncertainty. Weather is stochastic.
SIGMA_FLOOR = 1.0  # °F

# Model reliability weights (tunable based on backtesting).
# Higher weight = more trust in that source.
MODEL_WEIGHTS = {
    "nws": 1.0,       # NWS is the primary, most trusted source
    "owm": 0.7,       # OWM is supplementary, slightly less accurate for US cities
}

# City-specific peak temperature hour in UTC.
# Daily high typically occurs 2-4 PM local time.
# These approximate the afternoon peak for each city's timezone.
CITY_PEAK_HOUR_UTC = {
    "New York": 19,      # ~3 PM EDT (UTC-4) or ~2 PM EST (UTC-5)
    "Chicago": 20,       # ~3 PM CDT (UTC-5) or ~2 PM CST (UTC-6)
    "Los Angeles": 23,   # ~4 PM PDT (UTC-7) or ~3 PM PST (UTC-8)
    "Miami": 19,         # ~3 PM EDT (UTC-4)
    "Houston": 20,       # ~3 PM CDT (UTC-5)
    "Dallas": 20,        # ~3 PM CDT (UTC-5)
}
DEFAULT_PEAK_HOUR_UTC = 20  # fallback


@dataclass
class ForecastPoint:
    """A single temperature forecast from one source."""
    source: str       # "nws" or "owm"
    high_f: float     # Forecast high in °F
    sigma: float      # Source-specific uncertainty (from horizon model)
    weight: float     # Reliability weight


@dataclass
class EnsembleForecast:
    """Blended forecast from multiple sources."""
    blended_high: float       # Weighted mean forecast
    ensemble_sigma: float     # Uncertainty from inter-model spread + base uncertainty
    source_count: int         # How many sources contributed
    sources: List[ForecastPoint]
    model_spread: float       # Max - min across sources (raw disagreement)

    @property
    def is_single_source(self) -> bool:
        return self.source_count <= 1


class EnsembleBlender:
    """
    Fetches supplemental forecasts and blends with NWS data.

    Blending strategy: inverse-variance weighted mean.
      - Each source contributes a (high_temp, sigma) pair.
      - The blended mean is the weighted average of highs.
      - The blended sigma combines two uncertainty components:
        (a) Base uncertainty: weighted average of individual sigmas.
        (b) Model disagreement: standard deviation across source forecasts.
      - Final sigma = sqrt(base_uncertainty^2 + model_disagreement^2).

    This means:
      - If NWS=82 and OWM=82 → sigma stays tight (models agree).
      - If NWS=82 and OWM=86 → sigma widens automatically (4°F spread).
      - If only NWS available → falls back to NWS sigma (no degradation).
    """

    def __init__(self):
        self.enabled = bool(OWM_API_KEY)
        if not self.enabled:
            logger.info("Ensemble blending disabled (no OPENWEATHER_API_KEY). "
                        "Set it in .env for multi-source forecasts.")
        self._owm_cache: Dict[str, Dict[str, float]] = {}  # "city_date" → {high_f, fetched_at}

    def prune_cache(self):
        """Remove cached forecasts for past dates to prevent memory growth."""
        today = date.today()
        stale_keys = []
        for key in list(self._owm_cache.keys()):
            # Key format: "city_YYYY-MM-DD"
            try:
                date_str = key.rsplit("_", 1)[1]
                cached_date = date.fromisoformat(date_str)
                if cached_date < today:
                    stale_keys.append(key)
            except (IndexError, ValueError):
                stale_keys.append(key)
        for key in stale_keys:
            del self._owm_cache[key]
        if stale_keys:
            logger.debug(f"Pruned {len(stale_keys)} stale OWM cache entries")

    async def fetch_owm_forecast(
        self,
        session: aiohttp.ClientSession,
        city: str,
        target_date: date,
    ) -> Optional[ForecastPoint]:
        """
        Fetch OpenWeatherMap One Call 3.0 forecast for a city/date.
        Returns ForecastPoint or None if unavailable.
        """
        if not self.enabled:
            return None

        coords = cfg.nws_points.get(city)
        if not coords:
            return None

        lat, lon = coords
        cache_key = f"{city}_{target_date.isoformat()}"

        # Check cache (OWM rate limits: avoid redundant calls)
        if cache_key in self._owm_cache:
            cached = self._owm_cache[cache_key]
            return ForecastPoint(
                source="owm",
                high_f=cached["high_f"],
                sigma=cached["sigma"],
                weight=MODEL_WEIGHTS["owm"],
            )

        url = (
            f"{OWM_BASE}?lat={lat:.4f}&lon={lon:.4f}"
            f"&appid={OWM_API_KEY}&units=imperial&exclude=minutely,hourly,alerts"
        )
        data = await fetch_with_retry(session, url, label=f"OWM-{city}")
        if not data:
            return None

        # Find the daily forecast matching target_date
        daily = data.get("daily", [])
        for day in daily:
            dt_unix = day.get("dt", 0)
            forecast_date = datetime.fromtimestamp(dt_unix, tz=timezone.utc).date()
            if forecast_date == target_date:
                temp_max = day.get("temp", {}).get("max")
                if temp_max is None:
                    return None

                # OWM doesn't provide uncertainty directly;
                # estimate from "feels_like" spread as a proxy, or use horizon-based
                now = datetime.now(timezone.utc)
                peak_hour = CITY_PEAK_HOUR_UTC.get(city, DEFAULT_PEAK_HOUR_UTC)
                hours_out = max(0, (datetime(
                    target_date.year, target_date.month, target_date.day,
                    peak_hour, 0, tzinfo=timezone.utc  # city-specific afternoon peak
                ) - now).total_seconds() / 3600)

                # Use same horizon model as NWS but slightly inflated
                # (OWM is generally less accurate for US cities)
                from forecast_scanner import _sigma_for_horizon
                sigma = _sigma_for_horizon(hours_out, forecast_date=target_date) * 1.15

                result = ForecastPoint(
                    source="owm",
                    high_f=temp_max,
                    sigma=sigma,
                    weight=MODEL_WEIGHTS["owm"],
                )

                self._owm_cache[cache_key] = {
                    "high_f": temp_max,
                    "sigma": sigma,
                }

                return result

        return None  # target date not in forecast range

    def blend(
        self,
        nws_high: float,
        nws_sigma: float,
        supplemental: List[ForecastPoint],
    ) -> EnsembleForecast:
        """
        Blend NWS forecast with supplemental sources.

        Math:
          1. Weighted mean: high_blend = sum(w_i * h_i) / sum(w_i)
          2. Base uncertainty: sigma_base = sum(w_i * sigma_i) / sum(w_i)
          3. Model disagreement: sigma_spread = std_dev of all h_i values
          4. Final: sigma = sqrt(sigma_base^2 + sigma_spread^2)
        """
        # Build source list starting with NWS
        sources = [ForecastPoint(
            source="nws",
            high_f=nws_high,
            sigma=nws_sigma,
            weight=MODEL_WEIGHTS["nws"],
        )]
        sources.extend([s for s in supplemental if s is not None])

        if len(sources) == 1:
            # Single source — no blending, pass through
            return EnsembleForecast(
                blended_high=nws_high,
                ensemble_sigma=max(nws_sigma, SIGMA_FLOOR),
                source_count=1,
                sources=sources,
                model_spread=0.0,
            )

        # Weighted mean
        total_weight = sum(s.weight for s in sources)
        blended_high = sum(s.weight * s.high_f for s in sources) / total_weight

        # Base uncertainty (weighted average of individual sigmas)
        sigma_base = sum(s.weight * s.sigma for s in sources) / total_weight

        # Model disagreement (standard deviation of highs)
        mean_high = sum(s.high_f for s in sources) / len(sources)
        variance = sum((s.high_f - mean_high) ** 2 for s in sources) / len(sources)
        sigma_spread = math.sqrt(variance)

        # Combined sigma
        ensemble_sigma = math.sqrt(sigma_base ** 2 + sigma_spread ** 2)
        ensemble_sigma = max(ensemble_sigma, SIGMA_FLOOR)

        model_spread = max(s.high_f for s in sources) - min(s.high_f for s in sources)

        logger.info(
            f"Ensemble blend: {len(sources)} sources, "
            f"highs=[{', '.join(f'{s.source}={s.high_f:.0f}' for s in sources)}], "
            f"blended={blended_high:.1f}°F, "
            f"sigma_base={sigma_base:.2f}, sigma_spread={sigma_spread:.2f}, "
            f"ensemble_sigma={ensemble_sigma:.2f}, model_spread={model_spread:.1f}°F"
        )

        return EnsembleForecast(
            blended_high=blended_high,
            ensemble_sigma=ensemble_sigma,
            source_count=len(sources),
            sources=sources,
            model_spread=model_spread,
        )

    async def fetch_all_supplemental(
        self,
        cities_dates: List[Tuple[str, date]],
    ) -> Dict[str, ForecastPoint]:
        """
        Batch-fetch supplemental forecasts for all city/date pairs.
        Returns dict keyed by "city_date" → ForecastPoint.
        """
        self.prune_cache()

        if not self.enabled:
            return {}

        results: Dict[str, ForecastPoint] = {}
        async with aiohttp.ClientSession() as session:
            tasks = []
            keys = []
            for city, dt in cities_dates:
                key = f"{city}_{dt.isoformat()}"
                keys.append(key)
                tasks.append(self.fetch_owm_forecast(session, city, dt))

            fetched = await asyncio.gather(*tasks, return_exceptions=True)
            for key, result in zip(keys, fetched):
                if isinstance(result, ForecastPoint):
                    results[key] = result
                elif isinstance(result, Exception):
                    logger.debug(f"OWM fetch error for {key}: {result}")

        logger.info(f"Fetched {len(results)} supplemental forecasts from OWM")
        return results
