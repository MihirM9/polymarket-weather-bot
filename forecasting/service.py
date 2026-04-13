"""
forecasting.service — Live forecast orchestration
=================================================
Encapsulates the "forecast ingestion + supplemental blending + same-day
observation adjustments" workflow so the main loop does not own that detail.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import aiohttp

from config import cfg, Config

from .blender import EnsembleBlender
from .metar import MetarFetcher
from .scanner import CityForecast, ForecastScanner, compute_confidence

logger = logging.getLogger(__name__)


class ForecastingService:
    """Coordinates forecast collection and blending for the trading loop."""

    def __init__(
        self,
        scanner: ForecastScanner,
        blender: EnsembleBlender,
        metar_fetcher: MetarFetcher,
        config: Config = cfg,
    ) -> None:
        self.config = config
        self.scanner = scanner
        self.blender = blender
        self.metar_fetcher = metar_fetcher

    async def enrich_forecasts(self, forecasts: List[CityForecast]) -> List[CityForecast]:
        """Blend supplemental sources and same-day observations into NWS forecasts."""
        if not forecasts:
            return forecasts

        cities_dates = [(forecast.city, forecast.forecast_date) for forecast in forecasts]
        supplemental = await self.blender.fetch_all_supplemental(cities_dates)
        metar_observations = await self.metar_fetcher.fetch_all()
        station_observations = await self._collect_station_observations(forecasts)

        for forecast in forecasts:
            self._apply_same_day_observation_adjustments(
                forecast,
                metar_observations,
                station_observations,
            )
            key = f"{forecast.city}_{forecast.forecast_date.isoformat()}"
            owm_point = supplemental.get(key)
            ensemble = self.blender.blend(
                forecast.high_f,
                forecast.sigma,
                [owm_point] if owm_point else [],
            )
            forecast.high_f = ensemble.blended_high
            forecast.sigma = ensemble.ensemble_sigma
            forecast.confidence = compute_confidence(
                forecast.sigma, forecast.is_stable, forecast.regime_multiplier
            )

        return forecasts

    async def _collect_station_observations(
        self,
        forecasts: List[CityForecast],
    ) -> Dict[str, float]:
        same_day_cities = {
            forecast.city for forecast in forecasts
            if forecast.forecast_date == self.config.city_local_date(forecast.city)
        }
        observations: Dict[str, float] = {}
        if not same_day_cities:
            return observations

        async with aiohttp.ClientSession() as session:
            for city in same_day_cities:
                station_id = self.config.noaa_stations.get(city)
                if not station_id:
                    continue
                temp_f = await self.scanner.fetch_station_observation(session, station_id)
                if temp_f is not None:
                    observations[city] = temp_f
        return observations

    def _apply_same_day_observation_adjustments(
        self,
        forecast: CityForecast,
        metar_observations: dict,
        station_observations: Dict[str, float],
    ) -> None:
        if forecast.forecast_date != self.config.city_local_date(forecast.city):
            return

        current_temps = []
        metar_obs = metar_observations.get(forecast.city)
        if metar_obs:
            current_temps.append(metar_obs.temp_f)
        station_temp = station_observations.get(forecast.city)
        if station_temp is not None:
            current_temps.append(station_temp)

        if not current_temps:
            return

        max_observed = max(current_temps)
        if max_observed > forecast.high_f:
            logger.info(
                f"Observed temp {max_observed:.0f}°F > forecast "
                f"{forecast.high_f:.0f}°F for {forecast.city} — raising floor"
            )
            forecast.high_f = max_observed
            forecast.sigma = max(forecast.sigma, 1.5)

        divergence = abs(forecast.high_f - max_observed)
        if divergence > 10:
            sigma_boost = 1.0 + (divergence - 10) * 0.05
            forecast.sigma *= sigma_boost
            logger.info(
                f"Large obs/forecast divergence ({divergence:.0f}°F) "
                f"for {forecast.city} — σ boosted to {forecast.sigma:.1f}"
            )
