"""
forecasting package
===================
Groups the live forecast ingestion and blending surface behind a single import
path so the app can reason about "forecasting" as one subsystem.
"""

from ensemble_blender import EnsembleBlender, EnsembleForecast, ForecastPoint
from forecast_scanner import CityForecast, ForecastScanner, bucket_probabilities, compute_confidence
from metar_fetcher import MetarFetcher, MetarObservation

from .service import ForecastingService

__all__ = [
    "CityForecast",
    "ForecastScanner",
    "bucket_probabilities",
    "compute_confidence",
    "EnsembleBlender",
    "EnsembleForecast",
    "ForecastPoint",
    "MetarFetcher",
    "MetarObservation",
    "ForecastingService",
]
