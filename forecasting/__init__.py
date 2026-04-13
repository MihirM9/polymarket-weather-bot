"""
forecasting package
===================
Groups the live forecast ingestion and blending surface behind a single import
path so the app can reason about "forecasting" as one subsystem.
"""

from .blender import EnsembleBlender, EnsembleForecast, ForecastPoint
from .metar import MetarFetcher, MetarObservation
from .scanner import CityForecast, ForecastScanner, bucket_probabilities, compute_confidence
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
