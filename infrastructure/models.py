"""
api_models.py — Strict validation models for external API responses
===================================================================
Uses pydantic to reduce silent failures when upstream API payloads drift.
"""

from __future__ import annotations

import json
from typing import Any, Optional, Self, TypeVar

from pydantic import BaseModel, Field, ValidationError, validator

ModelT = TypeVar("ModelT", bound="APIModel")


class APIModel(BaseModel):
    class Config:
        extra = "ignore"

    @classmethod
    def model_validate(cls, data: Any) -> Self:
        return cls.parse_obj(data)

    def model_dump(self) -> dict[str, Any]:
        return self.dict()


class NWSPointProperties(APIModel):
    gridId: str
    gridX: int
    gridY: int


class NWSPointsResponse(APIModel):
    properties: NWSPointProperties


class NWSForecastPeriod(APIModel):
    isDaytime: bool = False
    temperature: Optional[float] = None
    temperatureUnit: str = "F"
    startTime: str
    detailedForecast: str = ""
    shortForecast: str = ""


class NWSForecastProperties(APIModel):
    periods: list[NWSForecastPeriod] = Field(default_factory=list)


class NWSForecastResponse(APIModel):
    properties: NWSForecastProperties


class NWSTemperatureValue(APIModel):
    value: Optional[float] = None


class NWSObservationProperties(APIModel):
    temperature: NWSTemperatureValue


class NWSLatestObservationResponse(APIModel):
    properties: NWSObservationProperties


class GammaMarketWire(APIModel):
    id: Optional[Any] = None
    conditionId: Optional[Any] = None
    question: str = ""
    groupItemTitle: str = ""
    resolutionSource: str = ""
    active: bool = True
    closed: bool = False
    volume: float = 0.0
    liquidity: float = 0.0
    negRiskMarketID: Optional[str] = None
    outcomePrices: list[float] = Field(default_factory=list)
    clobTokenIds: list[str] = Field(default_factory=list)

    @validator("volume", "liquidity", pre=True)
    @classmethod
    def _coerce_float(cls, value: Any) -> float:
        if value in (None, ""):
            return 0.0
        return float(value)

    @validator("outcomePrices", pre=True)
    @classmethod
    def _parse_outcome_prices(cls, value: Any) -> list[float]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid outcomePrices JSON") from exc
        if not isinstance(value, list):
            raise ValueError("outcomePrices must be a list")
        return [float(item) for item in value]

    @validator("clobTokenIds", pre=True)
    @classmethod
    def _parse_clob_ids(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("invalid clobTokenIds JSON") from exc
        if not isinstance(value, list):
            raise ValueError("clobTokenIds must be a list")
        return [str(item) for item in value]


class ClobOrderStatusResponse(APIModel):
    status: str = ""
    size_matched: Optional[float] = 0.0
    original_size: Optional[float] = 0.0
    associate_trades_avg_price: Optional[float] = None
    price: Optional[float] = None

    @validator("size_matched", "original_size", "associate_trades_avg_price", "price", pre=True)
    @classmethod
    def _parse_optional_float(cls, value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        return float(value)


def validate_model(data: Any, model: type[ModelT], *, label: str) -> Optional[ModelT]:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        import logging
        logging.getLogger(__name__).warning(f"[{label}] Response validation failed: {exc.errors()[0]['msg']}")
        return None
