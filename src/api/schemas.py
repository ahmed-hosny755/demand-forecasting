"""Request / response models (validated automatically by FastAPI + Pydantic)."""
from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    store_id: str = Field(..., description="Store identifier, e.g. S001")
    product_id: str = Field(..., description="Product identifier, e.g. P0001")
    target_date: date = Field(..., description="Day to forecast. Must be within 7 days after the last day of history "
                                              "(or earlier, to replay a past day).")
    price: float = Field(..., gt=0, description="Planned price on the target day")
    discount: float = Field(0, ge=0, le=100, description="Planned discount (%) on the target day")
    promotion: int = Field(0, ge=0, le=1, description="1 if the target day is a planned promotion day")
    epidemic: Optional[int] = Field(None, ge=0, le=1,
                                    description="Optional scenario input: 1 if an epidemic regime is expected on the "
                                                "target day, 0 if not. If omitted, the model that does not need it is used.")

    model_config = ConfigDict(json_schema_extra={"example": {
        "store_id": "S001", "product_id": "P0001", "target_date": "2024-02-05",
        "price": 72.7, "discount": 10, "promotion": 1}})


class PredictionResponse(BaseModel):
    store_id: str
    product_id: str
    category: str
    target_date: date
    forecast_origin: date = Field(..., description="Last day whose data the forecast may use (target - horizon)")
    horizon_days: int
    forecast_units: float = Field(..., description="Forecast demand in units")
    model_used: str = Field(..., description="epidemic_unknown or epidemic_known")


class BatchRequest(BaseModel):
    requests: list[PredictionRequest] = Field(..., min_length=1, max_length=200)


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse]


class InfoResponse(BaseModel):
    horizon_days: int
    models_loaded: list[str]
    trained_until: str
    history_start: str
    history_end: str
    n_series: int
    api_version: str
