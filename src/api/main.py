"""Demand forecast API.

Run from the project root:   uvicorn src.api.main:app --reload
Interactive docs:            http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from src.api.schemas import (BatchRequest, BatchResponse, InfoResponse, PredictionRequest, PredictionResponse)
from src.serving import ForecastService

API_VERSION = "0.1.0"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = ForecastService.load()        # models + history are loaded once at startup
    yield


app = FastAPI(
    title="Demand Forecast API",
    version=API_VERSION,
    description="7-day-ahead demand forecasts for store-product series, served from XGBoost models.",
    lifespan=lifespan,
)


def get_service(request: Request) -> ForecastService:
    return request.app.state.service


def _call(fn, *args):
    """Translate service errors into proper HTTP status codes."""
    try:
        return fn(*args)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e.args[0])) from None
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from None


@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse, tags=["meta"])
def info(service: ForecastService = Depends(get_service)):
    return {**service.info(), "api_version": API_VERSION}


@app.get("/series", tags=["meta"])
def list_series(service: ForecastService = Depends(get_service)):
    """All store-product series the API can forecast."""
    return service.series_catalog()


@app.post("/predict", response_model=PredictionResponse, tags=["forecast"])
def predict(body: PredictionRequest, service: ForecastService = Depends(get_service)):
    return _call(lambda: service.predict(**body.model_dump()))


@app.post("/predict/batch", response_model=BatchResponse, tags=["forecast"])
def predict_batch(body: BatchRequest, service: ForecastService = Depends(get_service)):
    return {"predictions": _call(service.predict_many, [r.model_dump() for r in body.requests])}
