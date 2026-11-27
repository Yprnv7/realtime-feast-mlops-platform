"""FastAPI HTTP Service for Real-Time Feature Store & MLOps Platform."""

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.core.feature_view import Entity, Feature, FeatureType, FeatureView
from src.monitoring.drift import DriftReport, FeatureDriftMonitor
from src.offline.historical_join import HistoricalJoinEngine, ObservationRecord
from src.online.store import OnlineFeatureStore
from src.streaming.ingestion import StreamingIngestionEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Real-Time Feature Store & MLOps Platform",
    description="High-throughput online feature serving, point-in-time correct historical joins, and statistical drift monitoring.",
    version="0.1.0",
)

# Initialize Core Singletons
online_store = OnlineFeatureStore()
offline_engine = HistoricalJoinEngine(in_memory=True)
drift_monitor = FeatureDriftMonitor()
streaming_engine = StreamingIngestionEngine(
    online_store=online_store,
    offline_engine=offline_engine,
)

# Register default user_credit_features view
default_view = FeatureView(
    name="user_credit_features",
    entities=[Entity(name="user", join_key="user_id")],
    features=[
        Feature(name="credit_score", dtype=FeatureType.FLOAT64),
        Feature(name="monthly_income", dtype=FeatureType.FLOAT64),
        Feature(name="debt_to_income_ratio", dtype=FeatureType.FLOAT64),
        Feature(name="active_loans_count", dtype=FeatureType.INT64),
    ],
    ttl_seconds=86400 * 30,
)
online_store.register_view(default_view)


class OnlineFeaturesRequest(BaseModel):
    """Payload to retrieve online features for entity IDs."""

    entity_ids: list[str] = Field(..., min_length=1)
    features: list[str] = Field(..., min_length=1)


class IngestEventRequest(BaseModel):
    """Payload to ingest a real-time feature event."""

    entity_id: str
    feature_name: str
    value: float | int | str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class HistoricalJoinRequest(BaseModel):
    """Payload to perform a point-in-time historical join."""

    observations: list[ObservationRecord]
    features: list[str]
    ttl_seconds: int = 86400 * 30


class DriftEvaluationRequest(BaseModel):
    """Payload to evaluate distribution drift on a feature."""

    feature_name: str
    baseline_values: list[float]
    production_values: list[float]


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check and live feature store telemetry."""
    return {
        "status": "healthy",
        "service": "realtime-feast-mlops-platform",
        "online_entity_count": online_store.get_entity_count(),
        "total_ingested_events": streaming_engine.total_ingested,
    }


@app.get("/v1/features/views")
def list_feature_views() -> list[dict[str, Any]]:
    """List registered feature views and schemas."""
    return [default_view.model_dump()]


@app.post("/v1/features/online")
def get_online_features(req: OnlineFeaturesRequest) -> list[dict[str, Any]]:
    """Retrieve sub-millisecond online feature vectors for inference."""
    try:
        return online_store.get_online_features(
            entity_ids=req.entity_ids,
            feature_names=req.features,
        )
    except Exception as e:
        logger.error("Error retrieving online features: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving online features",
        ) from e


@app.post("/v1/features/ingest", status_code=status.HTTP_201_CREATED)
def ingest_feature(req: IngestEventRequest) -> dict[str, Any]:
    """Ingest a feature event with synchronous online write and offline log."""
    streaming_engine.ingest_event(
        entity_id=req.entity_id,
        feature_name=req.feature_name,
        value=req.value,
        timestamp=req.timestamp,
    )
    return {"status": "success", "entity_id": req.entity_id, "feature": req.feature_name}


@app.post("/v1/features/historical-join")
def historical_join(req: HistoricalJoinRequest) -> list[dict[str, Any]]:
    """Execute point-in-time correct join for offline model training."""
    try:
        return offline_engine.execute_point_in_time_join(
            observations=req.observations,
            feature_names=req.features,
            ttl_seconds=req.ttl_seconds,
        )
    except Exception as e:
        logger.error("Historical join execution failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Historical join execution failed",
        ) from e


@app.post("/v1/features/drift", response_model=DriftReport)
def evaluate_drift(req: DriftEvaluationRequest) -> DriftReport:
    """Evaluate distribution drift via KS-test and Population Stability Index."""
    return drift_monitor.evaluate_feature_drift(
        feature_name=req.feature_name,
        baseline_values=req.baseline_values,
        production_values=req.production_values,
    )
