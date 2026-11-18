"""Tests for FastAPI Feature Store HTTP Endpoints."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_api_health_and_views() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "online_entity_count" in data

    res_views = client.get("/v1/features/views")
    assert res_views.status_code == 200
    views = res_views.json()
    assert len(views) >= 1
    assert views[0]["name"] == "user_credit_features"


def test_api_ingest_and_online_query() -> None:
    # 1. Ingest event
    payload_ingest = {
        "entity_id": "u_999",
        "feature_name": "credit_score",
        "value": 780.0,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    res_ingest = client.post("/v1/features/ingest", json=payload_ingest)
    assert res_ingest.status_code == 201

    # 2. Query online features
    payload_query = {
        "entity_ids": ["u_999"],
        "features": ["credit_score", "monthly_income"],
    }
    res_query = client.post("/v1/features/online", json=payload_query)
    assert res_query.status_code == 200
    rows = res_query.json()
    assert len(rows) == 1
    assert rows[0]["entity_id"] == "u_999"
    assert rows[0]["credit_score"] == 780.0
    assert rows[0]["monthly_income"] is None


def test_api_historical_join() -> None:
    t0 = datetime(2026, 7, 1, 10, 0, 0, tzinfo=UTC).isoformat()
    t_obs = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC).isoformat()

    # Ingest historical event
    client.post(
        "/v1/features/ingest",
        json={
            "entity_id": "u_hist",
            "feature_name": "monthly_income",
            "value": 6500.0,
            "timestamp": t0,
        },
    )

    payload = {
        "observations": [
            {
                "observation_id": "obs_hist_1",
                "entity_id": "u_hist",
                "observation_timestamp": t_obs,
                "label": 1,
            }
        ],
        "features": ["monthly_income"],
    }
    res = client.post("/v1/features/historical-join", json=payload)
    assert res.status_code == 200
    rows = res.json()
    assert len(rows) == 1
    assert rows[0]["monthly_income"] == 6500.0


def test_api_drift_evaluation() -> None:
    payload = {
        "feature_name": "debt_ratio",
        "baseline_values": [0.1, 0.2, 0.15, 0.25, 0.3],
        "production_values": [0.12, 0.18, 0.16, 0.22, 0.28],
    }
    res = client.post("/v1/features/drift", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["feature_name"] == "debt_ratio"
    assert "ks_statistic" in data
    assert "psi" in data
