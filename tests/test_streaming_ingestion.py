"""Tests for Streaming Ingestion Engine."""

from datetime import UTC, datetime, timedelta

from src.core.feature_view import FeatureRecord
from src.offline.historical_join import HistoricalJoinEngine, ObservationRecord
from src.online.store import OnlineFeatureStore
from src.streaming.ingestion import StreamingIngestionEngine


def test_streaming_ingestion_dual_write() -> None:
    online_store = OnlineFeatureStore()
    offline_engine = HistoricalJoinEngine(in_memory=True)
    engine = StreamingIngestionEngine(online_store, offline_engine)

    t0 = datetime.now(UTC)

    # Ingest single event
    ok = engine.ingest_event("u_500", "daily_spend", 45.5, timestamp=t0)
    assert ok is True
    assert engine.total_ingested == 1

    # Verify online store has it immediately
    online_res = online_store.get_online_features(["u_500"], ["daily_spend"])
    assert online_res[0]["daily_spend"] == 45.5

    # Ingest batch
    records = [
        FeatureRecord(entity_id="u_501", feature_name="daily_spend", value=12.0, timestamp=t0),
        FeatureRecord(entity_id="u_502", feature_name="daily_spend", value=99.0, timestamp=t0),
    ]
    batch_count = engine.ingest_batch(records)
    assert batch_count == 2
    assert engine.total_ingested == 3

    # Verify offline historical query joins correctly
    obs = [
        ObservationRecord(
            observation_id="obs_501",
            entity_id="u_501",
            observation_timestamp=t0 + timedelta(minutes=1),
        )
    ]
    offline_res = offline_engine.execute_point_in_time_join(obs, ["daily_spend"])
    assert offline_res[0]["daily_spend"] == 12.0
    assert engine.ingest_batch([]) == 0
