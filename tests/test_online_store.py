"""Tests for Online Feature Store."""

import threading
from datetime import UTC, datetime, timedelta

from src.core.feature_view import Entity, Feature, FeatureRecord, FeatureType, FeatureView
from src.online.store import OnlineFeatureStore


def test_online_store_write_and_read() -> None:
    view = FeatureView(
        name="user_view",
        entities=[Entity(name="user", join_key="user_id")],
        features=[
            Feature(name="clicks_7d", dtype=FeatureType.INT64),
            Feature(name="avg_spend", dtype=FeatureType.FLOAT64),
        ],
        ttl_seconds=3600,
    )
    store = OnlineFeatureStore([view])

    now = datetime.now(UTC)
    records = [
        FeatureRecord(entity_id="u_1", feature_name="clicks_7d", value=42, timestamp=now),
        FeatureRecord(entity_id="u_1", feature_name="avg_spend", value=128.50, timestamp=now),
        FeatureRecord(entity_id="u_2", feature_name="clicks_7d", value=10, timestamp=now),
    ]

    written = store.write_features(records)
    assert written == 3
    assert store.get_entity_count() == 2

    # Query online features
    res = store.get_online_features(["u_1", "u_2", "u_3"], ["clicks_7d", "avg_spend"])
    assert len(res) == 3

    assert res[0]["entity_id"] == "u_1"
    assert res[0]["clicks_7d"] == 42
    assert res[0]["avg_spend"] == 128.50

    assert res[1]["entity_id"] == "u_2"
    assert res[1]["clicks_7d"] == 10
    assert res[1]["avg_spend"] is None

    assert res[2]["entity_id"] == "u_3"
    assert res[2]["clicks_7d"] is None


def test_online_store_ttl_staleness_eviction() -> None:
    view = FeatureView(
        name="fast_view",
        entities=[Entity(name="user", join_key="user_id")],
        features=[Feature(name="temp_metric", dtype=FeatureType.FLOAT64)],
        ttl_seconds=60,  # 1 minute TTL
    )
    store = OnlineFeatureStore([view])

    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
    store.write_features(
        [FeatureRecord(entity_id="u_10", feature_name="temp_metric", value=99.0, timestamp=t0)]
    )

    # Query within TTL (30s later)
    within_ttl = t0 + timedelta(seconds=30)
    res_valid = store.get_online_features(["u_10"], ["temp_metric"], as_of_time=within_ttl)
    assert res_valid[0]["temp_metric"] == 99.0

    # Query beyond TTL (120s later)
    after_ttl = t0 + timedelta(seconds=120)
    res_stale = store.get_online_features(["u_10"], ["temp_metric"], as_of_time=after_ttl)
    assert res_stale[0]["temp_metric"] is None
    assert res_stale[0]["temp_metric__is_stale"] is True


def test_online_store_concurrency() -> None:
    store = OnlineFeatureStore()
    now = datetime.now(UTC)

    def writer(idx: int) -> None:
        for i in range(50):
            store.write_features(
                [
                    FeatureRecord(
                        entity_id=f"ent_{idx}",
                        feature_name="metric",
                        value=i,
                        timestamp=now,
                    )
                ]
            )

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert store.get_entity_count() == 10
    features = store.get_online_features(["ent_0", "ent_1"], ["metric"])
    assert features[0]["metric"] == 49
    store.clear()
    assert store.get_entity_count() == 0
