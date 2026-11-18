"""Tests for Point-in-Time Correct Historical Join Engine."""

from datetime import UTC, datetime, timedelta

from src.offline.historical_join import (
    HistoricalFeatureEvent,
    HistoricalJoinEngine,
    ObservationRecord,
)


def test_point_in_time_correctness_zero_leakage() -> None:
    engine = HistoricalJoinEngine(in_memory=True)

    # Entity user_100 mutations over time:
    # t0 (10:00): balance = 100
    # t1 (11:00): balance = 200
    # t2 (12:00): balance = 300
    # t3 (13:00): balance = 400
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    t1 = datetime(2026, 5, 1, 11, 0, 0, tzinfo=UTC)
    t2 = datetime(2026, 5, 1, 12, 0, 0, tzinfo=UTC)
    t3 = datetime(2026, 5, 1, 13, 0, 0, tzinfo=UTC)

    events = [
        HistoricalFeatureEvent(
            entity_id="u_100", feature_name="balance", feature_value=100.0, feature_timestamp=t0
        ),
        HistoricalFeatureEvent(
            entity_id="u_100", feature_name="balance", feature_value=200.0, feature_timestamp=t1
        ),
        HistoricalFeatureEvent(
            entity_id="u_100", feature_name="balance", feature_value=300.0, feature_timestamp=t2
        ),
        HistoricalFeatureEvent(
            entity_id="u_100", feature_name="balance", feature_value=400.0, feature_timestamp=t3
        ),
    ]
    engine.insert_feature_events(events)

    # Observation at 11:30 AM (between t1 and t2)
    # MUST return balance = 200.0 (t1), NEVER 300.0 or 400.0 (future leakage)
    obs_time = datetime(2026, 5, 1, 11, 30, 0, tzinfo=UTC)
    obs = [
        ObservationRecord(
            observation_id="obs_1",
            entity_id="u_100",
            observation_timestamp=obs_time,
            label=1.0,
        )
    ]

    res = engine.execute_point_in_time_join(obs, ["balance"])
    assert len(res) == 1
    assert res[0]["observation_id"] == "obs_1"
    assert res[0]["balance"] == 200.0, "Point-in-time join leaked future feature state!"


def test_point_in_time_ttl_eviction() -> None:
    engine = HistoricalJoinEngine(in_memory=True)
    t0 = datetime(2026, 5, 1, 10, 0, 0, tzinfo=UTC)
    events = [
        HistoricalFeatureEvent(
            entity_id="u_200", feature_name="credit", feature_value=750.0, feature_timestamp=t0
        )
    ]
    engine.insert_feature_events(events)

    # Observation 5 hours later with TTL of 1 hour (3600s) -> should be evicted (None)
    obs_time = t0 + timedelta(hours=5)
    obs = [
        ObservationRecord(
            observation_id="obs_stale",
            entity_id="u_200",
            observation_timestamp=obs_time,
        )
    ]

    res = engine.execute_point_in_time_join(obs, ["credit"], ttl_seconds=3600)
    assert res[0]["credit"] is None


def test_empty_observations() -> None:
    engine = HistoricalJoinEngine(in_memory=True)
    assert engine.execute_point_in_time_join([], ["feat"]) == []
    assert engine.insert_feature_events([]) == 0
