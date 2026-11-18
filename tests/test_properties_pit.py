"""Property-based invariant testing for Point-in-Time correctness."""

from datetime import UTC, datetime, timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from src.offline.historical_join import (
    HistoricalFeatureEvent,
    HistoricalJoinEngine,
    ObservationRecord,
)


@settings(max_examples=30, deadline=None)
@given(
    time_offsets=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=1, max_size=15),
    obs_offset=st.integers(min_value=-500, max_value=500),
)
def test_hypothesis_point_in_time_invariant(time_offsets: list[int], obs_offset: int) -> None:
    """Invariant: The joined feature timestamp must NEVER exceed observation_timestamp."""
    engine = HistoricalJoinEngine(in_memory=True)
    base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    events: list[HistoricalFeatureEvent] = []
    expected_valid_events: list[tuple[datetime, float]] = []
    obs_time = base_time + timedelta(seconds=obs_offset)

    for i, offset in enumerate(time_offsets):
        f_time = base_time + timedelta(seconds=offset)
        f_val = float(i * 10)
        events.append(
            HistoricalFeatureEvent(
                entity_id="entity_prop",
                feature_name="metric",
                feature_value=f_val,
                feature_timestamp=f_time,
            )
        )
        if f_time <= obs_time:
            expected_valid_events.append((f_time, f_val))

    engine.insert_feature_events(events)

    obs = [
        ObservationRecord(
            observation_id="obs_prop",
            entity_id="entity_prop",
            observation_timestamp=obs_time,
        )
    ]

    res = engine.execute_point_in_time_join(obs, ["metric"], ttl_seconds=86400 * 365)
    joined_val = res[0]["metric"]

    if not expected_valid_events:
        # If no events occurred before or at obs_time, joined value must be None
        assert joined_val is None
    else:
        # The expected value must be the one with the latest timestamp <= obs_time
        expected_valid_events.sort(key=lambda x: (x[0], x[1]))
        expected_latest_val = expected_valid_events[-1][1]
        assert joined_val == expected_latest_val
