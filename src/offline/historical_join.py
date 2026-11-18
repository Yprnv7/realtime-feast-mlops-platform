"""Offline Point-in-Time Correct Historical Join Engine using DuckDB."""

from datetime import datetime
from typing import Any

import duckdb
from pydantic import BaseModel, Field


class ObservationRecord(BaseModel):
    """An observation event needing training features as of observation_timestamp."""

    observation_id: str = Field(..., description="Unique event identifier")
    entity_id: str = Field(..., description="Target entity join key")
    observation_timestamp: datetime = Field(..., description="Point-in-time cutoff timestamp")
    label: float | int | None = Field(default=None, description="Target label for training")


class HistoricalFeatureEvent(BaseModel):
    """Historical feature event emitted at a specific timestamp."""

    entity_id: str
    feature_name: str
    feature_value: float | int | str
    feature_timestamp: datetime


class HistoricalJoinEngine:
    """Computes mathematically point-in-time correct historical feature joins."""

    def __init__(self, in_memory: bool = True) -> None:
        self.conn = duckdb.connect(":memory:" if in_memory else "data/feature_store.duckdb")
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize staging tables for feature mutations."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_events (
                entity_id VARCHAR,
                feature_name VARCHAR,
                feature_value DOUBLE,
                feature_timestamp TIMESTAMP
            );
        """)

    def insert_feature_events(self, events: list[HistoricalFeatureEvent]) -> int:
        """Insert historical feature events into offline store."""
        if not events:
            return 0

        rows = [
            (e.entity_id, e.feature_name, float(e.feature_value), e.feature_timestamp)
            for e in events
        ]
        self.conn.executemany(
            "INSERT INTO feature_events VALUES (?, ?, ?, ?)",
            rows,
        )
        return len(rows)

    def execute_point_in_time_join(
        self,
        observations: list[ObservationRecord],
        feature_names: list[str],
        ttl_seconds: int = 86400 * 30,
    ) -> list[dict[str, Any]]:
        """
        Execute point-in-time correct historical join.

        Guarantees zero data leakage: for each observation, only feature values
        written BEFORE or AT observation_timestamp are joined, taking the most
        recent state prior to the cutoff.
        """
        if not observations:
            return []

        # Create temporary table for observations
        self.conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS obs_temp (
                observation_id VARCHAR,
                entity_id VARCHAR,
                observation_timestamp TIMESTAMP,
                label DOUBLE
            );
        """)
        self.conn.execute("DELETE FROM obs_temp;")

        obs_rows = [
            (
                o.observation_id,
                o.entity_id,
                o.observation_timestamp,
                float(o.label) if o.label is not None else None,
            )
            for o in observations
        ]
        self.conn.executemany("INSERT INTO obs_temp VALUES (?, ?, ?, ?)", obs_rows)

        # Windowed ASOF snapshot join
        # For each feature name, join latest value where f_ts <= obs_ts and f_ts >= obs_ts - ttl
        results_by_obs: dict[str, dict[str, Any]] = {
            o.observation_id: {
                "observation_id": o.observation_id,
                "entity_id": o.entity_id,
                "observation_timestamp": o.observation_timestamp.isoformat(),
                "label": o.label,
                **{feat: None for feat in feature_names},
            }
            for o in observations
        }

        query = f"""
            WITH ranked_features AS (
                SELECT
                    o.observation_id,
                    f.feature_name,
                    f.feature_value,
                    f.feature_timestamp,
                    ROW_NUMBER() OVER (
                        PARTITION BY o.observation_id, f.feature_name
                        ORDER BY f.feature_timestamp DESC, f.feature_value DESC
                    ) as rank
                FROM obs_temp o
                JOIN feature_events f
                  ON o.entity_id = f.entity_id
                 AND f.feature_timestamp <= o.observation_timestamp
                 AND f.feature_timestamp >= o.observation_timestamp - INTERVAL '{ttl_seconds} SECONDS'
                WHERE f.feature_name IN ({", ".join(f"'{name}'" for name in feature_names)})
            )
            SELECT observation_id, feature_name, feature_value
            FROM ranked_features
            WHERE rank = 1;
        """

        joined_rows = self.conn.execute(query).fetchall()
        for obs_id, feat_name, feat_val in joined_rows:
            if obs_id in results_by_obs:
                results_by_obs[obs_id][feat_name] = feat_val

        return list(results_by_obs.values())
