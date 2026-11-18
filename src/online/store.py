"""Ultra-low latency Online Feature Store for real-time model inference."""

import threading
from datetime import UTC, datetime
from typing import Any

from src.core.feature_view import FeatureRecord, FeatureView


class OnlineFeatureStore:
    """Thread-safe, microsecond-tier in-memory online feature store."""

    def __init__(self, feature_views: list[FeatureView] | None = None) -> None:
        self._lock = threading.RLock()
        # Storage schema: {entity_id: {feature_name: (value, timestamp)}}
        self._storage: dict[str, dict[str, tuple[Any, datetime]]] = {}
        self._views: dict[str, FeatureView] = {v.name: v for v in (feature_views or [])}
        self._feature_to_ttl: dict[str, int] = {}
        for view in self._views.values():
            for feat in view.features:
                self._feature_to_ttl[feat.name] = view.ttl_seconds

    def register_view(self, view: FeatureView) -> None:
        """Register a feature view with associated TTL rules."""
        with self._lock:
            self._views[view.name] = view
            for feat in view.features:
                self._feature_to_ttl[feat.name] = view.ttl_seconds

    def write_features(self, records: list[FeatureRecord]) -> int:
        """Atomic batch write of streaming feature records into online store."""
        written = 0
        with self._lock:
            for rec in records:
                if rec.entity_id not in self._storage:
                    self._storage[rec.entity_id] = {}
                self._storage[rec.entity_id][rec.feature_name] = (rec.value, rec.timestamp)
                written += 1
        return written

    def get_online_features(
        self,
        entity_ids: list[str],
        feature_names: list[str],
        as_of_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve latest feature vector for specified entities with staleness check."""
        query_time = as_of_time or datetime.now(UTC)
        results: list[dict[str, Any]] = []

        with self._lock:
            for entity_id in entity_ids:
                row: dict[str, Any] = {"entity_id": entity_id}
                entity_features = self._storage.get(entity_id, {})

                for feat in feature_names:
                    if feat in entity_features:
                        val, ts = entity_features[feat]
                        # Check TTL staleness
                        ttl = self._feature_to_ttl.get(feat, 86400 * 30)
                        age_seconds = (query_time - ts).total_seconds()
                        if age_seconds > ttl:
                            row[feat] = None  # Stale feature evicted
                            row[f"{feat}__is_stale"] = True
                        else:
                            row[feat] = val
                    else:
                        row[feat] = None
                results.append(row)

        return results

    def get_entity_count(self) -> int:
        """Return total unique entities indexed."""
        with self._lock:
            return len(self._storage)

    def clear(self) -> None:
        """Flush store state."""
        with self._lock:
            self._storage.clear()
