"""Streaming Ingestion Engine with Dual-Write to Online & Offline Stores."""

import queue
import threading
from datetime import UTC, datetime
from typing import Any

from src.core.feature_view import FeatureRecord, FeatureView
from src.offline.historical_join import HistoricalFeatureEvent, HistoricalJoinEngine
from src.online.store import OnlineFeatureStore


class StreamingIngestionEngine:
    """Manages dual-write ingestion across low-latency online and analytical offline stores."""

    def __init__(
        self,
        online_store: OnlineFeatureStore,
        offline_engine: HistoricalJoinEngine,
        feature_views: list[FeatureView] | None = None,
    ) -> None:
        self.online_store = online_store
        self.offline_engine = offline_engine
        self.views = {v.name: v for v in (feature_views or [])}
        self._queue: queue.Queue[FeatureRecord] = queue.Queue(maxsize=100_000)
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._processed_count = 0

    def ingest_event(
        self,
        entity_id: str,
        feature_name: str,
        value: Any,
        timestamp: datetime | None = None,
    ) -> bool:
        """Synchronously updates online store and queues for offline persistence."""
        event_time = timestamp or datetime.now(UTC)
        record = FeatureRecord(
            entity_id=entity_id,
            feature_name=feature_name,
            value=value,
            timestamp=event_time,
        )

        # 1. Immediate Online Write for zero-lag model serving
        self.online_store.write_features([record])

        # 2. Offline Store Write for point-in-time training joins
        if isinstance(value, (int, float)):
            hist_event = HistoricalFeatureEvent(
                entity_id=entity_id,
                feature_name=feature_name,
                feature_value=value,
                feature_timestamp=event_time,
            )
            self.offline_engine.insert_feature_events([hist_event])

        self._processed_count += 1
        return True

    def ingest_batch(self, records: list[FeatureRecord]) -> int:
        """Batch ingestion across both online and offline stores."""
        if not records:
            return 0

        # Online write
        self.online_store.write_features(records)

        # Offline write
        hist_events = [
            HistoricalFeatureEvent(
                entity_id=r.entity_id,
                feature_name=r.feature_name,
                feature_value=r.value,
                feature_timestamp=r.timestamp,
            )
            for r in records
            if isinstance(r.value, (int, float))
        ]
        if hist_events:
            self.offline_engine.insert_feature_events(hist_events)

        self._processed_count += len(records)
        return len(records)

    @property
    def total_ingested(self) -> int:
        """Return cumulative event count ingested."""
        return self._processed_count
