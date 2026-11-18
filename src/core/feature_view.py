"""Feature View and Schema Definitions for Feature Store."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class FeatureType(StrEnum):
    """Supported data types for feature store values."""

    INT64 = "int64"
    FLOAT64 = "float64"
    STRING = "string"
    BOOLEAN = "boolean"


class Entity(BaseModel):
    """Defines an entity identity key in the feature store."""

    name: str = Field(..., description="Logical name of the entity, e.g. user")
    join_key: str = Field(..., description="Primary key field used in joins, e.g. user_id")
    description: str = Field(default="", description="Business context")


class Feature(BaseModel):
    """Defines an individual feature specification."""

    name: str = Field(..., description="Feature identifier")
    dtype: FeatureType = Field(..., description="Data type of the feature")
    description: str = Field(default="", description="Feature documentation")


class FeatureView(BaseModel):
    """Logical grouping of features sharing an entity and ingestion cadence."""

    name: str = Field(..., description="Feature view identifier")
    entities: list[Entity] = Field(..., description="Associated entity join keys")
    features: list[Feature] = Field(..., description="Features contained in this view")
    ttl_seconds: int = Field(default=86400 * 30, description="Time-to-live before staleness")

    def get_feature(self, name: str) -> Feature | None:
        """Find a feature by name in this view."""
        for f in self.features:
            if f.name == name:
                return f
        return None

    def validate_value(self, feature_name: str, value: Any) -> bool:
        """Validate if value conforms to feature type definition."""
        feature = self.get_feature(feature_name)
        if not feature:
            return False
        if value is None:
            return True

        if feature.dtype == FeatureType.INT64:
            return isinstance(value, int) and not isinstance(value, bool)
        elif feature.dtype == FeatureType.FLOAT64:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif feature.dtype == FeatureType.STRING:
            return isinstance(value, str)
        elif feature.dtype == FeatureType.BOOLEAN:
            return isinstance(value, bool)
        return False


class FeatureRecord(BaseModel):
    """Represents a feature value mutation or record."""

    entity_id: str = Field(..., description="Entity identifier instance, e.g. user_101")
    feature_name: str = Field(..., description="Feature name")
    value: Any = Field(..., description="Feature payload value")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when feature was produced",
    )
