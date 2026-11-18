"""Tests for FeatureView, Entity, and Schema Validation."""

from src.core.feature_view import Entity, Feature, FeatureType, FeatureView


def test_feature_view_definition_and_validation() -> None:
    view = FeatureView(
        name="customer_risk",
        entities=[Entity(name="customer", join_key="customer_id")],
        features=[
            Feature(name="score", dtype=FeatureType.FLOAT64),
            Feature(name="rank", dtype=FeatureType.INT64),
            Feature(name="tier", dtype=FeatureType.STRING),
            Feature(name="active", dtype=FeatureType.BOOLEAN),
        ],
        ttl_seconds=3600,
    )

    assert view.name == "customer_risk"
    assert view.get_feature("score") is not None
    assert view.get_feature("non_existent") is None

    # Valid types
    assert view.validate_value("score", 750.5)
    assert view.validate_value("score", 700)  # int accepted for float
    assert view.validate_value("rank", 3)
    assert view.validate_value("tier", "gold")
    assert view.validate_value("active", True)
    assert view.validate_value("score", None)

    # Invalid types
    assert not view.validate_value("score", "invalid_str")
    assert not view.validate_value("score", True)
    assert not view.validate_value("rank", 3.14)
    assert not view.validate_value("tier", 123)
    assert not view.validate_value("active", "yes")
    assert not view.validate_value("unknown_field", 100)
