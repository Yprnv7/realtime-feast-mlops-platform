"""Tests for Feature Drift and Data Quality Monitoring."""

import numpy as np

from src.monitoring.drift import FeatureDriftMonitor


def test_drift_monitoring_no_drift() -> None:
    monitor = FeatureDriftMonitor()
    np.random.seed(42)

    # Both drawn from N(0, 1)
    baseline = np.random.normal(loc=0.0, scale=1.0, size=1000).tolist()
    production = np.random.normal(loc=0.0, scale=1.0, size=1000).tolist()

    report = monitor.evaluate_feature_drift("embedding_norm", baseline, production)
    assert report.drift_detected is False
    assert report.drift_severity == "NONE"
    assert report.psi < 0.10
    assert report.baseline_missing_rate == 0.0


def test_drift_monitoring_high_drift() -> None:
    monitor = FeatureDriftMonitor()
    np.random.seed(42)

    # Baseline N(0, 1) vs Production N(3, 1) - significant distribution shift
    baseline = np.random.normal(loc=0.0, scale=1.0, size=1000).tolist()
    production = np.random.normal(loc=3.0, scale=1.0, size=1000).tolist()

    report = monitor.evaluate_feature_drift("user_activity_score", baseline, production)
    assert report.drift_detected is True
    assert report.drift_severity == "HIGH"
    assert report.psi >= 0.20
    assert report.ks_statistic > 0.50
    assert report.ks_p_value < 0.001


def test_drift_monitoring_moderate_drift() -> None:
    monitor = FeatureDriftMonitor()
    np.random.seed(42)

    # Mild shift N(0, 1) vs N(0.35, 1)
    baseline = np.random.normal(loc=0.0, scale=1.0, size=1000).tolist()
    production = np.random.normal(loc=0.35, scale=1.0, size=1000).tolist()

    report = monitor.evaluate_feature_drift("session_duration", baseline, production)
    assert report.drift_detected is True
    assert report.drift_severity in ("MODERATE", "HIGH")


def test_drift_monitoring_empty_and_missing() -> None:
    monitor = FeatureDriftMonitor()
    report = monitor.evaluate_feature_drift("empty_feat", [], [])
    assert report.drift_detected is False
    assert report.drift_severity == "NONE"
    assert report.psi == 0.0

    # Test missing values
    base = [1.0, None, 2.0, 3.0]  # type: ignore[list-item]
    prod = [1.0, 2.0, 3.0, 4.0]
    report_miss = monitor.evaluate_feature_drift("metric", base, prod)
    assert report_miss.baseline_missing_rate == 0.25
    assert report_miss.production_missing_rate == 0.0
