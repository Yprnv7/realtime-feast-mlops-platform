"""Statistical Feature Drift & Data Quality Monitoring Engine."""

import numpy as np
from pydantic import BaseModel, Field
from scipy import stats


class DriftReport(BaseModel):
    """Encapsulates statistical drift metrics for a feature."""

    feature_name: str
    ks_statistic: float = Field(..., description="Two-sample Kolmogorov-Smirnov test statistic")
    ks_p_value: float = Field(..., description="KS test p-value")
    psi: float = Field(..., description="Population Stability Index")
    drift_detected: bool = Field(..., description="True if PSI >= 0.2 or p-value < 0.01")
    drift_severity: str = Field(..., description="NONE, MODERATE, or HIGH")
    baseline_missing_rate: float
    production_missing_rate: float
    sample_size_baseline: int
    sample_size_production: int


class FeatureDriftMonitor:
    """Computes distribution drift between baseline (training) and production features."""

    @staticmethod
    def calculate_psi(
        baseline: np.ndarray,
        production: np.ndarray,
        num_buckets: int = 10,
        eps: float = 1e-4,
    ) -> float:
        """
        Calculate Population Stability Index (PSI) using quantile binning.

        PSI = sum((Actual% - Expected%) * ln(Actual% / Expected%))
        """
        if len(baseline) == 0 or len(production) == 0:
            return 0.0

        # Create quantile break points from baseline
        quantiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.percentile(baseline, quantiles)
        # Ensure strictly increasing breakpoints
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        # Bin counts
        baseline_counts, _ = np.histogram(baseline, bins=breakpoints)
        prod_counts, _ = np.histogram(production, bins=breakpoints)

        # Proportions
        p_base = (baseline_counts / len(baseline)) + eps
        p_prod = (prod_counts / len(production)) + eps

        # Normalize proportions
        p_base = p_base / np.sum(p_base)
        p_prod = p_prod / np.sum(p_prod)

        psi = np.sum((p_prod - p_base) * np.log(p_prod / p_base))
        return float(psi)

    def evaluate_feature_drift(
        self,
        feature_name: str,
        baseline_values: list[float],
        production_values: list[float],
    ) -> DriftReport:
        """Run KS-test and PSI against feature values to detect distribution shift."""
        base_arr = np.array([x for x in baseline_values if x is not None and not np.isnan(x)])
        prod_arr = np.array([x for x in production_values if x is not None and not np.isnan(x)])

        base_missing = 1.0 - (len(base_arr) / max(len(baseline_values), 1))
        prod_missing = 1.0 - (len(prod_arr) / max(len(production_values), 1))

        if len(base_arr) == 0 or len(prod_arr) == 0:
            return DriftReport(
                feature_name=feature_name,
                ks_statistic=0.0,
                ks_p_value=1.0,
                psi=0.0,
                drift_detected=False,
                drift_severity="NONE",
                baseline_missing_rate=float(base_missing),
                production_missing_rate=float(prod_missing),
                sample_size_baseline=len(baseline_values),
                sample_size_production=len(production_values),
            )

        # 1. Kolmogorov-Smirnov Test
        ks_res = stats.ks_2samp(base_arr, prod_arr)
        ks_stat = float(ks_res.statistic)
        ks_p = float(ks_res.pvalue)

        # 2. Population Stability Index
        psi = self.calculate_psi(base_arr, prod_arr)

        # Determine Drift Status & Severity
        if psi >= 0.20 or (ks_p < 0.01 and ks_stat > 0.15):
            drift_detected = True
            severity = "HIGH"
        elif psi >= 0.10 or ks_p < 0.05:
            drift_detected = True
            severity = "MODERATE"
        else:
            drift_detected = False
            severity = "NONE"

        return DriftReport(
            feature_name=feature_name,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
            psi=psi,
            drift_detected=drift_detected,
            drift_severity=severity,
            baseline_missing_rate=float(base_missing),
            production_missing_rate=float(prod_missing),
            sample_size_baseline=len(baseline_values),
            sample_size_production=len(production_values),
        )
