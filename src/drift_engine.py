import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from pathlib import Path

from src.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    PSI_MODERATE_THRESHOLD,
    PSI_CRITICAL_THRESHOLD,
    KS_ALPHA,
    ARTIFACTS_DIR
)

def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_buckets: int = 10, epsilon: float = 1e-4) -> float:
    """
    Calculates Population Stability Index (PSI) between expected (reference) and actual (production) data.
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins based on reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    try:
        bin_edges = np.percentile(expected, percentiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            return 0.0
        # Expand outer edges slightly to include min and max
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf
    except Exception:
        return 0.0

    # Calculate bucket frequencies
    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    # Convert to fractions with smoothing epsilon
    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * len(expected_counts))
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * len(actual_counts))

    # PSI Formula: sum((actual - expected) * ln(actual / expected))
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(np.round(psi_value, 4))

def perform_ks_test(reference: np.ndarray, production: np.ndarray) -> Tuple[float, float, bool]:
    """
    Performs Two-Sample Kolmogorov-Smirnov Test.
    Returns: (ks_stat, p_value, is_drifted)
    """
    reference = reference[~np.isnan(reference)]
    production = production[~np.isnan(production)]
    
    res = stats.ks_2samp(reference, production)
    is_drifted = bool(res.pvalue < KS_ALPHA)
    return float(np.round(res.statistic, 4)), float(res.pvalue), is_drifted

def perform_chi2_test(reference_cat: pd.Series, production_cat: pd.Series) -> Tuple[float, float, bool]:
    """
    Performs Chi-Square Goodness of Fit test for categorical features.
    """
    categories = sorted(list(set(reference_cat.unique()).union(set(production_cat.unique()))))
    ref_counts = reference_cat.value_counts().reindex(categories, fill_value=1)
    prod_counts = production_cat.value_counts().reindex(categories, fill_value=1)
    
    # Scale reference counts to match production sample size
    expected_counts = ref_counts * (len(production_cat) / len(reference_cat))
    res = stats.chisquare(f_obs=prod_counts, f_exp=expected_counts)
    is_drifted = bool(res.pvalue < KS_ALPHA)
    return float(np.round(res.statistic, 4)), float(res.pvalue), is_drifted

def evaluate_dataset_drift(
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    batch_name: str = "Production_Batch",
    ref_probs: np.ndarray = None,
    prod_probs: np.ndarray = None
) -> Dict[str, Any]:
    """
    Evaluates complete MLOps Drift:
    1. Feature Drift (Covariate Shift via PSI and KS-Test)
    2. Prediction Drift (Model Output Probabilities PSI)
    3. Label Drift (Prior Probability Shift P(Y))
    """
    feature_metrics = {}
    drifted_features_count = 0
    total_features = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
    psi_values = {}

    # 1. Numerical Features Analysis
    for feat in NUMERICAL_FEATURES:
        ref_vals = ref_df[feat].to_numpy()
        prod_vals = prod_df[feat].to_numpy()

        psi = calculate_psi(ref_vals, prod_vals)
        ks_stat, p_val, ks_drift = perform_ks_test(ref_vals, prod_vals)
        
        is_feature_drifted = (psi >= PSI_MODERATE_THRESHOLD) or ks_drift
        if is_feature_drifted:
            drifted_features_count += 1

        psi_values[feat] = psi
        feature_metrics[feat] = {
            "type": "numerical",
            "psi": psi,
            "ks_statistic": ks_stat,
            "ks_pvalue": p_val,
            "drift_detected": is_feature_drifted,
            "psi_level": "CRITICAL" if psi >= PSI_CRITICAL_THRESHOLD else ("MODERATE" if psi >= PSI_MODERATE_THRESHOLD else "STABLE")
        }

    # 2. Categorical Features Analysis
    for feat in CATEGORICAL_FEATURES:
        chi2_stat, p_val, chi_drift = perform_chi2_test(ref_df[feat], prod_df[feat])
        if chi_drift:
            drifted_features_count += 1
            
        feature_metrics[feat] = {
            "type": "categorical",
            "chi2_statistic": chi2_stat,
            "chi2_pvalue": p_val,
            "drift_detected": chi_drift
        }

    dataset_drift_ratio = drifted_features_count / total_features
    dataset_drift_detected = dataset_drift_ratio >= 0.30

    mean_psi = float(np.mean(list(psi_values.values())))
    max_psi = float(np.max(list(psi_values.values())))
    most_drifted_feature = max(psi_values, key=psi_values.get)

    # 3. Prediction Drift (PSI of predicted probabilities)
    prediction_psi = 0.0
    prediction_drift_detected = False
    if ref_probs is not None and prod_probs is not None:
        prediction_psi = calculate_psi(ref_probs, prod_probs)
        prediction_drift_detected = bool(prediction_psi >= PSI_MODERATE_THRESHOLD)

    # 4. Label Drift (Prior Probability Shift)
    ref_label_rate = float(ref_df[TARGET_COL].mean()) if TARGET_COL in ref_df else None
    prod_label_rate = float(prod_df[TARGET_COL].mean()) if TARGET_COL in prod_df else None
    label_drift_delta = 0.0
    label_drift_detected = False
    if ref_label_rate is not None and prod_label_rate is not None:
        label_drift_delta = abs(prod_label_rate - ref_label_rate)
        label_drift_detected = bool(label_drift_delta >= 0.15)

    summary = {
        "batch_name": batch_name,
        "sample_size": len(prod_df),
        "total_features": total_features,
        "drifted_features_count": drifted_features_count,
        "drift_feature_ratio": round(dataset_drift_ratio, 3),
        "dataset_drift_detected": dataset_drift_detected,
        "mean_psi": round(mean_psi, 4),
        "max_psi": round(max_psi, 4),
        "most_drifted_feature": most_drifted_feature,
        "prediction_psi": round(prediction_psi, 4),
        "prediction_drift_detected": prediction_drift_detected,
        "ref_default_rate": round(ref_label_rate, 4) if ref_label_rate is not None else None,
        "prod_default_rate": round(prod_label_rate, 4) if prod_label_rate is not None else None,
        "label_drift_delta": round(label_drift_delta, 4),
        "label_drift_detected": label_drift_detected,
        "feature_details": feature_metrics
    }

    return summary

def plot_drift_visualizations(
    ref_df: pd.DataFrame,
    prod_df: pd.DataFrame,
    drift_summary: Dict[str, Any],
    output_filename: str = "drift_analysis.png"
) -> str:
    """
    Generates and saves visual artifact comparing distributions and PSI metrics.
    """
    top_features = ["interest_rate", "debt_to_income_ratio", "revolving_utilization_rate", "credit_score"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"MLOps Drift Monitoring: Reference vs {drift_summary['batch_name']}", fontsize=15, fontweight="bold")

    for idx, feat in enumerate(top_features):
        ax = axes[idx // 2, idx % 2]
        
        # Reference KDE/histogram
        ax.hist(ref_df[feat], bins=30, alpha=0.5, density=True, label="Reference (Baseline)", color="#1f77b4")
        ax.hist(prod_df[feat], bins=30, alpha=0.5, density=True, label="Production Batch", color="#ff7f0e")

        feat_psi = drift_summary["feature_details"][feat]["psi"]
        feat_drift = drift_summary["feature_details"][feat]["drift_detected"]
        status_color = "red" if feat_drift else "green"

        ax.set_title(f"{feat}\nPSI: {feat_psi:.4f} (Drift: {feat_drift})", fontsize=11, fontweight="semibold", color=status_color)
        ax.set_xlabel(feat, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = ARTIFACTS_DIR / output_filename
    plt.savefig(output_path, dpi=200)
    plt.close()
    return str(output_path)
