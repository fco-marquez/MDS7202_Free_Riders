"""
Data Drift Detection Module
============================
Detects statistical drift in data distributions using KS-test and Chi-square test.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List
import json
from pathlib import Path
from datetime import datetime


def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (tuple, list)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    return obj


# Features to monitor for drift
NUMERICAL_FEATURES = [
    'size',
    'num_deliver_per_week',
    'num_visit_per_week',
    'items',
    'recency',
    'frequency',
    'customer_product_share',
    'trend'
]

CATEGORICAL_FEATURES = [
    'customer_type',
    'brand',
    'category',
    'sub_category',
    'segment',
    'package'
]

# Drift thresholds
KS_THRESHOLD = 0.05  # p-value threshold for KS test
CHI2_THRESHOLD = 0.05  # p-value threshold for Chi-square test
DRIFT_THRESHOLD = 0.3  # Proportion of features that can show drift before triggering retrain


def kolmogorov_smirnov_test(
    reference_data: pd.Series,
    current_data: pd.Series
) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test to detect distribution drift.

    Parameters
    ----------
    reference_data : pd.Series
        Historical/reference data
    current_data : pd.Series
        New/current data to compare

    Returns
    -------
    tuple
        (KS statistic, p-value)
    """
    # Remove NaN values
    ref_clean = reference_data.dropna()
    curr_clean = current_data.dropna()

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0, 1.0

    # Perform KS test
    statistic, p_value = stats.ks_2samp(ref_clean, curr_clean)

    return statistic, p_value


def chi_square_test(
    reference_data: pd.Series,
    current_data: pd.Series
) -> Tuple[float, float]:
    """
    Perform Chi-square test to detect categorical drift.

    Parameters
    ----------
    reference_data : pd.Series
        Historical/reference data
    current_data : pd.Series
        New/current data to compare

    Returns
    -------
    tuple
        (Chi-square statistic, p-value)
    """
    # Get value counts
    ref_counts = reference_data.value_counts()
    curr_counts = current_data.value_counts()

    # Get all unique categories
    all_categories = set(ref_counts.index) | set(curr_counts.index)

    # Create frequency tables
    ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
    curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]

    # Avoid zero frequencies (add 1 smoothing)
    ref_freq = [f + 1 for f in ref_freq]
    curr_freq = [f + 1 for f in curr_freq]

    # Perform chi-square test
    try:
        statistic, p_value = stats.chisquare(curr_freq, ref_freq)
    except Exception as e:
        print(f"Chi-square test failed: {e}")
        return 0.0, 1.0

    return statistic, p_value


def calculate_drift_statistics(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Calculate drift statistics for all monitored features.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Historical/reference dataset
    current_df : pd.DataFrame
        New/current dataset

    Returns
    -------
    dict
        Dictionary with drift statistics for each feature
    """
    drift_stats = {}

    # Numerical features - KS test
    for feature in NUMERICAL_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            print(f"Warning: Feature '{feature}' not found in data, skipping")
            continue

        ks_stat, p_value = kolmogorov_smirnov_test(
            reference_df[feature],
            current_df[feature]
        )

        drift_stats[feature] = {
            'test': 'kolmogorov_smirnov',
            'statistic': float(ks_stat),
            'p_value': float(p_value),
            'drift_detected': p_value < KS_THRESHOLD,
            'threshold': KS_THRESHOLD
        }

    # Categorical features - Chi-square test
    for feature in CATEGORICAL_FEATURES:
        if feature not in reference_df.columns or feature not in current_df.columns:
            print(f"Warning: Feature '{feature}' not found in data, skipping")
            continue

        chi2_stat, p_value = chi_square_test(
            reference_df[feature],
            current_df[feature]
        )

        drift_stats[feature] = {
            'test': 'chi_square',
            'statistic': float(chi2_stat),
            'p_value': float(p_value),
            'drift_detected': p_value < CHI2_THRESHOLD,
            'threshold': CHI2_THRESHOLD
        }

    return drift_stats


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = None
) -> Tuple[bool, Dict]:
    """
    Detect if drift exists in the data.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Historical/reference dataset
    current_df : pd.DataFrame
        New/current dataset
    output_path : str, optional
        Path to save drift report JSON

    Returns
    -------
    tuple
        (drift_detected: bool, drift_report: dict)
    """
    print("=" * 60)
    print("DRIFT DETECTION ANALYSIS")
    print("=" * 60)

    # Calculate drift statistics
    drift_stats = calculate_drift_statistics(reference_df, current_df)

    # Count how many features show drift
    total_features = len(drift_stats)
    features_with_drift = sum(
        1 for stats in drift_stats.values() if stats['drift_detected']
    )

    drift_ratio = features_with_drift / total_features if total_features > 0 else 0

    # Decision: retrain if more than threshold of features show drift
    needs_retrain = drift_ratio > DRIFT_THRESHOLD

    # Create detailed report
    drift_report = {
        'timestamp': datetime.now().isoformat(),
        'reference_data_shape': reference_df.shape,
        'current_data_shape': current_df.shape,
        'total_features_monitored': total_features,
        'features_with_drift': features_with_drift,
        'drift_ratio': float(drift_ratio),
        'drift_threshold': DRIFT_THRESHOLD,
        'needs_retrain': needs_retrain,
        'feature_statistics': drift_stats
    }

    # Print summary
    print(f"\nReference data: {reference_df.shape[0]:,} samples")
    print(f"Current data: {current_df.shape[0]:,} samples")
    print(f"\nFeatures monitored: {total_features}")
    print(f"Features with drift: {features_with_drift}")
    print(f"Drift ratio: {drift_ratio:.2%}")
    print(f"Drift threshold: {DRIFT_THRESHOLD:.2%}")
    print(f"\n{'🚨 DRIFT DETECTED - Retraining recommended' if needs_retrain else '✅ No significant drift - Retraining not needed'}")

    # Print details of features with drift
    if features_with_drift > 0:
        print("\nFeatures with detected drift:")
        for feature, stats in drift_stats.items():
            if stats['drift_detected']:
                print(f"  - {feature}: p-value = {stats['p_value']:.4f} (test: {stats['test']})")

    print("=" * 60)

    # Save report
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert to JSON-serializable format
        serializable_report = convert_to_json_serializable(drift_report)
        with open(output_path, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        print(f"\nDrift report saved to: {output_path}")

    return needs_retrain, drift_report


def load_reference_data(reference_path: str) -> pd.DataFrame:
    """
    Load reference/historical data for drift comparison.

    Parameters
    ----------
    reference_path : str
        Path to reference data (parquet file)

    Returns
    -------
    pd.DataFrame
        Reference dataset
    """
    print(f"Loading reference data from: {reference_path}")
    df = pd.read_parquet(reference_path)
    print(f"Loaded {len(df):,} rows")
    return df


def sample_for_drift_detection(
    df: pd.DataFrame,
    sample_size: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample data for drift detection (for large datasets).

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    sample_size : int
        Number of samples to use
    random_state : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Sampled dataset
    """
    if len(df) <= sample_size:
        return df

    return df.sample(n=sample_size, random_state=random_state)


# Main function for use in Airflow DAG
def run_drift_detection(
    reference_data_path: str,
    current_data_path: str,
    output_report_path: str = None
) -> bool:
    """
    Main function to run drift detection (for Airflow task).

    Parameters
    ----------
    reference_data_path : str
        Path to reference/historical data
    current_data_path : str
        Path to new/current data
    output_report_path : str, optional
        Path to save drift report

    Returns
    -------
    bool
        True if drift detected (needs retrain), False otherwise
    """
    # Load data
    reference_df = load_reference_data(reference_data_path)
    current_df = pd.read_parquet(current_data_path)

    # Sample if too large (to speed up drift detection)
    reference_sample = sample_for_drift_detection(reference_df, sample_size=10000)
    current_sample = sample_for_drift_detection(current_df, sample_size=10000)

    # Detect drift
    needs_retrain, drift_report = detect_drift(
        reference_sample,
        current_sample,
        output_path=output_report_path
    )

    return needs_retrain
