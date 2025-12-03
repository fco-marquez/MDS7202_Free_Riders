"""
Model Training Module with Optuna and MLflow
=============================================
Handles hyperparameter optimization, model training, and experiment tracking.
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from optuna.visualization import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

import gc
import os

from mlflow_config import (
    get_experiment_name,
    get_or_create_experiment,
    log_metrics_from_dict,
    setup_mlflow,
)

# Import local modules
from pipeline import create_pipeline

# Configuration from environment
# ✅ INCREASED: Use 60% of training data instead of 20% for better learning
TRAIN_SAMPLE_FRAC = float(os.getenv("TRAIN_SAMPLE_FRAC", "0.6"))
VAL_SAMPLE_FRAC = float(os.getenv("VAL_SAMPLE_FRAC", "0.3"))
SHAP_SAMPLE_SIZE = int(os.getenv("SHAP_SAMPLE_SIZE", "500"))
# ✅ INCREASED: Allow higher imbalance ratio (15:1 instead of 8:1) to keep more data
IMBALANCE_RATIO_THRESHOLD = float(os.getenv("IMBALANCE_RATIO_THRESHOLD", "15"))
N_JOBS = int(os.getenv("N_JOBS", "-1"))


def load_train_val_data(
    train_path: str, val_path: str, sample_frac: float = None
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load training and validation data.

    Parameters
    ----------
    train_path : str
        Path to training data parquet
    val_path : str
        Path to validation data parquet

    Returns
    -------
    tuple
        (X_train, y_train, X_val, y_val)
    """
    # Use environment variable if sample_frac not provided
    if sample_frac is None:
        sample_frac = TRAIN_SAMPLE_FRAC

    print(f"Loading training data from: {train_path}")
    train_df = pd.read_parquet(train_path)
    original_size = len(train_df)
    print(f"Loaded {len(train_df):,} training samples")

    print(f"\nLoading validation data from: {val_path}")
    val_df = pd.read_parquet(val_path)
    original_val_size = len(val_df)
    print(f"Loaded {len(val_df):,} validation samples")

    # =========================================================================
    # STRATIFIED SAMPLING: Keep ALL positives, sample negatives
    # This is crucial for imbalanced data - random sampling loses signal!
    # =========================================================================
    if sample_frac < 1.0:
        print(f"\n📊 Applying STRATIFIED sampling (keeping all positives)...")

        # Training set
        train_positives = train_df[train_df["bought"] == 1]
        train_negatives = train_df[train_df["bought"] == 0]
        n_neg_to_keep = int(len(train_negatives) * sample_frac)
        train_negatives_sampled = train_negatives.sample(
            n=n_neg_to_keep, random_state=42
        )
        train_df = pd.concat(
            [train_positives, train_negatives_sampled], ignore_index=True
        )
        print(
            f"  Training: kept all {len(train_positives):,} positives + {n_neg_to_keep:,} negatives = {len(train_df):,} total"
        )

        # Validation set
        val_positives = val_df[val_df["bought"] == 1]
        val_negatives = val_df[val_df["bought"] == 0]
        n_val_neg_to_keep = int(len(val_negatives) * VAL_SAMPLE_FRAC)
        val_negatives_sampled = val_negatives.sample(
            n=n_val_neg_to_keep, random_state=42
        )
        val_df = pd.concat([val_positives, val_negatives_sampled], ignore_index=True)
        print(
            f"  Validation: kept all {len(val_positives):,} positives + {n_val_neg_to_keep:,} negatives = {len(val_df):,} total"
        )

        # Clean up
        del train_positives, train_negatives, train_negatives_sampled
        del val_positives, val_negatives, val_negatives_sampled
        gc.collect()

    # Separate features and target
    X_train = train_df.drop(columns=["bought"])
    y_train = train_df["bought"]

    X_val = val_df.drop(columns=["bought"])
    y_val = val_df["bought"]

    # Print class distribution
    print("\nClass distribution in training set:")
    print(y_train.value_counts())

    n_positive = (y_train == 1).sum()
    n_negative = (y_train == 0).sum()

    # Handle edge case: no positive samples
    if n_positive == 0:
        raise ValueError("No positive samples found in training data!")

    # Subsample if imbalance ratio is too high (IMBALANCE_RATIO_THRESHOLD)
    imbalance_ratio = n_negative / n_positive

    if imbalance_ratio > IMBALANCE_RATIO_THRESHOLD:
        print(
            f"\nImbalance ratio {imbalance_ratio:.2f} exceeds threshold {IMBALANCE_RATIO_THRESHOLD}. Subsampling majority class..."
        )

        minority_indices = y_train[y_train == 1].index
        majority_indices = y_train[y_train == 0].index

        # Calculate number of majority samples to keep
        n_minority = len(minority_indices)
        n_majority_to_keep = int(n_minority * IMBALANCE_RATIO_THRESHOLD)

        # Ensure we don't try to sample more than available
        n_majority_to_keep = min(n_majority_to_keep, len(majority_indices))

        # Randomly sample majority class indices (with seed for reproducibility)
        np.random.seed(42)
        sampled_majority_indices = np.random.choice(
            majority_indices, size=n_majority_to_keep, replace=False
        )

        # Combine minority indices with sampled majority indices
        selected_indices = np.concatenate([minority_indices, sampled_majority_indices])

        # Shuffle indices to avoid ordered pattern (minority first, then majority)
        np.random.seed(42)
        np.random.shuffle(selected_indices)

        # Subset training data
        X_train = X_train.loc[selected_indices]
        y_train = y_train.loc[selected_indices]

        print(f"After subsampling, training set size: {len(y_train):,}")
        print(f"  New class distribution:")
        print(f"    Class 0: {(y_train == 0).sum():,}")
        print(f"    Class 1: {(y_train == 1).sum():,}")
        print(
            f"  New imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1"
        )
    else:
        print(
            f"Imbalance ratio {imbalance_ratio:.2f} within acceptable range. No subsampling applied."
        )

    return X_train, y_train, X_val, y_val


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate scale_pos_weight for XGBoost to handle class imbalance.

    Parameters
    ----------
    y_train : pd.Series
        Training labels

    Returns
    -------
    float
        Scale pos weight value
    """
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    return scale_pos_weight


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    experiment_name: str = None,
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna with MLflow tracking.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    n_trials : int
        Number of Optuna trials
    experiment_name : str
        MLflow experiment name

    Returns
    -------
    dict
        Best hyperparameters
    """
    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 60)

    # Setup MLflow
    setup_mlflow()
    if experiment_name is None:
        # Use fixed experiment name from environment variable
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sodai_drinks_prediction")

    experiment_id = get_or_create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    # Create preprocessing pipeline
    preprocessing_pipeline = create_pipeline()

    # Fit preprocessing pipeline on training data
    # IMPORTANT: Add 'bought' column to X so FeatureEngineer can calculate features
    print("\nFitting preprocessing pipeline...")
    X_train_with_target = X_train.copy()
    X_train_with_target["bought"] = y_train.values
    X_train_processed = preprocessing_pipeline.fit_transform(
        X_train_with_target, y_train
    )
    del X_train_with_target
    gc.collect()

    # Transform validation data (also needs 'bought' for feature calculation)
    X_val_with_target = X_val.copy()
    X_val_with_target["bought"] = y_val.values
    X_val_processed = preprocessing_pipeline.transform(X_val_with_target)
    del X_val_with_target
    gc.collect()

    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed validation data shape: {X_val_processed.shape}")

    # Objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters - EXPANDED GRID for better model capacity
        # ✅ FIXED: Use calculated scale_pos_weight instead of fixed threshold
        n_jobs = N_JOBS

        params = {
            # --- Parámetros Fijos (Eficiencia y Entorno) ---
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,  # ✅ Uses value calculated from data
            "tree_method": "hist",
            "random_state": 42,
            # Adjust to the resources of your Docker container.
            # Avoid using -1 to prevent CPU overhead in limited containers.
            "n_jobs": n_jobs,
            # ✅ EXPANDED: Deeper trees (5-10 instead of 3-6)
            "max_depth": trial.suggest_int("max_depth", 5, 10),
            # ✅ EXPANDED: Higher learning rates (up to 0.3 instead of 0.1)
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            # ✅ EXPANDED: Less restrictive (1-5 instead of 1-7)
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 5),
            # ✅ EXPANDED: Allow full sampling (up to 1.0 instead of 0.9)
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # ✅ TUNABLE: Gamma now tunable instead of fixed at 1.0
            "gamma": trial.suggest_float("gamma", 0, 2),
            # ✅ TUNABLE: L1 regularization now tunable instead of fixed at 0.5
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            # ✅ TUNABLE: L2 regularization now tunable instead of fixed at 1.0
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2),
        }

        # Start MLflow run
        with mlflow.start_run(
            run_name=f"trial_{trial.number}", nested=True, log_system_metrics=True
        ):
            # Train model
            model = XGBClassifier(**params)
            model.fit(X_train_processed, y_train)

            # Predict
            y_pred = model.predict(X_val_processed)
            y_pred_proba = model.predict_proba(X_val_processed)[:, 1]

            # Calculate metrics
            recall = recall_score(y_val, y_pred, zero_division=0)
            precision = precision_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            auc_pr = average_precision_score(y_val, y_pred_proba)

            # ✅ F2-score: Weighted F-score that prioritizes recall over precision
            # Beta=2 means recall is 2x more important than precision
            # This helps detect more positive cases (bought=1) even if precision drops slightly
            beta = 2
            f2 = (
                ((1 + beta**2) * precision * recall / (beta**2 * precision + recall))
                if (precision + recall) > 0
                else 0
            )

            # Log parameters and metrics
            mlflow.log_params(params)
            metrics = {
                "val_recall": recall,
                "val_precision": precision,
                "val_f1": f1,
                "val_f2": f2,  # ✅ Log F2-score
                "val_auc_pr": auc_pr,
            }
            log_metrics_from_dict(metrics)

            # Clean up to save memory
            del model
            gc.collect()

            # ✅ Optuna optimizes for F2-score (prioritizes recall for imbalanced data)
            return f2

    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"xgboost_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Start parent MLflow run with timestamp for organization
    optuna_run_name = f"optuna_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=optuna_run_name):
        # Run optimization
        print(f"\nStarting optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Log best results
        print("\n" + "=" * 60)
        print("OPTIMIZATION RESULTS")
        print("=" * 60)
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best F2-score: {study.best_value:.4f}")  # ✅ Changed from f1 to F2
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Log best params
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f2", study.best_value)  # ✅ Changed from recall to f2

        # Generate and log Optuna visualizations
        print("\nGenerating Optuna visualization plots...")

        # Optimization history
        fig1 = plot_optimization_history(study)
        fig1.update_layout(title="Optimization History")
        mlflow.log_figure(fig1, "plots/optimization_history.html")

        # Parameter importances
        fig2 = plot_param_importances(study)
        fig2.update_layout(title="Hyperparameter Importances")
        mlflow.log_figure(fig2, "plots/param_importances.html")

        # Parallel coordinates
        fig3 = plot_parallel_coordinate(study)
        fig3.update_layout(title="Parallel Coordinate Plot")
        mlflow.log_figure(fig3, "plots/parallel_coordinate.html")

        print("Optuna plots logged to MLflow")

    print("=" * 60)

    # Add fixed parameters to best params
    best_params = study.best_params.copy()

    return best_params


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    best_params: Dict[str, Any],
    experiment_name: str = None,
    output_model_path: str = None,
) -> Pipeline:
    """
    Train final model with best hyperparameters and log to MLflow.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame
        Validation features
    y_val : pd.Series
        Validation labels
    best_params : dict
        Best hyperparameters from optimization
    experiment_name : str
        MLflow experiment name
    output_model_path : str
        Path to save model locally

    Returns
    -------
    Pipeline
        Trained pipeline
    """
    print("=" * 60)
    print("TRAINING FINAL MODEL")
    print("=" * 60)

    # Setup MLflow
    setup_mlflow()
    if experiment_name is None:
        # Use same fixed experiment name as optimization
        experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sodai_drinks_prediction")

    mlflow.set_experiment(experiment_name)

    # Use descriptive run name with timestamp
    run_name = f"final_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # Create full pipeline
        preprocessing_pipeline = create_pipeline()
        model = XGBClassifier(**best_params)

        full_pipeline = Pipeline(
            [("preprocessing", preprocessing_pipeline), ("model", model)]
        )

        # Train
        print("\nTraining final model...")
        full_pipeline.fit(X_train, y_train)

        # Predictions
        # IMPORTANT: Add 'bought' column to X so FeatureEngineer can calculate features
        # This is needed because predict() doesn't pass y, but features need historical purchases
        print("Generating predictions...")

        # For training set evaluation
        X_train_with_target = X_train.copy()
        X_train_with_target["bought"] = y_train.values
        y_train_pred = full_pipeline.predict(X_train_with_target)
        y_train_proba = full_pipeline.predict_proba(X_train_with_target)[:, 1]
        del X_train_with_target
        gc.collect()

        # For validation set evaluation
        X_val_with_target = X_val.copy()
        X_val_with_target["bought"] = y_val.values
        y_val_pred = full_pipeline.predict(X_val_with_target)
        y_val_proba = full_pipeline.predict_proba(X_val_with_target)[:, 1]
        del X_val_with_target
        gc.collect()

        # Calculate metrics
        train_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred, zero_division=0),
            "train_recall": recall_score(y_train, y_train_pred, zero_division=0),
            "train_f1": f1_score(y_train, y_train_pred, zero_division=0),
            "train_auc_pr": average_precision_score(y_train, y_train_proba),
        }

        val_metrics = {
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred, zero_division=0),
            "val_recall": recall_score(y_val, y_val_pred, zero_division=0),
            "val_f1": f1_score(y_val, y_val_pred, zero_division=0),
            "val_auc_pr": average_precision_score(y_val, y_val_proba),
        }

        # Log parameters and metrics
        mlflow.log_params(best_params)
        log_metrics_from_dict(train_metrics)
        log_metrics_from_dict(val_metrics)

        # Print metrics
        print("\nTraining Metrics:")
        for key, value in train_metrics.items():
            print(f"  {key}: {value:.4f}")

        print("\nValidation Metrics:")
        for key, value in val_metrics.items():
            print(f"  {key}: {value:.4f}")

        # Classification report
        print("\nClassification Report (Validation):")
        report = classification_report(y_val, y_val_pred)
        print(report)
        mlflow.log_text(report, "classification_report.txt")

        # Confusion matrix
        print("\nGenerating confusion matrix...")
        cm = confusion_matrix(y_val, y_val_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Validation Set")
        mlflow.log_figure(fig, "plots/confusion_matrix.png")
        plt.close(fig)

        # Precision-Recall curve
        print("Generating Precision-Recall curve...")
        precision, recall, _ = precision_recall_curve(y_val, y_val_proba)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(
            f'Precision-Recall Curve (AUC-PR: {val_metrics["val_auc_pr"]:.4f})'
        )
        ax.grid(True, alpha=0.3)
        mlflow.log_figure(fig, "plots/precision_recall_curve.png")
        plt.close(fig)

        # SHAP values
        print("\nCalculating SHAP values...")
        try:
            # Get processed data for SHAP - need to include 'bought' for feature engineering
            X_val_for_shap = X_val.copy()
            X_val_for_shap["bought"] = y_val.values
            X_val_processed = full_pipeline.named_steps["preprocessing"].transform(
                X_val_for_shap
            )
            del X_val_for_shap

            # Sample for SHAP (to speed up)
            sample_size = min(SHAP_SAMPLE_SIZE, len(X_val_processed))
            X_sample = X_val_processed.sample(n=sample_size, random_state=42)

            # Create SHAP explainer
            explainer = shap.TreeExplainer(full_pipeline.named_steps["model"])
            shap_values = explainer.shap_values(X_sample)

            # SHAP summary plot
            fig, ax = plt.subplots(figsize=(12, 8))
            shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
            plt.tight_layout()
            mlflow.log_figure(fig, "plots/shap_summary.png")
            plt.close(fig)

            # SHAP bar plot (feature importance)
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                shap_values, X_sample, plot_type="bar", show=False, max_display=20
            )
            plt.tight_layout()
            mlflow.log_figure(fig, "plots/shap_importance.png")
            plt.close(fig)

            print("SHAP plots generated and logged")

            # Clean up SHAP objects
            del explainer, shap_values, X_sample
            gc.collect()
        except Exception as e:
            print(f"Warning: SHAP calculation failed: {e}")

        # Log model to MLflow
        print("\nLogging model to MLflow...")
        mlflow.sklearn.log_model(full_pipeline, "model")
        print("Model logged to MLflow successfully")

        # Save model locally as fallback
        if output_model_path:
            output_path = Path(output_model_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(full_pipeline, output_path)
            print(f"Model saved locally to: {output_path}")

    print("=" * 60)

    return full_pipeline


def run_full_training(
    train_data_path: str,
    val_data_path: str,
    n_trials: int = 50,
    output_model_path: str = None,
) -> Pipeline:
    """
    Main function to run full training pipeline (for Airflow task).

    Parameters
    ----------
    train_data_path : str
        Path to training data
    val_data_path : str
        Path to validation data
    n_trials : int
        Number of Optuna trials
    output_model_path : str
        Path to save model locally

    Returns
    -------
    Pipeline
        Trained pipeline
    """
    # Load data
    X_train, y_train, X_val, y_val = load_train_val_data(train_data_path, val_data_path)
    print("\nData loaded successfully.")

    # Optimize hyperparameters
    best_params = optimize_hyperparameters(
        X_train, y_train, X_val, y_val, n_trials=n_trials
    )
    print("\nHyperparameter optimization completed.")

    # Clean up after optimization
    gc.collect()

    # Train final model
    pipeline = train_final_model(
        X_train,
        y_train,
        X_val,
        y_val,
        best_params=best_params,
        output_model_path=output_model_path,
    )

    # Final cleanup
    del X_train, y_train, X_val, y_val
    gc.collect()

    return pipeline
