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
TRAIN_SAMPLE_FRAC = float(os.getenv("TRAIN_SAMPLE_FRAC", "0.2"))  # Default 20%
VAL_SAMPLE_FRAC = float(os.getenv("VAL_SAMPLE_FRAC", "0.3"))  # Default 30%
SHAP_SAMPLE_SIZE = int(os.getenv("SHAP_SAMPLE_SIZE", "500"))  # Reduced from 1000


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
    # Read only needed columns to save memory
    train_df = pd.read_parquet(train_path)
    original_size = len(train_df)

    # ✅ MUESTREAR para reducir memoria durante desarrollo
    if sample_frac < 1.0:
        print(
            f"Sampling {sample_frac*100}% of training data (from {original_size:,} samples)..."
        )
        train_df = train_df.sample(frac=sample_frac, random_state=42)
    print(f"Loaded {len(train_df):,} training samples")

    print(f"\nLoading validation data from: {val_path}")
    val_df = pd.read_parquet(val_path)
    original_val_size = len(val_df)

    # Also sample validation to reduce memory
    if VAL_SAMPLE_FRAC < 1.0:
        print(
            f"Sampling {VAL_SAMPLE_FRAC*100}% of validation data (from {original_val_size:,} samples)..."
        )
        val_df = val_df.sample(frac=VAL_SAMPLE_FRAC, random_state=42)
    print(f"Loaded {len(val_df):,} validation samples")

    # Separate features and target
    X_train = train_df.drop(columns=["bought"])
    y_train = train_df["bought"]

    X_val = val_df.drop(columns=["bought"])
    y_val = val_df["bought"]

    # Print class distribution
    print("\nClass distribution in training set:")
    print(y_train.value_counts())
    print(f"Class imbalance ratio: {(y_train == 0).sum() / (y_train == 1).sum():.2f}:1")

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
    print("\nFitting preprocessing pipeline...")
    X_train_processed = preprocessing_pipeline.fit_transform(X_train, y_train)
    X_val_processed = preprocessing_pipeline.transform(X_val)

    print(f"Processed training data shape: {X_train_processed.shape}")
    print(f"Processed validation data shape: {X_val_processed.shape}")

    # Objective function for Optuna
    def objective(trial):
        # Suggest hyperparameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
            "tree_method": "hist",  # Much faster and memory efficient
            # Hyperparameters to optimize
            "max_depth": trial.suggest_int("max_depth", 3, 8),  # Reduced from 10
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            ),  # Higher min
            "n_estimators": trial.suggest_int(
                "n_estimators", 50, 300
            ),  # Reduced from 500
            "min_child_weight": trial.suggest_int(
                "min_child_weight", 1, 7
            ),  # Reduced from 10
            "gamma": trial.suggest_float("gamma", 0, 0.3),  # Reduced from 0.5
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),  # Higher min
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.7, 1.0
            ),  # Higher min
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5),  # Reduced from 1.0
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 0.5),  # Reduced from 1.0
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
            recall = recall_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            auc_pr = average_precision_score(y_val, y_pred_proba)

            # Log parameters and metrics
            mlflow.log_params(params)
            metrics = {
                "val_recall": recall,
                "val_precision": precision,
                "val_f1": f1,
                "val_auc_pr": auc_pr,
            }
            log_metrics_from_dict(metrics)

            # Clean up to save memory
            del model
            gc.collect()

            # Optuna optimizes for recall (primary metric)
            return recall

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
        print(f"Best recall: {study.best_value:.4f}")
        print("\nBest hyperparameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")

        # Log best params
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_recall", study.best_value)

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
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "scale_pos_weight": scale_pos_weight,
            "random_state": 42,
            "n_jobs": -1,
        }
    )

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
        print("Generating predictions...")
        y_train_pred = full_pipeline.predict(X_train)
        y_val_pred = full_pipeline.predict(X_val)
        y_train_proba = full_pipeline.predict_proba(X_train)[:, 1]
        y_val_proba = full_pipeline.predict_proba(X_val)[:, 1]

        # Calculate metrics
        train_metrics = {
            "train_accuracy": accuracy_score(y_train, y_train_pred),
            "train_precision": precision_score(y_train, y_train_pred),
            "train_recall": recall_score(y_train, y_train_pred),
            "train_f1": f1_score(y_train, y_train_pred),
            "train_auc_pr": average_precision_score(y_train, y_train_proba),
        }

        val_metrics = {
            "val_accuracy": accuracy_score(y_val, y_val_pred),
            "val_precision": precision_score(y_val, y_val_pred),
            "val_recall": recall_score(y_val, y_val_pred),
            "val_f1": f1_score(y_val, y_val_pred),
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
            # Get processed data for SHAP
            X_val_processed = full_pipeline.named_steps["preprocessing"].transform(
                X_val
            )

            # Sample for SHAP (to speed up)
            sample_size = min(1000, len(X_val_processed))
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
