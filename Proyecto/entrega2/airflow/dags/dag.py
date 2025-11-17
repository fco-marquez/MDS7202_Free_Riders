"""
SodAI Drinks Prediction Pipeline
=================================
Complete Airflow DAG for automated ML pipeline with drift detection and retraining.

This DAG orchestrates:
1. Data extraction and preprocessing
2. Data splitting
3. Drift detection
4. Conditional model retraining (if drift detected)
5. Prediction generation for next week

Author: Free Riders Team
Date: 2024
"""

import datetime as dt
import os
from pathlib import Path

from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from drift_detector import run_drift_detection

# Import local modules
from load_and_preprocess import run_preprocessing_pipeline
from pipeline import run_data_splitting
from predict_module import run_prediction_pipeline
from train_module import run_full_training

from airflow import DAG

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base paths
BASE_DIR = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow"))
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "predictions"
DRIFT_REPORTS_DIR = BASE_DIR / "drift_reports"
MODELS_DIR = BASE_DIR / "models"

# Create directories
for directory in [PROCESSED_DATA_DIR, PREDICTIONS_DIR, DRIFT_REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
FINAL_DATA_PATH = PROCESSED_DATA_DIR / "final_data.parquet"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.parquet"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.parquet"
CUSTOMERS_PATH = RAW_DATA_DIR / "clientes.parquet"
PRODUCTS_PATH = RAW_DATA_DIR / "productos.parquet"
DRIFT_REPORT_PATH = DRIFT_REPORTS_DIR / "drift_report_{execution_date}.json"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREDICTIONS_PATH = PREDICTIONS_DIR / "predictions_{execution_date}.parquet"

# Training configuration from environment variables
N_OPTUNA_TRIALS = int(os.getenv("N_OPTUNA_TRIALS", "50"))
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "sodai_drinks_prediction")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))


# ============================================================================
# TASK FUNCTIONS
# ============================================================================


def extract_new_data(**context):
    """
    Extract new data (placeholder for actual data extraction).

    In production, this would:
    - Check for new data files in a specific location
    - Download from database or API
    - Validate data schema

    For now, it assumes data is already in the raw folder.
    """
    print("=" * 60)
    print("EXTRACTING NEW DATA")
    print("=" * 60)

    # Check if raw data exists
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    raw_files = list(RAW_DATA_DIR.glob("*.parquet"))
    print(f"Found {len(raw_files)} raw data files:")
    for file in raw_files:
        print(f"  - {file.name}")

    if len(raw_files) == 0:
        raise FileNotFoundError("No raw data files found!")

    print("\n✓ Data extraction completed")
    print("=" * 60)


def preprocess_data(**context):
    """
    Preprocess raw data: clean, transform, and create universe.
    """
    run_preprocessing_pipeline(
        raw_data_folder=str(RAW_DATA_DIR), output_data_path=str(FINAL_DATA_PATH)
    )


def split_data(**context):
    """
    Split processed data into train, validation, and test sets.
    """
    run_data_splitting(
        input_data_path=str(FINAL_DATA_PATH), output_dir=str(PROCESSED_DATA_DIR)
    )


def detect_drift(**context):
    """
    Detect drift in new data compared to reference data.
    """
    execution_date = context["ds"]
    drift_report_path = str(DRIFT_REPORT_PATH).format(execution_date=execution_date)

    needs_retrain = run_drift_detection(
        reference_data_path=str(TRAIN_DATA_PATH),
        current_data_path=str(FINAL_DATA_PATH),
        output_report_path=drift_report_path,
    )

    # Push result to XCom for branching
    context["task_instance"].xcom_push(key="needs_retrain", value=needs_retrain)

    return needs_retrain


def decide_retrain(**context):
    """
    Branching decision: retrain if drift detected OR no model exists, skip otherwise.
    """
    ti = context["task_instance"]
    needs_retrain = ti.xcom_pull(task_ids="detect_drift", key="needs_retrain")

    # Check if model exists
    model_exists = MODEL_PATH.exists()

    print("=" * 60)
    print("RETRAINING DECISION")
    print("=" * 60)
    print(f"Drift detected: {needs_retrain}")
    print(f"Model exists: {model_exists}")

    # Retrain if drift detected OR no model exists (first run)
    if needs_retrain or not model_exists:
        if not model_exists:
            print("→ Reason: No existing model found (first run)")
        else:
            print("→ Reason: Drift detected")
        print("→ Decision: RETRAIN MODEL")
        print("=" * 60)
        return "train_model"
    else:
        print("→ Decision: SKIP RETRAINING")
        print("=" * 60)
        return "skip_retrain"


def train_model(**context):
    """
    Train model with hyperparameter optimization using Optuna and MLflow.
    """
    pipeline = run_full_training(
        train_data_path=str(TRAIN_DATA_PATH),
        val_data_path=str(VAL_DATA_PATH),
        n_trials=N_OPTUNA_TRIALS,
        output_model_path=str(MODEL_PATH),
    )

    print(f"\nModel saved to: {MODEL_PATH}")


def generate_predictions(**context):
    """
    Generate predictions for next week using the best model.

    Creates universe for the week AFTER the most recent week in historical data.
    This follows the requirement: "predict for the week following the most recent in the data"

    NOTE: To avoid OOM errors, we limit to first 100 customers by default.
    Remove max_customers parameter to generate for all customers (requires more RAM).
    """
    execution_date = context["ds"]
    predictions_path = str(PREDICTIONS_PATH).format(execution_date=execution_date)

    # Limit customers to avoid OOM (1569 customers × 971 products = 1.5M predictions)
    # For production with sufficient RAM, remove max_customers parameter
    predictions = run_prediction_pipeline(
        customers_path=str(CUSTOMERS_PATH),
        products_path=str(PRODUCTS_PATH),
        historical_data_path=str(FINAL_DATA_PATH),
        model_experiment_name=MLFLOW_EXPERIMENT,
        model_fallback_path=str(MODEL_PATH),
        output_predictions_path=predictions_path,
        max_customers=100,  # Limit to 100 customers (~97K predictions instead of 1.5M)
        batch_size=20000,   # Process 20K rows at a time
    )

    print(f"\n✓ Predictions saved to: {predictions_path}")
    print(f"✓ Generated {len(predictions):,} predictions")


# ============================================================================
# DAG DEFINITION
# ============================================================================

default_args = {
    "owner": "free-riders",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": dt.timedelta(minutes=5),
}

with DAG(
    dag_id="sodai_prediction_pipeline",
    default_args=default_args,
    description="Complete ML pipeline for SodAI drinks prediction with drift detection",
    schedule_interval=None,  # Manual trigger (in production: '@weekly' or cron schedule)
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    tags=["ml", "prediction", "drinks", "mlflow", "optuna"],
) as dag:

    # ========================================================================
    # TASK DEFINITIONS
    # ========================================================================

    start = EmptyOperator(
        task_id="start",
        doc_md="""
        ### Pipeline Start
        Begins the SodAI drinks prediction pipeline.
        """,
    )

    extract = PythonOperator(
        task_id="extract_new_data",
        python_callable=extract_new_data,
        doc_md="""
        ### Extract New Data
        Checks for and validates new raw data files.
        In production, this would fetch data from external sources.
        """,
    )

    preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
        doc_md="""
        ### Preprocess Data
        - Loads raw data (customers, products, transactions)
        - Cleans transactions (remove duplicates, filter invalid items)
        - Optimizes datatypes
        - Creates week and objective variables
        - Generates customer-product-week universe
        """,
    )

    split = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        doc_md="""
        ### Split Data
        Splits processed data into train (80%) and validation (20%) sets.
        Respects temporal ordering to prevent data leakage.
        
        NO test set is created - predictions will be made for the NEXT WEEK
        after the most recent data, following project requirements.
        """,
    )

    drift_detection = PythonOperator(
        task_id="detect_drift",
        python_callable=detect_drift,
        doc_md="""
        ### Detect Drift
        Performs statistical drift detection:
        - KS test for numerical features
        - Chi-square test for categorical features
        - Generates drift report
        - Determines if retraining is needed
        """,
    )

    branch = BranchPythonOperator(
        task_id="decide_retrain",
        python_callable=decide_retrain,
        doc_md="""
        ### Branching Decision
        Decides whether to retrain the model based on drift detection results.
        - If drift detected → train_model
        - If no drift → skip_retrain
        """,
    )

    train = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        doc_md="""
        ### Train Model
        Full model training pipeline:
        1. Hyperparameter optimization with Optuna (50 trials)
        2. Train XGBoost model with best parameters
        3. Track experiments with MLflow
        4. Generate SHAP interpretability plots
        5. Save model to MLflow and locally
        """,
    )

    skip = EmptyOperator(
        task_id="skip_retrain",
        doc_md="""
        ### Skip Retraining
        Placeholder task when no drift is detected.
        Uses existing model for predictions.
        """,
    )

    predict = PythonOperator(
        task_id="generate_predictions",
        python_callable=generate_predictions,
        trigger_rule=TriggerRule.NONE_FAILED,  # Execute even if one branch was skipped
        doc_md="""
        ### Generate Predictions
        Generates predictions for the NEXT WEEK after the most recent data:
        - Loads best model (from MLflow or local)
        - Gets latest week from historical data
        - Creates customer-product universe for week N+1
        - Applies feature engineering pipeline
        - Generates predictions with probabilities
        - Saves results
        
        This follows the requirement: "predict for the week following the most recent in the data"
        """,
    )

    end = EmptyOperator(
        task_id="end",
        trigger_rule=TriggerRule.NONE_FAILED,
        doc_md="""
        ### Pipeline End
        Marks successful completion of the pipeline.
        """,
    )

    # ========================================================================
    # TASK DEPENDENCIES
    # ========================================================================

    # Linear flow until branching
    start >> extract >> preprocess >> split >> drift_detection >> branch

    # Branching: retrain or skip
    branch >> [train, skip]

    # Both branches converge to prediction
    [train, skip] >> predict >> end


# ============================================================================
# DAG DOCUMENTATION
# ============================================================================

dag.doc_md = """
# SodAI Drinks Prediction Pipeline

## Overview
This DAG implements a complete machine learning pipeline for predicting customer purchases
of drinks products for the next week. It includes automated drift detection and conditional
model retraining.

## Features
- **Data Extraction**: Validates and loads raw data
- **Preprocessing**: Cleans and transforms data, creates customer-product universe
- **Drift Detection**: Statistical tests to detect distribution changes
- **Conditional Retraining**: Only retrains when drift is detected
- **Hyperparameter Optimization**: Uses Optuna for automated tuning
- **Experiment Tracking**: MLflow integration for reproducibility
- **Interpretability**: SHAP values for model explanation
- **Predictions**: Generates next week forecasts

## Data Flow
```
Raw Data → Preprocessing → Train/Val/Test Split → Drift Detection
                                                         ↓
                                            ┌─── Drift? ───┐
                                            ↓              ↓
                                        Retrain         Skip
                                            ↓              ↓
                                            └─→ Predict ←─┘
```

## Configuration
- **Training ratio**: 80% train, 20% validation (no test set)
- **Optuna trials**: 50
- **Model**: XGBoost with class balancing
- **Primary metric**: Recall (detect purchases)
- **Drift threshold**: 30% of features showing drift
- **Predictions**: For week N+1 (where N = latest week in data)

## Outputs
- **Processed data**: `data/processed/`
- **Drift reports**: `drift_reports/`
- **Models**: `models/` + MLflow
- **Predictions**: `predictions/`

## MLflow
All experiments are tracked in MLflow. Access the UI with:
```bash
mlflow ui --backend-store-uri file:///path/to/mlruns
```

## Execution
Trigger manually from Airflow UI or set a schedule (e.g., `@weekly`).

## Team
Free Riders - MDS7202 Laboratory Project
"""
