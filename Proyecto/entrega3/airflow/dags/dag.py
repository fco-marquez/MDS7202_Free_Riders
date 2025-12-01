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
import shutil
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule
from drift_detector import run_drift_detection

# Import local modules
from load_and_preprocess import run_preprocessing_pipeline
from pipeline import run_data_splitting
from predict_module import run_prediction_pipeline
from train_module import run_full_training

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base paths
BASE_DIR = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow"))
INCOMING_DATA_DIR = BASE_DIR / "data" / "incoming"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
STATIC_DATA_DIR = BASE_DIR / "data" / "static"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
PREDICTIONS_DIR = BASE_DIR / "predictions"
DRIFT_REPORTS_DIR = BASE_DIR / "drift_reports"
MODELS_DIR = BASE_DIR / "models"

# Create directories
for directory in [INCOMING_DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, DRIFT_REPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File paths
CURRENT_DATA_PATH = PROCESSED_DATA_DIR / "current_data.parquet"
FINAL_DATA_PATH = PROCESSED_DATA_DIR / "final_data.parquet"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train_data.parquet"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val_data.parquet"
CUSTOMERS_PATH = STATIC_DATA_DIR / "clientes.parquet"
PRODUCTS_PATH = STATIC_DATA_DIR / "productos.parquet"
DRIFT_REPORT_PATH = DRIFT_REPORTS_DIR / "drift_report_{execution_date}.json"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
PREDICTIONS_PATH = PREDICTIONS_DIR / "predictions_{execution_date}.parquet"

# CodaLab CSV output directory (inside predictions dir, same as parquet)
CODALAB_CSV_DIR = PREDICTIONS_DIR
CODALAB_CSV_PATH = CODALAB_CSV_DIR / "prediccion_{execution_date}.csv"

# Training configuration from environment variables
N_OPTUNA_TRIALS = int(os.getenv("N_OPTUNA_TRIALS", "50"))
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "sodai_drinks_prediction")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.3"))

# Prediction threshold for CodaLab (probability >= this value = predicted purchase)
PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.112"))


# ============================================================================
# TASK FUNCTIONS
# ============================================================================


def ingest_and_preprocess(**context):
    """
    Ingest raw data and preprocess in a single step.
    Automatically detects and moves new files from incoming/ to raw/.
    """
    print("=" * 60)
    print("INGESTING AND PREPROCESSING DATA")
    print("=" * 60)

    # Ensure directories exist
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INCOMING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    new_data_arrived = False

    # Check for new files in incoming directory
    incoming_files = list(INCOMING_DATA_DIR.glob("*.parquet"))

    if incoming_files:
        print(f"\n📦 Found {len(incoming_files)} new batch file(s) in incoming directory:")
        for src in incoming_files:
            try:
                dest = RAW_DATA_DIR / src.name
                print(f"  - Moving {src.name} to raw data directory...")
                shutil.move(str(src), str(dest))
                print(f"    ✓ Successfully moved to {RAW_DATA_DIR}")
                new_data_arrived = True
            except Exception as e:
                print(f"    ✗ Error moving {src.name}: {e}")
    else:
        print("\nℹ️  No new files in incoming directory")

    # Legacy support: Get configuration from dag_run.conf
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", {}) or {}
    new_parquet_paths = conf.get("new_parquet_paths")  # List of new fragment paths

    # Copy new parquet fragments if provided via conf (manual trigger)
    if new_parquet_paths:
        print(f"\n📦 New parquet fragments from config: {new_parquet_paths}")
        for p in new_parquet_paths:
            try:
                src = Path(p)
                if not src.exists():
                    print(f"  ✗ File not found: {p}")
                    continue
                dest = RAW_DATA_DIR / src.name
                shutil.copyfile(src, dest)
                print(f"  ✓ Copied {src.name} to {RAW_DATA_DIR}")
                new_data_arrived = True
            except Exception as e:
                print(f"  ✗ Error copying {p}: {e}")

    # Verify raw data exists
    raw_files = list(RAW_DATA_DIR.glob("*.parquet"))
    print(f"\nFound {len(raw_files)} total parquet file(s) in {RAW_DATA_DIR}")

    if len(raw_files) == 0:
        raise FileNotFoundError(f"No raw data files in {RAW_DATA_DIR}!")

    # Determine output path based on whether this is first run
    if not CURRENT_DATA_PATH.exists():
        # First run: create current_data.parquet
        output_path = CURRENT_DATA_PATH
        print("\nNew first run detected - creating initial dataset")
    else:
        # Subsequent run: create final_data.parquet for comparison
        output_path = FINAL_DATA_PATH
        print("\nUpdating dataset with new data")

    # Run preprocessing pipeline

    run_preprocessing_pipeline(
        raw_data_folder=str(RAW_DATA_DIR),
        output_data_path=str(output_path),
        static_data_folder=str(STATIC_DATA_DIR),
    )

    # Push flags to XCom
    context["task_instance"].xcom_push(key="new_data_arrived", value=new_data_arrived)
    context["task_instance"].xcom_push(key="output_path", value=str(output_path))

    print(f"\n✓ Data saved to: {output_path}")
    print("=" * 60)


def split_and_prepare_training(**context):
    """
    Split data into train/validation sets.
    Uses CURRENT_DATA_PATH (reference data after potential update).
    """
    print("=" * 60)
    print("SPLITTING DATA FOR TRAINING")
    print("=" * 60)

    # Use CURRENT_DATA_PATH as the source (updated in decide_retrain if needed)
    if not CURRENT_DATA_PATH.exists():
        raise FileNotFoundError(f"Current data not found: {CURRENT_DATA_PATH}")

    run_data_splitting(
        input_data_path=str(CURRENT_DATA_PATH), output_dir=str(PROCESSED_DATA_DIR)
    )

    print(f"\n✓ Train/Val data saved to {PROCESSED_DATA_DIR}")
    print("=" * 60)


def detect_drift_and_decide(**context):
    """
    Detect drift and decide whether to retrain.
    Returns task_id for branching: 'split_and_train' or 'skip_retrain'
    """
    ti = context["task_instance"]
    execution_date = context["ds"]

    print("=" * 60)
    print("DRIFT DETECTION & RETRAINING DECISION")
    print("=" * 60)

    # Check if model exists
    model_exists = MODEL_PATH.exists()
    print(f"Model exists: {model_exists}")

    # If no model, must train (first run)
    if not model_exists:
        print("\nNo existing model → MUST TRAIN")
        print("=" * 60)
        return "split_and_train"

    # Check if new data arrived in this run
    new_data_arrived = ti.xcom_pull(
        task_ids="ingest_and_preprocess", key="new_data_arrived"
    )
    print(f"New data arrived: {new_data_arrived}")

    # If no new data, use existing model
    if not new_data_arrived:
        print("\nNo new transactions → Use existing model")
        print("=" * 60)
        return "skip_retrain"

    # New data arrived: check for drift
    if not FINAL_DATA_PATH.exists():
        print("\nFinal data not found → RETRAIN")
        # Update reference data
        shutil.copyfile(CURRENT_DATA_PATH, CURRENT_DATA_PATH)
        print("=" * 60)
        return "split_and_train"

    # Run drift detection
    print("\n🔍 Running drift detection...")
    drift_report_path = str(DRIFT_REPORT_PATH).format(execution_date=execution_date)

    drift_detected = run_drift_detection(
        reference_data_path=str(CURRENT_DATA_PATH),
        current_data_path=str(FINAL_DATA_PATH),
        output_report_path=drift_report_path,
    )

    print(f"Drift detected: {drift_detected}")

    if drift_detected:
        print("\nDRIFT DETECTED → RETRAIN")
        # Update reference data for next run
        shutil.copyfile(FINAL_DATA_PATH, CURRENT_DATA_PATH)
        print(f"Updated reference data: {CURRENT_DATA_PATH}")
        print("=" * 60)
        return "split_and_train"
    else:
        print("\nNo significant drift → Use existing model")
        print("=" * 60)
        return "skip_retrain"


def split_and_train(**context):
    """
    Split data and train model in a single optimized step.
    """
    print("=" * 60)
    print("SPLITTING DATA & TRAINING MODEL")
    print("=" * 60)

    # Step 1: Split data
    print("\nStep 1: Splitting data...")
    from pipeline import run_data_splitting

    run_data_splitting(
        input_data_path=str(CURRENT_DATA_PATH), output_dir=str(PROCESSED_DATA_DIR)
    )
    print(f"Train/Val splits saved to {PROCESSED_DATA_DIR}")

    # Step 2: Train model
    print("\nStep 2: Training model...")

    pipeline = run_full_training(
        train_data_path=str(TRAIN_DATA_PATH),
        val_data_path=str(VAL_DATA_PATH),
        n_trials=N_OPTUNA_TRIALS,
        output_model_path=str(MODEL_PATH),
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print("=" * 60)


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
        historical_data_path=CURRENT_DATA_PATH,
        model_experiment_name=MLFLOW_EXPERIMENT,
        model_fallback_path=str(MODEL_PATH),
        output_predictions_path=predictions_path,
        max_customers=None,  # Limit to 100 customers (~97K predictions instead of 1.5M)
        batch_size=20000,  # Process 20K rows at a time
    )

    print(f"\nPredictions saved to: {predictions_path}")
    print(f"Generated {len(predictions):,} predictions")


def convert_predictions_to_csv(**context):
    """
    Convert predictions from parquet to CSV format for CodaLab submission.

    Filters predictions by probability threshold and outputs only customer_id, product_id
    for pairs where the model predicts a purchase will occur.

    Output format:
        customer_id,product_id
        123,456
        789,012
        ...
    """
    import pandas as pd

    execution_date = context["ds"]
    predictions_parquet_path = str(PREDICTIONS_PATH).format(execution_date=execution_date)
    csv_output_path = str(CODALAB_CSV_PATH).format(execution_date=execution_date)

    print("=" * 60)
    print("CONVERTING PREDICTIONS TO CODALAB CSV FORMAT")
    print("=" * 60)

    # Load predictions
    print(f"\nLoading predictions from: {predictions_parquet_path}")
    preds = pd.read_parquet(predictions_parquet_path)

    print(f"Total predictions: {len(preds):,}")
    print(f"Columns: {preds.columns.tolist()}")

    # Show probability distribution
    print(f"\nProbability statistics:")
    print(f"  Min: {preds['probability'].min():.4f}")
    print(f"  Max: {preds['probability'].max():.4f}")
    print(f"  Mean: {preds['probability'].mean():.4f}")
    print(f"  Median: {preds['probability'].median():.4f}")

    # Filter by threshold
    print(f"\nApplying threshold: {PREDICTION_THRESHOLD}")
    positive_preds = preds[preds['probability'] >= PREDICTION_THRESHOLD][['customer_id', 'product_id']]

    # Ensure integer types for IDs
    positive_preds['customer_id'] = positive_preds['customer_id'].astype(int)
    positive_preds['product_id'] = positive_preds['product_id'].astype(int)

    # Save to CSV
    positive_preds.to_csv(csv_output_path, index=False)

    # Summary
    pct = 100 * len(positive_preds) / len(preds) if len(preds) > 0 else 0
    print(f"\n✓ CSV saved to: {csv_output_path}")
    print(f"  Predicted purchases: {len(positive_preds):,} ({pct:.1f}% of universe)")
    print(f"  File size: {Path(csv_output_path).stat().st_size:,} bytes")

    # Show sample
    print(f"\nSample (first 5 rows):")
    print(positive_preds.head().to_string(index=False))

    print("=" * 60)


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
    schedule_interval="@daily",  # Runs daily, FileSensor waits for new batch files
    start_date=dt.datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,  # Only one run at a time
    tags=["ml", "prediction", "drinks", "mlflow", "optuna", "automated"],
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

    wait_for_data = FileSensor(
        task_id="wait_for_new_batch",
        filepath="data/incoming/*.parquet",
        fs_conn_id="fs_default",
        poke_interval=30,  # Check every 30 seconds
        timeout=60 * 60 * 24 * 7,  # Wait up to 7 days
        mode="reschedule",  # Free up worker slot while waiting
        doc_md="""
        ### Wait for New Batch Data
        Monitors the `incoming/` directory for new .parquet batch files.
        - Checks every 30 seconds
        - Uses reschedule mode to not block the scheduler
        - Times out after 7 days
        - Automatically triggers pipeline when new data arrives
        """,
    )

    ingest_preprocess = PythonOperator(
        task_id="ingest_and_preprocess",
        python_callable=ingest_and_preprocess,
        doc_md="""
        ### Ingest & Preprocess Data
        - Automatically moves files from incoming/ to raw/
        - Loads raw data (customers, products, transactions)
        - Cleans transactions (remove duplicates, filter invalid items)
        - Optimizes datatypes
        - Creates week and objective variables
        - Generates customer-product-week universe
        """,
    )

    branch = BranchPythonOperator(
        task_id="detect_drift_and_decide",
        python_callable=detect_drift_and_decide,
        doc_md="""
        ### Drift Detection & Branching
        Intelligent decision logic:
        1. No model exists → TRAIN (first run)
        2. No new data → SKIP (use existing model)
        3. New data arrived → Check drift:
           - Drift detected → RETRAIN
           - No drift → SKIP
        """,
    )

    split_train_task = PythonOperator(
        task_id="split_and_train",
        python_callable=split_and_train,
        doc_md="""
        ### Split Data & Train Model
        - Splits data into train (80%) / val (20%)
        - Runs hyperparameter optimization with Optuna
        - Trains XGBoost model
        - Logs to MLflow
        - Saves best model
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
        - Saves results as parquet

        This follows the requirement: "predict for the week following the most recent in the data"
        """,
    )

    export_csv = PythonOperator(
        task_id="export_codalab_csv",
        python_callable=convert_predictions_to_csv,
        doc_md="""
        ### Export CodaLab CSV
        Converts predictions from parquet to CSV format for CodaLab submission:
        - Loads predictions parquet file
        - Filters by probability threshold (default: 0.112)
        - Outputs only customer_id, product_id columns
        - Saves to predictions/ directory outside airflow

        **Output format:**
        ```
        customer_id,product_id
        123,456
        789,012
        ```

        The threshold can be configured via PREDICTION_THRESHOLD environment variable.
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

    # Simplified linear flow with intelligent branching
    start >> wait_for_data >> ingest_preprocess >> branch

    # Branch: retrain or skip
    branch >> [split_train_task, skip]

    # Both paths converge to prediction, then export CSV for CodaLab
    [split_train_task, skip] >> predict >> export_csv >> end


# ============================================================================
# DAG DOCUMENTATION
# ============================================================================

dag.doc_md = """
# SodAI Drinks Prediction Pipeline

## Overview
This DAG implements a complete machine learning pipeline for predicting customer purchases
of drinks products for the next week. It includes automated drift detection, conditional
model retraining, and CodaLab CSV export.

## Features
- **Data Extraction**: Validates and loads raw data
- **Preprocessing**: Cleans and transforms data, creates customer-product universe
- **Drift Detection**: Statistical tests to detect distribution changes
- **Conditional Retraining**: Only retrains when drift is detected
- **Hyperparameter Optimization**: Uses Optuna for automated tuning
- **Experiment Tracking**: MLflow integration for reproducibility
- **Interpretability**: SHAP values for model explanation
- **Predictions**: Generates next week forecasts
- **CodaLab Export**: Converts predictions to CSV format for competition submission

## Data Flow
```
Incoming/ → Wait for File → Move to Raw/ → Preprocessing → Drift Detection
                                                                  ↓
                                                          ┌─── Drift? ───┐
                                                          ↓              ↓
                                                      Retrain         Skip
                                                          ↓              ↓
                                                          └─→ Predict ←─┘
                                                                  ↓
                                                          Export CSV (CodaLab)
```

## Automated Processing
- **FileSensor** monitors `data/incoming/` for new .parquet batch files
- When new data arrives, it's automatically moved to `data/raw/`
- Pipeline processes the new batch with drift detection
- Predictions are automatically exported to CSV for CodaLab
- Runs daily to check for new files (configurable)

## Configuration
- **Training ratio**: 80% train, 20% validation (no test set)
- **Optuna trials**: 50
- **Model**: XGBoost with class balancing
- **Primary metric**: F1-score
- **Drift threshold**: 30% of features showing drift
- **Prediction threshold**: 0.112 (configurable via PREDICTION_THRESHOLD env var)
- **Predictions**: For week N+1 (where N = latest week in data)

## Outputs
- **Processed data**: `data/processed/`
- **Drift reports**: `drift_reports/`
- **Models**: `models/` + MLflow
- **Predictions (parquet)**: `airflow/predictions/`
- **Predictions (CSV for CodaLab)**: `predictions/` (root directory)

## CodaLab CSV Format
The exported CSV contains only customer-product pairs predicted as purchases:
```csv
customer_id,product_id
123,456
789,012
```

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
