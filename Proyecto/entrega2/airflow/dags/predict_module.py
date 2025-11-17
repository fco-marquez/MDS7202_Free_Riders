"""
Prediction Module
=================
Generates predictions for the next week using the best trained model.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow_config import get_experiment_name, load_model_from_mlflow, setup_mlflow


def load_best_model(experiment_name: str = None, fallback_model_path: str = None):
    """
    Load the best model from MLflow or fallback to local file.

    Parameters
    ----------
    experiment_name : str, optional
        MLflow experiment name
    fallback_model_path : str, optional
        Local model path as fallback

    Returns
    -------
    model
        Loaded model/pipeline
    """
    print("=" * 60)
    print("LOADING BEST MODEL")
    print("=" * 60)

    model = None

    # Try loading from MLflow first
    if experiment_name:
        try:
            print(f"Attempting to load model from MLflow experiment: {experiment_name}")
            model = load_model_from_mlflow(experiment_name)
            print("✓ Model loaded from MLflow successfully")
        except Exception as e:
            print(f"✗ Failed to load from MLflow: {e}")

    # Fallback to local file
    if model is None and fallback_model_path:
        try:
            print(f"Loading model from local file: {fallback_model_path}")
            model = joblib.load(fallback_model_path)
            print("✓ Model loaded from local file successfully")
        except Exception as e:
            print(f"✗ Failed to load from local file: {e}")
            raise

    if model is None:
        raise ValueError("Failed to load model from both MLflow and local file")

    print("=" * 60)
    return model


def get_latest_week(df: pd.DataFrame) -> int:
    """
    Get the latest week number from the data.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'week' column

    Returns
    -------
    int
        Latest week number
    """
    if "week" not in df.columns:
        raise ValueError("Data must contain 'week' column")

    latest_week = df["week"].max()
    print(f"Latest week in data: {latest_week}")
    return latest_week


def create_next_week_universe(
    customers_df: pd.DataFrame, products_df: pd.DataFrame, latest_week: int
) -> pd.DataFrame:
    """
    Create universe of all customer-product combinations for next week.

    Parameters
    ----------
    customers_df : pd.DataFrame
        Customer data
    products_df : pd.DataFrame
        Product data
    latest_week : int
        Latest week in historical data

    Returns
    -------
    pd.DataFrame
        Universe for next week prediction
    """
    print("\n" + "=" * 60)
    print("CREATING NEXT WEEK UNIVERSE")
    print("=" * 60)

    next_week = latest_week + 1
    print(f"Generating universe for week: {next_week}")

    print(f"\nCustomers columns: {customers_df.columns.tolist()}")
    print(f"Products columns: {products_df.columns.tolist()}")

    # Create cartesian product
    customers_copy = customers_df.copy()
    products_copy = products_df.copy()

    customers_copy["key"] = 1
    products_copy["key"] = 1

    universe = customers_copy.merge(products_copy, on="key", how="outer").drop(
        columns=["key"]
    )
    universe["week"] = next_week

    # Verify critical columns exist
    print(f"\nUniverse columns after merge: {universe.columns.tolist()}")

    # Check for NaN in geographic coordinates
    if "X" in universe.columns and "Y" in universe.columns:
        nan_x = universe["X"].isna().sum()
        nan_y = universe["Y"].isna().sum()
        if nan_x > 0 or nan_y > 0:
            print(f"WARNING: Found {nan_x} NaN values in 'X' and {nan_y} in 'Y'")
            print("Filling NaN geographic coordinates with median values...")
            universe["X"].fillna(universe["X"].median(), inplace=True)
            universe["Y"].fillna(universe["Y"].median(), inplace=True)
    else:
        print("ERROR: Columns 'X' and 'Y' not found in universe after merge!")
        print("This should not happen. Check data files.")
        raise ValueError(
            "Geographic coordinates 'X' and 'Y' are required but not found"
        )

    print(f"\nUniverse size: {len(universe):,} customer-product pairs")
    print(f"Unique customers: {universe['customer_id'].nunique():,}")
    print(f"Unique products: {universe['product_id'].nunique():,}")

    print("=" * 60)
    return universe


def merge_historical_data_for_features(
    universe_df: pd.DataFrame, historical_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge historical data with universe to enable feature engineering.

    This is needed because features like recency, frequency, etc. are calculated
    based on historical purchases.

    Parameters
    ----------
    universe_df : pd.DataFrame
        Universe for next week
    historical_df : pd.DataFrame
        Historical transaction data with features

    Returns
    -------
    pd.DataFrame
        Combined data ready for feature engineering
    """
    print("\n" + "=" * 60)
    print("MERGING HISTORICAL DATA FOR FEATURE ENGINEERING")
    print("=" * 60)

    # We need to append the universe to historical data to calculate rolling features
    # The universe will have bought=0 (placeholder) since we don't know actual purchases yet

    # Ensure universe has the same columns as historical data
    for col in historical_df.columns:
        if col not in universe_df.columns and col != "bought":
            # Fill with placeholder values if column missing
            universe_df[col] = 0

    # Add bought column (placeholder, will be predicted)
    universe_df["bought"] = 0

    # Combine historical + universe
    combined = pd.concat([historical_df, universe_df], ignore_index=True)

    # Sort by customer_id, product_id, week to ensure proper time ordering
    combined = combined.sort_values(["customer_id", "product_id", "week"]).reset_index(
        drop=True
    )

    print(f"Combined data size: {len(combined):,} rows")
    print(f"Historical data: {len(historical_df):,} rows")
    print(f"Universe (next week): {len(universe_df):,} rows")

    print("=" * 60)
    return combined


def generate_predictions(
    model, X_pred: pd.DataFrame, output_path: str = None, batch_size: int = 50000
) -> pd.DataFrame:
    """
    Generate predictions using the trained model with batch processing.

    Parameters
    ----------
    model
        Trained model/pipeline
    X_pred : pd.DataFrame
        Features for prediction
    output_path : str, optional
        Path to save predictions
    batch_size : int, optional
        Number of samples to process at once (default: 50000)

    Returns
    -------
    pd.DataFrame
        Predictions with probabilities
    """
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS (BATCH MODE)")
    print("=" * 60)

    total_samples = len(X_pred)
    print(f"Total samples: {total_samples:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Number of batches: {(total_samples + batch_size - 1) // batch_size}")

    # Store predictions
    all_predictions = []
    all_probabilities = []

    # Process in batches to avoid memory issues
    for i in range(0, total_samples, batch_size):
        batch_end = min(i + batch_size, total_samples)
        batch_num = (i // batch_size) + 1
        total_batches = (total_samples + batch_size - 1) // batch_size

        print(f"\nProcessing batch {batch_num}/{total_batches} (rows {i:,} to {batch_end:,})...")

        X_batch = X_pred.iloc[i:batch_end]

        # Make predictions for this batch
        y_pred_batch = model.predict(X_batch)
        y_pred_proba_batch = model.predict_proba(X_batch)[:, 1]

        all_predictions.extend(y_pred_batch)
        all_probabilities.extend(y_pred_proba_batch)

        print(f"  ✓ Batch {batch_num} completed")

    print("\n✓ All batches processed successfully")

    # Combine all predictions
    y_pred = np.array(all_predictions)
    y_pred_proba = np.array(all_probabilities)

    # Create results dataframe
    predictions = pd.DataFrame(
        {
            "customer_id": X_pred["customer_id"].values,
            "product_id": X_pred["product_id"].values,
            "week": X_pred["week"].values,
            "prediction": y_pred,
            "probability": y_pred_proba,
        }
    )

    # Summary statistics
    print("\nPrediction Summary:")
    print(f"Total predictions: {len(predictions):,}")
    print(
        f"Predicted purchases (1): {(predictions['prediction'] == 1).sum():,} ({(predictions['prediction'] == 1).sum() / len(predictions) * 100:.2f}%)"
    )
    print(
        f"Predicted non-purchases (0): {(predictions['prediction'] == 0).sum():,} ({(predictions['prediction'] == 0).sum() / len(predictions) * 100:.2f}%)"
    )
    print(f"\nProbability statistics:")
    print(predictions["probability"].describe())

    # Top predictions
    print("\nTop 10 most likely purchases:")
    top_predictions = predictions.nlargest(10, "probability")[
        ["customer_id", "product_id", "probability"]
    ]
    print(top_predictions.to_string(index=False))

    # Save predictions
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

    print("=" * 60)
    return predictions


def run_prediction_pipeline(
    customers_path: str,
    products_path: str,
    historical_data_path: str,
    model_experiment_name: str = None,
    model_fallback_path: str = None,
    output_predictions_path: str = None,
    max_customers: int = None,
    batch_size: int = 50000,
) -> pd.DataFrame:
    """
    Main function to run complete prediction pipeline (for Airflow task).

    Creates universe for the NEXT WEEK after the most recent week in historical data,
    as per project requirements.

    Parameters
    ----------
    customers_path : str
        Path to customers parquet
    products_path : str
        Path to products parquet
    historical_data_path : str
        Path to historical data with features
    model_experiment_name : str, optional
        MLflow experiment name for model
    model_fallback_path : str, optional
        Local model path as fallback
    output_predictions_path : str, optional
        Path to save predictions
    max_customers : int, optional
        Limit universe to N customers (to reduce memory usage). If None, uses all customers.
    batch_size : int, optional
        Batch size for predictions (default: 50000)

    Returns
    -------
    pd.DataFrame
        Predictions
    """
    print("\n" + "=" * 80)
    print("PREDICTION PIPELINE FOR NEXT WEEK")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Load model
    model = load_best_model(
        experiment_name=model_experiment_name, fallback_model_path=model_fallback_path
    )

    # Load data
    print("\nLoading data files...")
    customers_df = pd.read_parquet(customers_path)
    products_df = pd.read_parquet(products_path)
    historical_df = pd.read_parquet(historical_data_path)

    print(f"Customers (total): {len(customers_df):,}")
    print(f"Products (total): {len(products_df):,}")
    print(f"Historical data: {len(historical_df):,}")

    # Limit customers if specified (to reduce memory usage)
    if max_customers is not None and len(customers_df) > max_customers:
        print(f"\n⚠️  Limiting to {max_customers:,} customers (out of {len(customers_df):,})")
        print("   This is to reduce memory usage. Remove max_customers param for full universe.")
        customers_df = customers_df.head(max_customers)
        print(f"   Using customers: {customers_df['customer_id'].min()} to {customers_df['customer_id'].max()}")

    print(f"\nCustomers (for prediction): {len(customers_df):,}")
    print(f"Products (for prediction): {len(products_df):,}")
    print(f"Expected universe size: ~{len(customers_df) * len(products_df):,} pairs")

    # Get latest week from historical data
    latest_week = get_latest_week(historical_df)

    # Create universe for NEXT WEEK (latest_week + 1)
    # This follows the requirement: "predict for the week following the most recent in the data"
    print(f"\n📅 Creating universe for prediction week: {latest_week + 1}")
    X_pred = create_next_week_universe(customers_df, products_df, latest_week)

    # Generate predictions with batch processing
    predictions = generate_predictions(
        model, X_pred, output_path=output_predictions_path, batch_size=batch_size
    )

    print("\n" + "=" * 80)
    print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return predictions


def create_prediction_summary_report(
    predictions: pd.DataFrame, output_path: str = None
) -> pd.DataFrame:
    """
    Create a summary report of predictions.

    Parameters
    ----------
    predictions : pd.DataFrame
        Prediction results
    output_path : str, optional
        Path to save summary report

    Returns
    -------
    pd.DataFrame
        Summary report
    """
    # Summary by customer
    customer_summary = (
        predictions.groupby("customer_id")
        .agg({"prediction": "sum", "probability": "mean"})
        .reset_index()
    )
    customer_summary.columns = ["customer_id", "predicted_purchases", "avg_probability"]

    # Summary by product
    product_summary = (
        predictions.groupby("product_id")
        .agg({"prediction": "sum", "probability": "mean"})
        .reset_index()
    )
    product_summary.columns = ["product_id", "predicted_purchases", "avg_probability"]

    # Top customers (most likely to buy)
    top_customers = customer_summary.nlargest(20, "predicted_purchases")

    # Top products (most likely to be purchased)
    top_products = product_summary.nlargest(20, "predicted_purchases")

    print("\nTop 10 Customers (most predicted purchases):")
    print(top_customers.head(10).to_string(index=False))

    print("\nTop 10 Products (most predicted purchases):")
    print(top_products.head(10).to_string(index=False))

    # Save summary
    if output_path:
        summary = {
            "customer_summary": customer_summary,
            "product_summary": product_summary,
            "top_customers": top_customers,
            "top_products": top_products,
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            customer_summary.to_excel(
                writer, sheet_name="Customer_Summary", index=False
            )
            product_summary.to_excel(writer, sheet_name="Product_Summary", index=False)
            top_customers.to_excel(writer, sheet_name="Top_Customers", index=False)
            top_products.to_excel(writer, sheet_name="Top_Products", index=False)

        print(f"\nSummary report saved to: {output_path}")

    return {"customer_summary": customer_summary, "product_summary": product_summary}
