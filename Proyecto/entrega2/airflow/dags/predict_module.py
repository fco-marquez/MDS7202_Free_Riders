"""
Prediction Module
=================
Generates predictions for the next week using the best trained model.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib
from typing import Optional, Tuple
import mlflow
from mlflow_config import load_model_from_mlflow, get_experiment_name, setup_mlflow


def load_best_model(
    experiment_name: str = None,
    fallback_model_path: str = None
):
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
    if 'week' not in df.columns:
        raise ValueError("Data must contain 'week' column")

    latest_week = df['week'].max()
    print(f"Latest week in data: {latest_week}")
    return latest_week


def create_next_week_universe(
    customers_df: pd.DataFrame,
    products_df: pd.DataFrame,
    latest_week: int
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

    # Create cartesian product
    customers_df['key'] = 1
    products_df['key'] = 1

    universe = customers_df.merge(products_df, on='key').drop(columns=['key'])
    universe['week'] = next_week

    print(f"Universe size: {len(universe):,} customer-product pairs")
    print(f"Unique customers: {universe['customer_id'].nunique():,}")
    print(f"Unique products: {universe['product_id'].nunique():,}")

    print("=" * 60)
    return universe


def merge_historical_data_for_features(
    universe_df: pd.DataFrame,
    historical_df: pd.DataFrame
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
        if col not in universe_df.columns and col != 'bought':
            # Fill with placeholder values if column missing
            universe_df[col] = 0

    # Add bought column (placeholder, will be predicted)
    universe_df['bought'] = 0

    # Combine historical + universe
    combined = pd.concat([historical_df, universe_df], ignore_index=True)

    # Sort by customer_id, product_id, week to ensure proper time ordering
    combined = combined.sort_values(['customer_id', 'product_id', 'week']).reset_index(drop=True)

    print(f"Combined data size: {len(combined):,} rows")
    print(f"Historical data: {len(historical_df):,} rows")
    print(f"Universe (next week): {len(universe_df):,} rows")

    print("=" * 60)
    return combined


def generate_predictions(
    model,
    X_pred: pd.DataFrame,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate predictions using the trained model.

    Parameters
    ----------
    model
        Trained model/pipeline
    X_pred : pd.DataFrame
        Features for prediction
    output_path : str, optional
        Path to save predictions

    Returns
    -------
    pd.DataFrame
        Predictions with probabilities
    """
    print("\n" + "=" * 60)
    print("GENERATING PREDICTIONS")
    print("=" * 60)

    print(f"Generating predictions for {len(X_pred):,} samples...")

    # Make predictions
    y_pred = model.predict(X_pred)
    y_pred_proba = model.predict_proba(X_pred)[:, 1]

    # Create results dataframe
    predictions = pd.DataFrame({
        'customer_id': X_pred['customer_id'],
        'product_id': X_pred['product_id'],
        'week': X_pred['week'],
        'prediction': y_pred,
        'probability': y_pred_proba
    })

    # Summary statistics
    print("\nPrediction Summary:")
    print(f"Total predictions: {len(predictions):,}")
    print(f"Predicted purchases (1): {(predictions['prediction'] == 1).sum():,} ({(predictions['prediction'] == 1).sum() / len(predictions) * 100:.2f}%)")
    print(f"Predicted non-purchases (0): {(predictions['prediction'] == 0).sum():,} ({(predictions['prediction'] == 0).sum() / len(predictions) * 100:.2f}%)")
    print(f"\nProbability statistics:")
    print(predictions['probability'].describe())

    # Top predictions
    print("\nTop 10 most likely purchases:")
    top_predictions = predictions.nlargest(10, 'probability')[['customer_id', 'product_id', 'probability']]
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
    output_predictions_path: str = None
) -> pd.DataFrame:
    """
    Main function to run complete prediction pipeline (for Airflow task).

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
        experiment_name=model_experiment_name,
        fallback_model_path=model_fallback_path
    )

    # Load data
    print("\nLoading data files...")
    customers_df = pd.read_parquet(customers_path)
    products_df = pd.read_parquet(products_path)
    historical_df = pd.read_parquet(historical_data_path)

    print(f"Customers: {len(customers_df):,}")
    print(f"Products: {len(products_df):,}")
    print(f"Historical data: {len(historical_df):,}")

    # Get latest week
    latest_week = get_latest_week(historical_df)

    # Create next week universe
    next_week_universe = create_next_week_universe(
        customers_df,
        products_df,
        latest_week
    )

    # For prediction, we don't need to merge historical data if the pipeline
    # handles feature engineering internally. However, if features depend on
    # historical data (like recency, frequency), we need to provide that context.

    # Since our FeatureEngineer needs historical context, we'll use the next week universe directly
    # The pipeline's FeatureEngineer will handle the feature creation

    # Keep only necessary columns for prediction
    X_pred = next_week_universe.copy()

    # Generate predictions
    predictions = generate_predictions(
        model,
        X_pred,
        output_path=output_predictions_path
    )

    print("\n" + "=" * 80)
    print("PREDICTION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    return predictions


def create_prediction_summary_report(
    predictions: pd.DataFrame,
    output_path: str = None
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
    customer_summary = predictions.groupby('customer_id').agg({
        'prediction': 'sum',
        'probability': 'mean'
    }).reset_index()
    customer_summary.columns = ['customer_id', 'predicted_purchases', 'avg_probability']

    # Summary by product
    product_summary = predictions.groupby('product_id').agg({
        'prediction': 'sum',
        'probability': 'mean'
    }).reset_index()
    product_summary.columns = ['product_id', 'predicted_purchases', 'avg_probability']

    # Top customers (most likely to buy)
    top_customers = customer_summary.nlargest(20, 'predicted_purchases')

    # Top products (most likely to be purchased)
    top_products = product_summary.nlargest(20, 'predicted_purchases')

    print("\nTop 10 Customers (most predicted purchases):")
    print(top_customers.head(10).to_string(index=False))

    print("\nTop 10 Products (most predicted purchases):")
    print(top_products.head(10).to_string(index=False))

    # Save summary
    if output_path:
        summary = {
            'customer_summary': customer_summary,
            'product_summary': product_summary,
            'top_customers': top_customers,
            'top_products': top_products
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            customer_summary.to_excel(writer, sheet_name='Customer_Summary', index=False)
            product_summary.to_excel(writer, sheet_name='Product_Summary', index=False)
            top_customers.to_excel(writer, sheet_name='Top_Customers', index=False)
            top_products.to_excel(writer, sheet_name='Top_Products', index=False)

        print(f"\nSummary report saved to: {output_path}")

    return {
        'customer_summary': customer_summary,
        'product_summary': product_summary
    }
