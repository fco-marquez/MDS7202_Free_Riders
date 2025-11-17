"""
Predictor module for making predictions using the trained model.
Handles data loading, feature engineering, and prediction generation.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """Handles prediction logic with automatic feature engineering."""

    def __init__(
        self,
        model,
        data_dir: str = "/data",
    ):
        self.model = model
        self.data_dir = Path(data_dir)

        # Cache for data
        self._historical_data = None
        self._clientes_df = None
        self._productos_df = None

    def _load_historical_data(self):
        """Load historical data for feature engineering."""
        if self._historical_data is None:
            logger.info("Loading historical data...")

            # Try to load processed data first
            processed_path = self.data_dir / "processed" / "final_data.parquet"
            if processed_path.exists():
                self._historical_data = pd.read_parquet(processed_path)
                logger.info(f"Loaded {len(self._historical_data):,} historical records")
            else:
                raise FileNotFoundError(
                    f"Historical data not found at {processed_path}. "
                    "Please ensure the Airflow pipeline has been executed."
                )

        return self._historical_data

    def _load_client_data(self):
        """Load client information."""
        if self._clientes_df is None:
            logger.info("Loading client data...")
            clientes_path = self.data_dir / "raw" / "clientes.parquet"
            if clientes_path.exists():
                self._clientes_df = pd.read_parquet(clientes_path)
                logger.info(f"Loaded {len(self._clientes_df):,} clients")
            else:
                raise FileNotFoundError(f"Client data not found at {clientes_path}")

        return self._clientes_df

    def _load_product_data(self):
        """Load product information."""
        if self._productos_df is None:
            logger.info("Loading product data...")
            productos_path = self.data_dir / "raw" / "productos.parquet"
            if productos_path.exists():
                self._productos_df = pd.read_parquet(productos_path)
                logger.info(f"Loaded {len(self._productos_df):,} products")
            else:
                raise FileNotFoundError(f"Product data not found at {productos_path}")

        return self._productos_df

    def predict(self, customer_id: int, product_id: int) -> Dict[str, Any]:
        """
        Make a prediction for a customer-product pair.

        Parameters
        ----------
        customer_id : int
            Customer identifier
        product_id : int
            Product identifier

        Returns
        -------
        dict
            Prediction result with probability and additional info
        """
        logger.info(f"Making prediction for customer={customer_id}, product={product_id}")

        # Load necessary data
        historical_data = self._load_historical_data()
        clientes_df = self._load_client_data()
        productos_df = self._load_product_data()

        # Validate customer and product exist
        if customer_id not in clientes_df['customer_id'].values:
            raise ValueError(f"Customer {customer_id} not found in database")

        if product_id not in productos_df['product_id'].values:
            raise ValueError(f"Product {product_id} not found in database")

        # Get next week to predict
        max_week = historical_data['week'].max()
        next_week = max_week + 1

        logger.info(f"Predicting for week {next_week} (last historical week: {max_week})")

        # Create prediction row
        prediction_row = self._create_prediction_row(
            customer_id, product_id, next_week,
            historical_data, clientes_df, productos_df
        )

        # Make prediction
        try:
            probability = self.model.predict_proba(prediction_row)[0, 1]
            prediction = int(probability >= 0.5)

            # Get customer and product info for response
            customer_info = clientes_df[clientes_df['customer_id'] == customer_id].iloc[0]
            product_info = productos_df[productos_df['product_id'] == product_id].iloc[0]

            result = {
                'customer_id': int(customer_id),
                'product_id': int(product_id),
                'prediction': prediction,
                'probability': float(probability),
                'week': int(next_week),
                'customer_type': customer_info['customer_type'],
                'product_brand': product_info['brand'],
                'product_category': product_info['category'],
            }

            logger.info(f"Prediction: {prediction} (probability: {probability:.4f})")

            return result

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise

    def _create_prediction_row(
        self,
        customer_id: int,
        product_id: int,
        week: int,
        historical_data: pd.DataFrame,
        clientes_df: pd.DataFrame,
        productos_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a single row with all features for prediction.

        This replicates the feature engineering from the training pipeline.
        """
        # Get customer and product info
        customer = clientes_df[clientes_df['customer_id'] == customer_id].iloc[0]
        product = productos_df[productos_df['product_id'] == product_id].iloc[0]

        # Filter historical data for this customer-product pair
        hist_cp = historical_data[
            (historical_data['customer_id'] == customer_id) &
            (historical_data['product_id'] == product_id)
        ].sort_values('week')

        # Calculate features based on historical data
        if len(hist_cp) > 0:
            # Recency: weeks since last purchase
            last_purchase_weeks = hist_cp[hist_cp['bought'] == 1]['week'].values
            if len(last_purchase_weeks) > 0:
                recency = week - last_purchase_weeks[-1]
            else:
                recency = week  # Never purchased

            # Frequency: purchases in last 6 weeks
            recent_weeks = hist_cp[hist_cp['week'] > (week - 6)]
            frequency = recent_weeks['bought'].sum()

            # Total purchases by customer
            customer_hist = historical_data[historical_data['customer_id'] == customer_id]
            total_purchases = customer_hist['bought'].sum()

            # Customer-product share
            if total_purchases > 0:
                customer_product_share = frequency / total_purchases
            else:
                customer_product_share = 0

            # Trend: recent vs past
            very_recent = hist_cp[hist_cp['week'] > (week - 3)]['bought'].sum()
            past = frequency - very_recent
            trend = very_recent - past
        else:
            # New customer-product pair
            recency = week
            frequency = 0
            customer_product_share = 0
            trend = 0

        # Create row with all features
        row = pd.DataFrame([{
            'customer_id': customer_id,
            'product_id': product_id,
            'week': week,
            'customer_type': customer['customer_type'],
            'X': customer['X'],
            'Y': customer['Y'],
            'num_deliver_per_week': customer['num_deliver_per_week'],
            'zone_id': customer['zone_id'],
            'region_id': customer['region_id'],
            'brand': product['brand'],
            'category': product['category'],
            'sub_category': product['sub_category'],
            'segment': product['segment'],
            'package': product['package'],
            'size': product['size'],
            'recency': recency,
            'frequency': frequency,
            'customer_product_share': customer_product_share,
            'trend': trend,
        }])

        # Ensure correct dtypes (matching training data)
        row = row.astype({
            'customer_id': 'int32',
            'product_id': 'int32',
            'week': 'int32',
            'customer_type': 'object',
            'X': 'float32',
            'Y': 'float32',
            'num_deliver_per_week': 'int32',
            'zone_id': 'int32',
            'region_id': 'int32',
            'brand': 'object',
            'category': 'object',
            'sub_category': 'object',
            'segment': 'object',
            'package': 'object',
            'size': 'float32',
            'recency': 'float32',
            'frequency': 'float32',
            'customer_product_share': 'float32',
            'trend': 'float32',
        })

        return row

    def get_available_customers(self, limit: int = 100):
        """Get list of available customer IDs."""
        clientes_df = self._load_client_data()
        return clientes_df['customer_id'].head(limit).tolist()

    def get_available_products(self, limit: int = 100):
        """Get list of available product IDs."""
        productos_df = self._load_product_data()
        return productos_df['product_id'].head(limit).tolist()
