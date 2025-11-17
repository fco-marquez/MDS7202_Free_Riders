"""
Recommender module for generating Top-N product recommendations.
Uses the trained model to recommend products with highest purchase probability.
"""
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Recommender:
    """Generates product recommendations for customers."""

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
            processed_path = self.data_dir / "processed" / "final_data.parquet"
            if processed_path.exists():
                self._historical_data = pd.read_parquet(processed_path)
                logger.info(f"Loaded {len(self._historical_data):,} historical records")
            else:
                raise FileNotFoundError(
                    f"Historical data not found at {processed_path}"
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

    def recommend(self, customer_id: int, top_n: int = 5, week: int = None) -> List[Dict[str, Any]]:
        """
        Generate top N product recommendations for a customer.

        Parameters
        ----------
        customer_id : int
            Customer identifier
        top_n : int
            Number of recommendations to return (default: 5)
        week : int, optional
            Week to predict (if None, uses next week after max historical week)

        Returns
        -------
        list of dict
            Top N recommended products with probabilities
        """
        logger.info(f"Generating top {top_n} recommendations for customer {customer_id}")

        # Load necessary data
        historical_data = self._load_historical_data()
        clientes_df = self._load_client_data()
        productos_df = self._load_product_data()

        # Validate customer exists
        if customer_id not in clientes_df['customer_id'].values:
            raise ValueError(f"Customer {customer_id} not found in database")

        # Get customer info
        customer = clientes_df[clientes_df['customer_id'] == customer_id].iloc[0]

        # Determine which week to predict
        max_week = historical_data['week'].max()
        if week is None:
            next_week = max_week + 1
            logger.info(f"Predicting for week {next_week} (next week after last historical week: {max_week})")
        else:
            next_week = week
            logger.info(f"Predicting for week {next_week} (user specified, last historical week: {max_week})")

        # Create predictions for all products
        predictions_df = self._create_all_predictions(
            customer_id, next_week, historical_data, customer, productos_df
        )

        # Make batch predictions
        try:
            probabilities = self.model.predict_proba(predictions_df)[:, 1]
            predictions_df['probability'] = probabilities

            # Sort by probability descending
            predictions_df = predictions_df.sort_values('probability', ascending=False)

            # Get top N
            top_recommendations = predictions_df.head(top_n)

            # Format results
            results = []
            for idx, row in top_recommendations.iterrows():
                results.append({
                    'rank': len(results) + 1,
                    'product_id': int(row['product_id']),
                    'probability': float(row['probability']),
                    'brand': row['brand'],
                    'category': row['category'],
                    'sub_category': row['sub_category'],
                    'segment': row['segment'],
                    'package': row['package'],
                    'size': float(row['size']),
                })

            logger.info(f"Generated {len(results)} recommendations")

            return results

        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise

    def _create_all_predictions(
        self,
        customer_id: int,
        week: int,
        historical_data: pd.DataFrame,
        customer: pd.Series,
        productos_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create prediction rows for all products for a given customer.
        """
        # Get customer's historical data
        customer_hist = historical_data[
            historical_data['customer_id'] == customer_id
        ].sort_values('week')

        total_customer_purchases = customer_hist['bought'].sum()

        # Prepare list to store rows
        rows = []

        for _, product in productos_df.iterrows():
            product_id = product['product_id']

            # Filter historical data for this customer-product pair
            hist_cp = customer_hist[customer_hist['product_id'] == product_id]

            # Calculate features
            if len(hist_cp) > 0:
                # Recency
                last_purchase_weeks = hist_cp[hist_cp['bought'] == 1]['week'].values
                if len(last_purchase_weeks) > 0:
                    recency = week - last_purchase_weeks[-1]
                else:
                    recency = week

                # Frequency: purchases in last 6 weeks
                recent_weeks = hist_cp[hist_cp['week'] > (week - 6)]
                frequency = recent_weeks['bought'].sum()

                # Customer-product share
                if total_customer_purchases > 0:
                    customer_product_share = frequency / total_customer_purchases
                else:
                    customer_product_share = 0

                # Trend
                very_recent = hist_cp[hist_cp['week'] > (week - 3)]['bought'].sum()
                past = frequency - very_recent
                trend = very_recent - past
            else:
                # New customer-product pair
                recency = week
                frequency = 0
                customer_product_share = 0
                trend = 0

            # Create row
            row = {
                'customer_id': customer_id,
                'product_id': product_id,
                'week': week,
                'customer_type': customer['customer_type'],
                'X': customer['X'],
                'Y': customer['Y'],
                'num_deliver_per_week': customer['num_deliver_per_week'],
                'num_visit_per_week': customer['num_visit_per_week'],
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
            }
            rows.append(row)

        # Create DataFrame
        df = pd.DataFrame(rows)

        # Ensure correct dtypes
        df = df.astype({
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

        return df
