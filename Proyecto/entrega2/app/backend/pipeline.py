import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

set_config(transform_output="pandas")


def load_final_data(file_path: str) -> pd.DataFrame:
    """Load the final processed data from a Parquet file."""
    df = pd.read_parquet(file_path)
    return df


def split_data(
    df: pd.DataFrame,
    output_dir: str = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.20,
):
    """
    Split the DataFrame into training and validation sets only.

    NOTE: We don't create a test set because predictions will be made
    for the NEXT WEEK after the most recent data, not on existing data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with 'week' column
    output_dir : str, optional
        Directory to save split data
    train_ratio : float
        Proportion of data for training (default 0.80)
    val_ratio : float
        Proportion of data for validation (default 0.20)

    Returns
    -------
    tuple
        (train_data, val_data)
    """
    # Ordenar datos por fecha para respetar temporalidad
    df_temporal = df.sort_values("week").copy()

    # Definir puntos de corte temporal
    train_end = df_temporal["week"].quantile(train_ratio)

    # Separar los conjuntos basados en tiempo
    train_data = df_temporal[df_temporal["week"] <= train_end]
    val_data = df_temporal[df_temporal["week"] > train_end]

    print("Distribución temporal de los conjuntos:")
    print(
        f"Train: {train_data['week'].min()} → {train_data['week'].max()} ({len(train_data)} registros)"
    )
    print(
        f"Val:   {val_data['week'].min()} → {val_data['week'].max()} ({len(val_data)} registros)"
    )
    print(
        "\n⚠ NO se crea conjunto de test - Las predicciones se harán para la semana siguiente"
    )

    # Save the data
    if output_dir is None:
        raise ValueError("output_dir must be specified")

    import os

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_data.parquet")
    val_path = os.path.join(output_dir, "val_data.parquet")

    train_data.to_parquet(train_path, index=False)
    val_data.to_parquet(val_path, index=False)

    print(f"\nSaved splits to: {output_dir}")

    return train_data, val_data


def create_advanced_features(X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
    X_new = X.copy()
    if y is not None:
        X_new["bought"] = y.values

    X_new = X_new.sort_values(by=["customer_id", "product_id", "week"])

    # Recency
    X_new["last_purchase_week"] = X_new.groupby(["customer_id", "product_id"])[
        "week"
    ].shift(1)
    X_new["recency"] = X_new["week"] - X_new["last_purchase_week"]
    X_new.fillna({"recency": X_new["recency"].max()}, inplace=True)

    if "bought" in X_new.columns:
        # Frequency: últimas 6 semanas sin incluir la actual
        X_new["frequency"] = X_new.groupby(["customer_id", "product_id"])[
            "bought"
        ].transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).sum())

        # Total de compras del cliente hasta la semana anterior
        X_new["total_purchases"] = X_new.groupby("customer_id")["bought"].transform(
            lambda s: s.shift(1).cumsum()
        )

        # Customer-product share
        X_new["customer_product_share"] = X_new["frequency"] / X_new["total_purchases"]
        X_new["customer_product_share"].replace([np.inf, -np.inf], np.nan, inplace=True)
        X_new.fillna({"customer_product_share": 0}, inplace=True)

        # Trend (sin data leakage)
        recent = X_new.groupby(["customer_id", "product_id"])["bought"].transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
        )
        past = X_new["frequency"] - recent
        X_new["trend"] = recent - past
    else:
        X_new["frequency"] = 0
        X_new["total_purchases"] = 0
        X_new["customer_product_share"] = 0
        X_new["trend"] = 0

    X_new.drop(
        columns=["bought", "last_purchase_week", "total_purchases"],
        errors="ignore",
        inplace=True,
    )
    return X_new


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.y_train_ = None

    def fit(self, X, y=None):
        # guardamos y para usar en transform
        self.y_train_ = y.copy() if y is not None else None
        return self

    def transform(self, X, y=None):
        X_new = X.copy()
        y_to_use = y if y is not None else self.y_train_

        if y_to_use is not None:
            X_new["bought"] = y_to_use
            X_new = X_new.sort_values(by=["customer_id", "product_id", "week"])

            X_new["last_purchase_week"] = X_new.groupby(["customer_id", "product_id"])[
                "week"
            ].shift(1)
            X_new["recency"] = X_new["week"] - X_new["last_purchase_week"]
            X_new.fillna({"recency": X_new["recency"].max()}, inplace=True)

            X_new["frequency"] = X_new.groupby(["customer_id", "product_id"])[
                "bought"
            ].transform(lambda s: s.shift(1).rolling(window=6, min_periods=1).sum())

            X_new["total_purchases"] = X_new.groupby("customer_id")["bought"].transform(
                lambda s: s.shift(1).cumsum()
            )

            X_new["customer_product_share"] = (
                (X_new["frequency"] / X_new["total_purchases"])
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

            recent = X_new.groupby(["customer_id", "product_id"])["bought"].transform(
                lambda s: s.shift(1).rolling(window=3, min_periods=1).sum()
            )
            past = X_new["frequency"] - recent
            X_new["trend"] = recent - past

            X_new.drop(
                columns=[
                    "bought",
                    "last_purchase_week",
                    "total_purchases",
                    "customer_id",
                ],
                inplace=True,
                errors="ignore",
            )

        return X_new


# =====================================================
# 2. Clustering geográfico
# =====================================================
class GeoClusterer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=2):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=1892)

    def fit(self, X, y=None):
        if not {"X", "Y"}.issubset(X.columns):
            raise ValueError("Se requieren las columnas 'X' y 'Y' para el clustering.")
        # Ensure float64 for training
        X_geo = X[["X", "Y"]].astype("float64")
        self.kmeans.fit(X_geo)
        return self

    def transform(self, X, y=None):
        if not {"X", "Y"}.issubset(X.columns):
            raise ValueError("Se requieren las columnas 'X' y 'Y' para el clustering.")
        X = X.copy()
        # Convert to float64 to match KMeans training dtype
        X_geo = X[["X", "Y"]].astype("float64")
        X["cluster"] = self.kmeans.predict(X_geo)
        return X


def create_pipeline():
    # TODO: Agregar las transformaciones a las columans zone_id y region_id
    numerical_features = [
        "size",
        "num_deliver_per_week",
        "recency",
        "frequency",
        "customer_product_share",
        "trend",
    ]
    categorical_features = [
        "customer_type",
        "brand",
        "category",
        "sub_category",
        "segment",
        "package",
        "cluster",
        "product_id",
        "zone_id",
        "region_id",
    ]

    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="passthrough",  # Mantener columnas no especificadas
        verbose_feature_names_out=False,  # mantiene nombres legibles
    )

    # =====================================================
    # 4. Pipeline final (Feature Engineering + Preprocesamiento)
    # =====================================================
    feature_engineering_pipeline = Pipeline(
        [
            ("geo_clustering", GeoClusterer(n_clusters=2)),
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", preprocessor),
        ]
    )

    return feature_engineering_pipeline


def run_data_splitting(input_data_path: str, output_dir: str = None) -> tuple:
    """
    Run data splitting pipeline (for Airflow task).

    Parameters
    ----------
    input_data_path : str
        Path to processed data parquet
    output_dir : str, optional
        Directory to save split data

    Returns
    -------
    tuple
        (train_data, val_data) - No test set created
    """
    print("=" * 60)
    print("DATA SPLITTING PIPELINE")
    print("=" * 60)

    data = load_final_data(input_data_path)
    train_data, val_data = split_data(data, output_dir=output_dir)

    print("\n" + "=" * 60)
    print("DATA SPLITTING COMPLETED")
    print("=" * 60)

    return train_data, val_data


if __name__ == "__main__":
    input_path = "Proyecto/entrega2/airflow/data/processed/final_data.parquet"
    output_dir = "Proyecto/entrega2/airflow/data/processed"

    run_data_splitting(input_path, output_dir)

    print("\nCreating pipeline...")
    pipeline = create_pipeline()
    print("Pipeline created.")
    print(pipeline)
