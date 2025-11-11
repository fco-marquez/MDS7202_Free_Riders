import glob
import os
from typing import Dict

import numpy as np
import pandas as pd


def load_data(folder_path: str) -> Dict[str, pd.DataFrame]:
    """Load data from Parquet files in a folder into a DataFrame."""
    all_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not all_files:
        raise FileNotFoundError(f"No Parquet files found in the folder: {folder_path}")
    df_dict = {}
    for file in all_files:
        print(f"Loading file: {file}")
        df = pd.read_parquet(file)
        df_dict[os.path.basename(file)] = df
    return df_dict


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess transactions DataFrame."""
    df = df.drop_duplicates()
    df = df[df["items"] != 0]
    df["items"] = df["items"].abs()
    return df


def optimize_dataframes(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    def _optimize_numeric_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte de una sola vez:
        - int64  -> int32
        - float64 -> float32
        Mantiene el resto igual.
        """
        dtypes = df.dtypes
        mapping = {}

        # Selección vectorizada con numpy
        mask_int64 = dtypes.values == np.dtype("int64")
        mask_float64 = dtypes.values == np.dtype("float64")

        if mask_int64.any():
            mapping.update(
                dict(
                    zip(dtypes.index[mask_int64], np.repeat("int32", mask_int64.sum()))
                )
            )
        if mask_float64.any():
            mapping.update(
                dict(
                    zip(
                        dtypes.index[mask_float64],
                        np.repeat("float32", mask_float64.sum()),
                    )
                )
            )

        return df.astype(mapping, copy=False)

    optimized_dfs = {}
    for name, df in dfs.items():
        optimized_dfs[name] = _optimize_numeric_types(df)

    return optimized_dfs


def create_week_and_objective(dfs: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Create 'week' and objective variable (bought) columns in the transactions DataFrame."""
    transactions_df = dfs.get("transacciones.parquet")
    transactions_df["week"] = transactions_df["purchase_date"].dt.isocalendar().week
    # Crear variable objetivo como 1 (dtype int8)
    transactions_df["bought"] = np.array(1, dtype="int8")

    dfs["transacciones.parquet"] = transactions_df

    return dfs


def join_data(dfs: Dict[str, pd.DataFrame], output_path: str = None):
    """Join transactions products and customers DataFrames."""
    transactions_df = dfs.get("transacciones.parquet")
    products_df = dfs.get("productos.parquet")
    customers_df = dfs.get("clientes.parquet")

    # Filtramos clientes activos
    customers_with_transactions = transactions_df["customer_id"].unique()
    customers_df = customers_df[
        customers_df["customer_id"].isin(customers_with_transactions)
    ]

    # Filtramos productos activos
    products_with_transactions = transactions_df["product_id"].unique()
    products_df = products_df[
        products_df["product_id"].isin(products_with_transactions)
    ]

    customers_df["customer_id"] = customers_df["customer_id"].astype(str)
    products_df["product_id"] = products_df["product_id"].astype(str)
    transactions_df["customer_id"] = transactions_df["customer_id"].astype(str)
    transactions_df["product_id"] = transactions_df["product_id"].astype(str)
    transactions_df["order_id"] = transactions_df["order_id"].astype(str)
    for frame, cols in [
        (customers_df, ["customer_id", "customer_type"]),
        (
            products_df,
            ["product_id", "brand", "category", "sub_category", "segment", "package"],
        ),
        (transactions_df, ["customer_id", "product_id"]),
    ]:
        frame[cols] = frame[cols].astype("category")

    # Crear universo cliente-producto-semana
    weeks = pd.Series(transactions_df["week"].unique(), name="week")
    weeks = weeks.astype("int16")
    universe = pd.MultiIndex.from_product(
        [
            transactions_df["customer_id"].cat.categories,
            transactions_df["product_id"].cat.categories,
            weeks,
        ],
        names=["customer_id", "product_id", "week"],
    ).to_frame(index=False)

    # Agregar las variables de clientes y productos
    universe = universe.merge(customers_df, on="customer_id", how="left")
    universe = universe.merge(products_df, on="product_id", how="left")
    print("Universe shape:", universe.shape)
    data = universe.merge(
        transactions_df, on=["customer_id", "product_id", "week"], how="left"
    )

    data["bought"] = data["bought"].fillna(0).astype("int8")
    data.drop(columns=["order_id", "purchase_date", "items"], inplace=True)

    # Save parquet file
    if output_path is None:
        output_path = "Proyecto/entrega2/airflow/data/processed/final_data.parquet"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_parquet(output_path, index=False)
    print(f"Saved processed data to: {output_path}")

    return data


def run_preprocessing_pipeline(
    raw_data_folder: str, output_data_path: str = None
) -> pd.DataFrame:
    """
    Run complete preprocessing pipeline (for Airflow task).

    Parameters
    ----------
    raw_data_folder : str
        Path to folder containing raw parquet files
    output_data_path : str, optional
        Path to save processed data

    Returns
    -------
    pd.DataFrame
        Processed data
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    print(f"Loading data from: {raw_data_folder}")

    # Load data
    data_frames = load_data(raw_data_folder)

    # Preprocess transactions
    print("\nPreprocessing transactions...")
    data_frames["transacciones.parquet"] = preprocess_transactions(
        data_frames["transacciones.parquet"]
    )

    # Optimize datatypes
    print("\nOptimizing datatypes...")
    data_frames = optimize_dataframes(data_frames)

    # Create week and objective
    print("\nCreating week and objective variables...")
    data_frames = create_week_and_objective(data_frames)

    # Join data
    print("\nJoining data and creating universe...")
    final_data = join_data(data_frames, output_path=output_data_path)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    return final_data


if __name__ == "__main__":
    folder_path = "Proyecto/entrega2/airflow/data/raw"
    output_path = "Proyecto/entrega2/airflow/data/processed/final_data.parquet"
    run_preprocessing_pipeline(folder_path, output_path)
