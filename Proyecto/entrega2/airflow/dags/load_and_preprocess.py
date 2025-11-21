import glob
import os
from typing import Dict

import numpy as np
import pandas as pd


def load_data(folder_path: str, static_folder_path: str) -> Dict[str, pd.DataFrame]:
    """Load data from Parquet files in a folder into a DataFrame."""
    transaction_files = glob.glob(os.path.join(folder_path, "*.parquet"))
    if not transaction_files:
        raise FileNotFoundError(f"No Parquet files found in the folder: {folder_path}")
    static_files = glob.glob(os.path.join(static_folder_path, "*.parquet"))
    if not static_files:
        raise FileNotFoundError(
            f"No Parquet files found in the static folder: {static_folder_path}"
        )
    df_dict = {}
    if len(transaction_files) > 1:
        # Concatenate multiple transaction files into one DataFrame
        print(
            "Multiple transaction files found. Concatenating them into one DataFrame."
        )
        dfs = [pd.read_parquet(file) for file in transaction_files]
        transactions_df = pd.concat(dfs, ignore_index=True)
        # Delete individual parquet files to free memory
        for file in transaction_files:
            os.remove(file)
        print("Deleted individual transaction parquet files to free memory.")
        # Save concatenated DataFrame to a single parquet file
        concatenated_path = os.path.join(folder_path, "transacciones.parquet")
        transactions_df.to_parquet(concatenated_path, index=False)
        df_dict = {"transacciones.parquet": transactions_df}
    else:
        for file in transaction_files:
            print(f"Loading file: {file}")
            df = pd.read_parquet(file)
            df_dict[os.path.basename(file)] = df
    for file in static_files:
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
    iso_calendar = transactions_df["purchase_date"].dt.isocalendar()
    transactions_df["week"] = (
        (iso_calendar.year - iso_calendar.year.min()) * 52 + iso_calendar.week
    ).astype("int16")
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

    # Drop zona_id y region_id de products_df si existen
    if "zone_id" in products_df.columns:
        products_df = products_df.drop(columns=["zone_id"])
    if "region_id" in products_df.columns:
        products_df = products_df.drop(columns=["region_id"])

    # Filtramos productos activos
    products_with_transactions = transactions_df["product_id"].unique()
    products_df = products_df[
        products_df["product_id"].isin(products_with_transactions)
    ]

    # Mantener IDs como int para consistencia (NO convertir a string)
    customers_df["customer_id"] = customers_df["customer_id"].astype("int32")
    products_df["product_id"] = products_df["product_id"].astype("int32")
    transactions_df["customer_id"] = transactions_df["customer_id"].astype("int32")
    transactions_df["product_id"] = transactions_df["product_id"].astype("int32")
    transactions_df["order_id"] = transactions_df["order_id"].astype(str)

    # Convertir a categorías solo las columnas categóricas (NO los IDs)
    customers_df["customer_type"] = customers_df["customer_type"].astype("category")
    products_df[["brand", "category", "sub_category", "segment", "package"]] = (
        products_df[["brand", "category", "sub_category", "segment", "package"]].astype(
            "category"
        )
    )

    # Crear universo cliente-producto-semana
    weeks = pd.Series(transactions_df["week"].unique(), name="week")
    weeks = weeks.astype("int16")

    # Obtener IDs únicos
    unique_customers = transactions_df["customer_id"].unique()
    unique_products = transactions_df["product_id"].unique()

    universe = pd.MultiIndex.from_product(
        [
            unique_customers,
            unique_products,
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
        raise ValueError("output_path must be specified")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_parquet(output_path, index=False)
    print(f"Saved processed data to: {output_path}")

    return data


def run_preprocessing_pipeline(
    raw_data_folder: str, output_data_path: str = None, static_data_folder: str = None
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
    print(f"Static data from: {static_data_folder}")

    # Load data
    data_frames = load_data(raw_data_folder, static_data_folder)

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
