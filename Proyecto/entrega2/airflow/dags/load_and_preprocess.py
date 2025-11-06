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


def join_data(dfs: Dict[str, pd.DataFrame]):
    """Join transactions products and customers DataFrames."""
    transactions_df = dfs.get("transacciones.parquet")
    products_df = dfs.get("productos.parquet")
    customers_df = dfs.get("clientes.parquet")

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

    # Save parquet file
    output_dir = "Proyecto/entrega2/airflow/data/processed"
    os.makedirs(output_dir, exist_ok=True)
    data.to_parquet(f"{output_dir}/final_data.parquet", index=False)


if __name__ == "__main__":
    folder_path = "Proyecto/entrega2/airflow/data/raw"
    data_frames = load_data(folder_path)
    data_frames["transacciones.parquet"] = preprocess_transactions(
        data_frames["transacciones.parquet"]
    )
    data_frames = optimize_dataframes(data_frames)
    data_frames = create_week_and_objective(data_frames)
    join_data(data_frames)
