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
    """
    Join transactions, products, and customers DataFrames.

    Processes data in batches BY WEEK to avoid memory issues.
    Each week creates a customer×product universe, which is much smaller
    than creating the full customer×product×week universe at once.
    """
    import gc

    transactions_df = dfs.get("transacciones.parquet")
    products_df = dfs.get("productos.parquet")
    customers_df = dfs.get("clientes.parquet")

    # Filtramos clientes activos
    customers_with_transactions = transactions_df["customer_id"].unique()
    customers_df = customers_df[
        customers_df["customer_id"].isin(customers_with_transactions)
    ].copy()

    # Drop zona_id y region_id de products_df si existen
    if "zone_id" in customers_df.columns:
        customers_df = customers_df.drop(columns=["zone_id"])
    if "region_id" in customers_df.columns:
        customers_df = customers_df.drop(columns=["region_id"])

    # Filtramos productos activos
    products_with_transactions = transactions_df["product_id"].unique()
    products_df = products_df[
        products_df["product_id"].isin(products_with_transactions)
    ].copy()

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

    # Get unique values
    weeks = np.sort(transactions_df["week"].unique()).astype("int16")
    unique_customers = transactions_df["customer_id"].unique()
    unique_products = transactions_df["product_id"].unique()

    n_customers = len(unique_customers)
    n_products = len(unique_products)
    n_weeks = len(weeks)

    total_universe_size = n_customers * n_products * n_weeks
    print(f"Universe dimensions: {n_customers} customers x {n_products} products x {n_weeks} weeks")
    print(f"Total universe size: {total_universe_size:,} rows")
    print(f"Processing in batches by week to save memory...")

    if output_path is None:
        raise ValueError("output_path must be specified")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Columns to drop after merge
    cols_to_drop = ["order_id", "purchase_date", "items"]

    # Process week by week
    all_chunks = []
    total_rows = 0

    for week_idx, week in enumerate(weeks):
        print(f"  Processing week {week} ({week_idx + 1}/{n_weeks})...")

        # Create universe for this week only (customers × products)
        week_universe = pd.MultiIndex.from_product(
            [unique_customers, unique_products],
            names=["customer_id", "product_id"],
        ).to_frame(index=False)
        week_universe["week"] = np.int16(week)

        # Get transactions for this week
        week_transactions = transactions_df[transactions_df["week"] == week]

        # Merge with customer and product info
        week_universe = week_universe.merge(customers_df, on="customer_id", how="left")
        week_universe = week_universe.merge(products_df, on="product_id", how="left")

        # Merge with transactions
        week_data = week_universe.merge(
            week_transactions, on=["customer_id", "product_id", "week"], how="left"
        )

        week_data["bought"] = week_data["bought"].fillna(0).astype("int8")

        # Drop unnecessary columns
        existing_cols_to_drop = [c for c in cols_to_drop if c in week_data.columns]
        if existing_cols_to_drop:
            week_data.drop(columns=existing_cols_to_drop, inplace=True)

        all_chunks.append(week_data)
        total_rows += len(week_data)

        # Clean up
        del week_universe, week_transactions, week_data
        gc.collect()

    # Concatenate all weeks
    print(f"\nConcatenating {len(all_chunks)} week chunks...")
    data = pd.concat(all_chunks, ignore_index=True)
    del all_chunks
    gc.collect()

    print(f"Universe shape: {data.shape}")

    # Save to parquet
    data.to_parquet(output_path, index=False)
    print(f"Saved processed data to: {output_path}")

    return data


def join_data_incremental(
    new_transactions_df: pd.DataFrame,
    existing_data_path: str,
    products_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """
    Incrementally update the universe with new transaction data.

    Instead of rebuilding the entire universe, this function:
    1. Loads existing processed data
    2. Identifies new weeks in the incoming transactions
    3. Creates universe only for new weeks
    4. Appends to existing data

    This is much more memory-efficient for incremental updates.
    """
    import gc

    print("=" * 60)
    print("INCREMENTAL DATA UPDATE")
    print("=" * 60)

    # Load existing data
    print(f"Loading existing data from: {existing_data_path}")
    existing_data = pd.read_parquet(existing_data_path)
    existing_weeks = set(existing_data["week"].unique())
    print(f"Existing weeks: {sorted(existing_weeks)}")

    # Get new weeks from transactions
    new_weeks = set(new_transactions_df["week"].unique())
    truly_new_weeks = new_weeks - existing_weeks
    print(f"New transaction weeks: {sorted(new_weeks)}")
    print(f"Truly new weeks to add: {sorted(truly_new_weeks)}")

    if not truly_new_weeks:
        print("No new weeks to add. Returning existing data.")
        return existing_data

    # Prepare dataframes
    customers_df = customers_df.copy()
    products_df = products_df.copy()

    # Drop zona_id y region_id de products_df si existen
    if "zone_id" in customers_df.columns:
        customers_df = customers_df.drop(columns=["zone_id"])
    if "region_id" in customers_df.columns:
        customers_df = customers_df.drop(columns=["region_id"])

    # Get unique customers and products from existing data (maintain same universe)
    unique_customers = existing_data["customer_id"].unique()
    unique_products = existing_data["product_id"].unique()

    # Filter to active customers/products
    customers_df = customers_df[customers_df["customer_id"].isin(unique_customers)].copy()
    products_df = products_df[products_df["product_id"].isin(unique_products)].copy()

    # Optimize types
    customers_df["customer_id"] = customers_df["customer_id"].astype("int32")
    products_df["product_id"] = products_df["product_id"].astype("int32")
    new_transactions_df["customer_id"] = new_transactions_df["customer_id"].astype("int32")
    new_transactions_df["product_id"] = new_transactions_df["product_id"].astype("int32")

    customers_df["customer_type"] = customers_df["customer_type"].astype("category")
    products_df[["brand", "category", "sub_category", "segment", "package"]] = (
        products_df[["brand", "category", "sub_category", "segment", "package"]].astype("category")
    )

    cols_to_drop = ["order_id", "purchase_date", "items"]

    # Process only new weeks
    new_chunks = []
    for week in sorted(truly_new_weeks):
        print(f"  Processing new week {week}...")

        # Create universe for this week
        week_universe = pd.MultiIndex.from_product(
            [unique_customers, unique_products],
            names=["customer_id", "product_id"],
        ).to_frame(index=False)
        week_universe["week"] = np.int16(week)

        # Get transactions for this week
        week_transactions = new_transactions_df[new_transactions_df["week"] == week]

        # Merge with customer and product info
        week_universe = week_universe.merge(customers_df, on="customer_id", how="left")
        week_universe = week_universe.merge(products_df, on="product_id", how="left")

        # Merge with transactions
        week_data = week_universe.merge(
            week_transactions, on=["customer_id", "product_id", "week"], how="left"
        )

        week_data["bought"] = week_data["bought"].fillna(0).astype("int8")

        # Drop unnecessary columns
        existing_cols_to_drop = [c for c in cols_to_drop if c in week_data.columns]
        if existing_cols_to_drop:
            week_data.drop(columns=existing_cols_to_drop, inplace=True)

        new_chunks.append(week_data)
        del week_universe, week_transactions, week_data
        gc.collect()

    # Concatenate new data
    print(f"\nConcatenating {len(new_chunks)} new week chunks...")
    new_data = pd.concat(new_chunks, ignore_index=True)
    del new_chunks
    gc.collect()

    # Combine with existing data
    print(f"Combining with existing data...")
    combined_data = pd.concat([existing_data, new_data], ignore_index=True)
    del existing_data, new_data
    gc.collect()

    print(f"Combined universe shape: {combined_data.shape}")

    # Save
    combined_data.to_parquet(output_path, index=False)
    print(f"Saved updated data to: {output_path}")

    print("=" * 60)
    return combined_data


def run_preprocessing_pipeline(
    raw_data_folder: str,
    output_data_path: str = None,
    static_data_folder: str = None,
    existing_data_path: str = None,
) -> pd.DataFrame:
    """
    Run complete preprocessing pipeline (for Airflow task).

    Parameters
    ----------
    raw_data_folder : str
        Path to folder containing raw parquet files
    output_data_path : str, optional
        Path to save processed data
    static_data_folder : str, optional
        Path to folder containing static data (customers, products)
    existing_data_path : str, optional
        Path to existing processed data for incremental updates.
        If provided and file exists, will use incremental update mode.

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

    # Check if we should do incremental update
    use_incremental = (
        existing_data_path is not None
        and os.path.exists(existing_data_path)
        and output_data_path != existing_data_path  # Don't use incremental if same file
    )

    if use_incremental:
        print("\n Using INCREMENTAL update mode...")
        final_data = join_data_incremental(
            new_transactions_df=data_frames["transacciones.parquet"],
            existing_data_path=existing_data_path,
            products_df=data_frames["productos.parquet"],
            customers_df=data_frames["clientes.parquet"],
            output_path=output_data_path,
        )
    else:
        # Full rebuild
        print("\nJoining data and creating universe (full rebuild)...")
        final_data = join_data(data_frames, output_path=output_data_path)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)

    return final_data


if __name__ == "__main__":
    folder_path = "Proyecto/entrega2/airflow/data/raw"
    output_path = "Proyecto/entrega2/airflow/data/processed/final_data.parquet"
    run_preprocessing_pipeline(folder_path, output_path)
