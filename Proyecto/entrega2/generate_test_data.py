"""
Test Data Generator
===================
Generates synthetic or modified data for testing the Airflow pipeline.

This script can:
1. Sample existing data and modify it
2. Generate completely synthetic data
3. Add new weeks to existing data to simulate new data arrival

Usage:
    python generate_test_data.py --mode sample --weeks 2
    python generate_test_data.py --mode synthetic --n_transactions 1000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil


# Paths - Dynamically determine base directory from script location
BASE_DIR = Path(__file__).parent.resolve()
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
BACKUP_DIR = BASE_DIR / "data" / "backup"


def backup_original_data():
    """Backup original data before generating test data."""
    print("=" * 60)
    print("BACKING UP ORIGINAL DATA")
    print("=" * 60)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    files_to_backup = ["clientes.parquet", "productos.parquet", "transacciones.parquet"]

    for filename in files_to_backup:
        source = RAW_DATA_DIR / filename
        if source.exists():
            dest = BACKUP_DIR / filename
            shutil.copy2(source, dest)
            print(f"✓ Backed up: {filename}")
        else:
            print(f"✗ File not found: {filename}")

    print(f"\nBackup saved to: {BACKUP_DIR}")
    print("=" * 60)


def restore_original_data():
    """Restore original data from backup."""
    print("=" * 60)
    print("RESTORING ORIGINAL DATA")
    print("=" * 60)

    if not BACKUP_DIR.exists():
        print("✗ No backup found!")
        return

    files_to_restore = ["clientes.parquet", "productos.parquet", "transacciones.parquet"]

    for filename in files_to_restore:
        source = BACKUP_DIR / filename
        if source.exists():
            dest = RAW_DATA_DIR / filename
            shutil.copy2(source, dest)
            print(f"✓ Restored: {filename}")
        else:
            print(f"✗ Backup file not found: {filename}")

    print(f"\nData restored from: {BACKUP_DIR}")
    print("=" * 60)


def add_new_weeks(n_weeks=1, noise_factor=0.1):
    """
    Add new weeks to transaction data to simulate new data arrival.

    Parameters
    ----------
    n_weeks : int
        Number of new weeks to add
    noise_factor : float
        Amount of noise to add (0-1)
    """
    print("=" * 60)
    print(f"ADDING {n_weeks} NEW WEEKS TO TRANSACTION DATA")
    print("=" * 60)

    # Backup first
    backup_original_data()

    # Load data
    transactions = pd.read_parquet(RAW_DATA_DIR / "transacciones.parquet")
    print(f"Loaded {len(transactions):,} transactions")

    # Get last week
    transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
    transactions['week'] = transactions['purchase_date'].dt.isocalendar().week

    last_week = transactions['week'].max()
    last_date = transactions['purchase_date'].max()

    print(f"Last week in data: {last_week}")
    print(f"Last date in data: {last_date}")

    # Sample recent transactions
    recent_weeks = transactions[transactions['week'] >= last_week - 4].copy()

    new_transactions = []

    for week_offset in range(1, n_weeks + 1):
        # Sample transactions
        sample = recent_weeks.sample(frac=0.8, replace=True)

        # Modify dates
        sample['purchase_date'] = last_date + pd.Timedelta(days=7 * week_offset)
        sample['week'] = last_week + week_offset

        # Add noise to items
        if noise_factor > 0:
            noise = np.random.normal(1, noise_factor, len(sample))
            sample['items'] = (sample['items'] * noise).clip(lower=1)

        new_transactions.append(sample)

    # Combine
    new_data = pd.concat(new_transactions, ignore_index=True)

    # Combine with original
    combined = pd.concat([transactions, new_data], ignore_index=True)

    # Drop week column before saving (it's calculated in preprocessing)
    combined = combined.drop(columns=['week'])

    # Save
    output_path = RAW_DATA_DIR / "transacciones.parquet"
    combined.to_parquet(output_path, index=False)

    print(f"\n✓ Added {len(new_data):,} new transactions")
    print(f"✓ Total transactions: {len(combined):,}")
    print(f"✓ New weeks added: {week_offset}")
    print(f"✓ Saved to: {output_path}")
    print("=" * 60)


def sample_existing_data(sample_frac=0.5, noise_factor=0.1):
    """
    Sample and modify existing data to create test dataset.

    Parameters
    ----------
    sample_frac : float
        Fraction of data to sample (0-1)
    noise_factor : float
        Amount of noise to add (0-1)
    """
    print("=" * 60)
    print(f"SAMPLING EXISTING DATA (frac={sample_frac}, noise={noise_factor})")
    print("=" * 60)

    # Backup first
    backup_original_data()

    # Load data
    transactions = pd.read_parquet(RAW_DATA_DIR / "transacciones.parquet")
    print(f"Original transactions: {len(transactions):,}")

    # Sample
    sampled = transactions.sample(frac=sample_frac, random_state=42)

    # Add noise to items
    if noise_factor > 0:
        noise = np.random.normal(1, noise_factor, len(sampled))
        sampled['items'] = (sampled['items'] * noise).clip(lower=1)

    # Save
    output_path = RAW_DATA_DIR / "transacciones.parquet"
    sampled.to_parquet(output_path, index=False)

    print(f"\n✓ Sampled {len(sampled):,} transactions ({sample_frac*100:.0f}%)")
    print(f"✓ Saved to: {output_path}")
    print("=" * 60)


def generate_synthetic_transactions(n_transactions=1000):
    """
    Generate completely synthetic transaction data.

    Parameters
    ----------
    n_transactions : int
        Number of transactions to generate
    """
    print("=" * 60)
    print(f"GENERATING {n_transactions:,} SYNTHETIC TRANSACTIONS")
    print("=" * 60)

    # Backup first
    backup_original_data()

    # Load customers and products to get valid IDs
    customers = pd.read_parquet(RAW_DATA_DIR / "clientes.parquet")
    products = pd.read_parquet(RAW_DATA_DIR / "productos.parquet")

    customer_ids = customers['customer_id'].values
    product_ids = products['product_id'].values

    print(f"Unique customers: {len(customer_ids):,}")
    print(f"Unique products: {len(product_ids):,}")

    # Generate synthetic transactions
    np.random.seed(42)

    synthetic_data = pd.DataFrame({
        'customer_id': np.random.choice(customer_ids, n_transactions),
        'product_id': np.random.choice(product_ids, n_transactions),
        'order_id': np.arange(1000000, 1000000 + n_transactions),
        'purchase_date': pd.date_range(start='2024-01-01', periods=n_transactions, freq='H'),
        'items': np.random.gamma(shape=2, scale=2, size=n_transactions).clip(min=1)
    })

    # Save
    output_path = RAW_DATA_DIR / "transacciones.parquet"
    synthetic_data.to_parquet(output_path, index=False)

    print(f"\n✓ Generated {len(synthetic_data):,} synthetic transactions")
    print(f"✓ Date range: {synthetic_data['purchase_date'].min()} to {synthetic_data['purchase_date'].max()}")
    print(f"✓ Saved to: {output_path}")
    print("=" * 60)


def print_data_summary():
    """Print summary of current data."""
    print("\n" + "=" * 60)
    print("CURRENT DATA SUMMARY")
    print("=" * 60)

    # Load all data
    customers = pd.read_parquet(RAW_DATA_DIR / "clientes.parquet")
    products = pd.read_parquet(RAW_DATA_DIR / "productos.parquet")
    transactions = pd.read_parquet(RAW_DATA_DIR / "transacciones.parquet")

    transactions['purchase_date'] = pd.to_datetime(transactions['purchase_date'])
    transactions['week'] = transactions['purchase_date'].dt.isocalendar().week

    print(f"\nCustomers: {len(customers):,}")
    print(f"Products: {len(products):,}")
    print(f"Transactions: {len(transactions):,}")

    print(f"\nTransaction date range:")
    print(f"  From: {transactions['purchase_date'].min()}")
    print(f"  To: {transactions['purchase_date'].max()}")

    print(f"\nWeeks:")
    print(f"  First week: {transactions['week'].min()}")
    print(f"  Last week: {transactions['week'].max()}")
    print(f"  Total weeks: {transactions['week'].nunique()}")

    print(f"\nItems statistics:")
    print(transactions['items'].describe())

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate test data for Airflow pipeline')

    parser.add_argument(
        '--mode',
        type=str,
        choices=['add_weeks', 'sample', 'synthetic', 'restore', 'summary'],
        required=True,
        help='Data generation mode'
    )

    parser.add_argument(
        '--weeks',
        type=int,
        default=1,
        help='Number of weeks to add (for add_weeks mode)'
    )

    parser.add_argument(
        '--sample_frac',
        type=float,
        default=0.5,
        help='Fraction of data to sample (for sample mode)'
    )

    parser.add_argument(
        '--n_transactions',
        type=int,
        default=1000,
        help='Number of transactions to generate (for synthetic mode)'
    )

    parser.add_argument(
        '--noise',
        type=float,
        default=0.1,
        help='Noise factor (0-1)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("TEST DATA GENERATOR")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    print("=" * 60)

    if args.mode == 'add_weeks':
        add_new_weeks(n_weeks=args.weeks, noise_factor=args.noise)
        print_data_summary()

    elif args.mode == 'sample':
        sample_existing_data(sample_frac=args.sample_frac, noise_factor=args.noise)
        print_data_summary()

    elif args.mode == 'synthetic':
        generate_synthetic_transactions(n_transactions=args.n_transactions)
        print_data_summary()

    elif args.mode == 'restore':
        restore_original_data()
        print_data_summary()

    elif args.mode == 'summary':
        print_data_summary()

    print("\n✓ DONE!")


if __name__ == "__main__":
    main()
