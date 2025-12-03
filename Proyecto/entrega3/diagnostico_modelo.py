"""
Script de Diagnóstico del Modelo
=================================
Verifica el estado actual de datos de entrenamiento, validación y predicciones.
Ayuda a identificar problemas de desbalance de clases, distribución de probabilidades, etc.

Uso:
    python diagnostico_modelo.py
"""

import os
from pathlib import Path

import pandas as pd


def diagnose_training_data(train_path: str, val_path: str):
    """Diagnostica los datos de entrenamiento y validación."""
    print("=" * 80)
    print("DIAGNÓSTICO DE DATOS DE ENTRENAMIENTO")
    print("=" * 80)

    try:
        # Cargar datos
        print(f"\n📂 Cargando datos de entrenamiento: {train_path}")
        train_df = pd.read_parquet(train_path)
        print(f"   Shape: {train_df.shape}")
        print(f"   Columnas: {train_df.columns.tolist()}")

        print(f"\n📂 Cargando datos de validación: {val_path}")
        val_df = pd.read_parquet(val_path)
        print(f"   Shape: {val_df.shape}")

        # Distribución de clases en training
        print("\n📊 DISTRIBUCIÓN DE CLASES - TRAINING")
        print("-" * 80)
        train_class_dist = train_df["bought"].value_counts().sort_index()
        print(train_class_dist)
        print(f"\nProporción:")
        print(train_df["bought"].value_counts(normalize=True).sort_index())

        n_positive = (train_df["bought"] == 1).sum()
        n_negative = (train_df["bought"] == 0).sum()
        imbalance_ratio = n_negative / n_positive if n_positive > 0 else float("inf")
        print(f"\n⚖️  Imbalance Ratio: {imbalance_ratio:.2f}:1 (negative:positive)")

        if imbalance_ratio > 20:
            print("   ⚠️  WARNING: Severe class imbalance (>20:1)")
        elif imbalance_ratio > 10:
            print("   ⚠️  WARNING: High class imbalance (>10:1)")
        else:
            print("   ✅ Imbalance ratio is manageable")

        # Distribución de clases en validation
        print("\n📊 DISTRIBUCIÓN DE CLASES - VALIDATION")
        print("-" * 80)
        val_class_dist = val_df["bought"].value_counts().sort_index()
        print(val_class_dist)
        print(f"\nProporción:")
        print(val_df["bought"].value_counts(normalize=True).sort_index())

        # Características principales
        print("\n📈 ESTADÍSTICAS DE FEATURES PRINCIPALES")
        print("-" * 80)
        numeric_cols = train_df.select_dtypes(include=["int", "float"]).columns
        numeric_cols = [c for c in numeric_cols if c != "bought"]

        if len(numeric_cols) > 0:
            print(train_df[numeric_cols[:10]].describe().T)

    except Exception as e:
        print(f"❌ Error al cargar datos de entrenamiento: {e}")


def diagnose_predictions(predictions_dir: str):
    """Diagnostica las predicciones más recientes."""
    print("\n\n" + "=" * 80)
    print("DIAGNÓSTICO DE PREDICCIONES")
    print("=" * 80)

    try:
        # Buscar archivo de predicciones más reciente
        pred_files = list(Path(predictions_dir).glob("predictions_*.parquet"))

        if not pred_files:
            print(f"❌ No se encontraron archivos de predicciones en {predictions_dir}")
            return

        latest_pred = max(pred_files, key=lambda p: p.stat().st_mtime)
        print(f"\n📂 Archivo de predicciones más reciente: {latest_pred.name}")

        # Cargar predicciones
        preds = pd.read_parquet(latest_pred)
        print(f"   Shape: {preds.shape}")
        print(f"   Columnas: {preds.columns.tolist()}")

        # Distribución de predicciones
        print("\n📊 DISTRIBUCIÓN DE PREDICCIONES")
        print("-" * 80)
        if "prediction" in preds.columns:
            print(preds["prediction"].value_counts().sort_index())
            print(f"\nProporción:")
            print(preds["prediction"].value_counts(normalize=True).sort_index())
        else:
            print("❌ Columna 'prediction' no encontrada")

        # Distribución de probabilidades
        print("\n📊 DISTRIBUCIÓN DE PROBABILIDADES")
        print("-" * 80)
        if "probability" in preds.columns:
            print(preds["probability"].describe())

            print(f"\n🎲 Percentiles de probabilidad:")
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
                val = preds["probability"].quantile(p / 100)
                print(f"   {p:2d}%: {val:.6f}")

            print(f"\n🔍 Predicciones por umbral:")
            for thresh in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
                count = (preds["probability"] >= thresh).sum()
                pct = 100 * count / len(preds)
                print(f"   >= {thresh:.2f}: {count:>8,} ({pct:>5.2f}%)")

            # Alertas
            max_prob = preds["probability"].max()
            if max_prob < 0.1:
                print(
                    f"\n   ⚠️  WARNING: Probabilidad máxima muy baja ({max_prob:.4f} < 0.1)"
                )
                print("       → El modelo podría estar sesgado hacia clase negativa")
                print(
                    "       → Revisar scale_pos_weight, hiperparámetros, y datos de entrenamiento"
                )
        else:
            print("❌ Columna 'probability' no encontrada")

        # Verificar CSV de salida
        print("\n📄 CSV DE SALIDA (CODALAB)")
        print("-" * 80)
        csv_files = list(Path(predictions_dir).glob("prediccion_*.csv"))
        if csv_files:
            latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
            print(f"   Archivo: {latest_csv.name}")
            csv_df = pd.read_csv(latest_csv)
            print(f"   Predicciones: {len(csv_df):,} pares (customer, product)")
            print(f"   Tamaño: {latest_csv.stat().st_size:,} bytes")
        else:
            print("   ❌ No se encontraron archivos CSV")

    except Exception as e:
        print(f"❌ Error al cargar predicciones: {e}")


def main():
    """Ejecuta diagnóstico completo."""
    base_dir = Path(__file__).parent / "airflow"

    # Rutas
    train_path = base_dir / "data" / "processed" / "train_data.parquet"
    val_path = base_dir / "data" / "processed" / "val_data.parquet"
    predictions_dir = base_dir / "predictions"

    # Diagnósticos
    diagnose_training_data(str(train_path), str(val_path))
    diagnose_predictions(str(predictions_dir))

    print("\n" + "=" * 80)
    print("✅ DIAGNÓSTICO COMPLETADO")
    print("=" * 80)


if __name__ == "__main__":
    main()
