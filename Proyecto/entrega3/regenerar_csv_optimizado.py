"""
Regenerar CSV de predicciones con umbral optimizado
====================================================
Genera nuevo CSV a partir del parquet existente sin re-entrenar.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

# Configuración
THRESHOLD = 0.066  # Optimizado para ~4,486 predicciones
PREDICTIONS_DIR = Path("airflow/predictions")
INPUT_PARQUET = PREDICTIONS_DIR / "predictions_2025-12-03.parquet"
OUTPUT_CSV = PREDICTIONS_DIR / f"prediccion_optimized_threshold_{THRESHOLD}.csv"

print("=" * 80)
print(f"REGENERANDO CSV CON UMBRAL OPTIMIZADO: {THRESHOLD}")
print("=" * 80)

# Cargar predicciones
print(f"\n📂 Cargando: {INPUT_PARQUET}")
preds = pd.read_parquet(INPUT_PARQUET)
print(f"   Total pares: {len(preds):,}")

# Estadísticas
print(f"\n📊 Distribución de probabilidades:")
print(f"   Min:    {preds['probability'].min():.6f}")
print(f"   Max:    {preds['probability'].max():.6f}")
print(f"   Mean:   {preds['probability'].mean():.6f}")
print(f"   Median: {preds['probability'].median():.6f}")

# Aplicar umbral
print(f"\n🎯 Aplicando umbral: {THRESHOLD}")
positive_preds = preds[preds["probability"] >= THRESHOLD][["customer_id", "product_id"]]

# Asegurar tipos integer
positive_preds["customer_id"] = positive_preds["customer_id"].astype(int)
positive_preds["product_id"] = positive_preds["product_id"].astype(int)

# Guardar CSV
positive_preds.to_csv(OUTPUT_CSV, index=False)

# Resumen
pct = 100 * len(positive_preds) / len(preds)
print(f"\n✅ CSV guardado: {OUTPUT_CSV}")
print(f"   Predicciones positivas: {len(positive_preds):,} ({pct:.2f}% del universo)")
print(f"   Esperado (~2.6%): {int(len(preds) * 0.026):,}")
print(f"   Diferencia: {len(positive_preds) - int(len(preds) * 0.026):+,}")

# Mostrar muestra
print(f"\n📄 Muestra del CSV (primeras 10 filas):")
print(positive_preds.head(10).to_string(index=False))

# Mostrar estadísticas de las predicciones seleccionadas
selected_probs = preds[preds["probability"] >= THRESHOLD]["probability"]
print(f"\n📊 Estadísticas de probabilidades seleccionadas:")
print(f"   Min:  {selected_probs.min():.6f}")
print(f"   Max:  {selected_probs.max():.6f}")
print(f"   Mean: {selected_probs.mean():.6f}")

print("\n" + "=" * 80)
print("🎯 PRÓXIMO PASO: Subir este CSV a CodaLab")
print(f"   Archivo: {OUTPUT_CSV}")
print("=" * 80)
