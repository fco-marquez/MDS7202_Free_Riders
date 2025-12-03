import pandas as pd

pq = pd.read_parquet("airflow/predictions/predictions_2025-12-03.parquet")

print("Búsqueda fina de umbral óptimo:")
print("=" * 50)
for t in [0.064, 0.065, 0.066, 0.067, 0.068, 0.069, 0.070, 0.071]:
    c = (pq["probability"] >= t).sum()
    diff = c - 4416
    print(f"  {t:.3f}: {c:>7,} predicciones (diff: {diff:>+6,})")

print("\nRecomendación: Usar 0.068 o 0.069")
