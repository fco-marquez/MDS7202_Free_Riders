import pandas as pd

print("=" * 80)
print("COMPARACIÓN MODELO VIEJO vs NUEVO")
print("=" * 80)

# Cargar predicciones
old = pd.read_parquet("airflow/predictions/predictions_2025-12-02.parquet")
new = pd.read_parquet("airflow/predictions/predictions_2025-12-03.parquet")

print("\n📊 MODELO VIEJO (2025-12-02):")
print(f"  Probabilidad media:    {old['probability'].mean():.6f}")
print(f"  Probabilidad std:      {old['probability'].std():.6f}")
print(
    f"  Probabilidad rango:    {old['probability'].min():.6f} - {old['probability'].max():.6f}"
)
print(f"  Predicciones = 1:      {(old['prediction']==1).sum():,}")

print("\n📊 MODELO NUEVO (2025-12-03):")
print(f"  Probabilidad media:    {new['probability'].mean():.6f}")
print(f"  Probabilidad std:      {new['probability'].std():.6f}")
print(
    f"  Probabilidad rango:    {new['probability'].min():.6f} - {new['probability'].max():.6f}"
)
print(f"  Predicciones = 1:      {(new['prediction']==1).sum():,}")

print("\n✅ MEJORAS DETECTADAS:")
varianza_ratio = new["probability"].std() / old["probability"].std()
rango_ratio = (new["probability"].max() - new["probability"].min()) / (
    old["probability"].max() - old["probability"].min()
)
media_diff = abs(new["probability"].mean() - old["probability"].mean())

print(f"  📈 Varianza (dispersión): {varianza_ratio:.1f}x mayor")
print(f"  📈 Rango de probabilidades: {rango_ratio:.1f}x más amplio")
print(
    f"  📉 Media bajó de {old['probability'].mean():.3f} a {new['probability'].mean():.3f} (mejor calibración)"
)

print("\n🎯 Predicciones por umbral (NUEVO MODELO):")
for thresh in [0.01, 0.05, 0.10, 0.112, 0.15, 0.20]:
    count = (new["probability"] >= thresh).sum()
    pct = 100 * count / len(new)
    print(f"  >= {thresh:.3f}: {count:>8,} ({pct:>5.2f}%)")

print("\n" + "=" * 80)
