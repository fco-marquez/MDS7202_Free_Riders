import pandas as pd

# Cargar predicciones
pq = pd.read_parquet("airflow/predictions/predictions_2025-12-03.parquet")

print("=" * 80)
print("ANÁLISIS DE UMBRAL ÓPTIMO PARA CODALAB")
print("=" * 80)

print(f"\nUniverso total: {len(pq):,} pares customer-product")
print(f"Esperado (~2.6% compran): {int(len(pq)*0.026):,} predicciones")

print("\n" + "=" * 80)
print("PREDICCIONES POR UMBRAL:")
print("=" * 80)
print(f"{'Umbral':<10} {'Predicciones':<15} {'% del Total':<15} {'Evaluación'}")
print("-" * 80)

for thresh in [0.01, 0.03, 0.05, 0.06, 0.065, 0.07, 0.08, 0.09, 0.10, 0.112]:
    count = (pq["probability"] >= thresh).sum()
    pct = 100 * count / len(pq)

    # Evaluación
    if count < 1000:
        eval_text = "❌ MUY POCAS"
    elif count < 3000:
        eval_text = "⚠️  POCAS"
    elif count < 6000:
        eval_text = "✅ RAZONABLE"
    elif count < 10000:
        eval_text = "⚠️  MUCHAS"
    else:
        eval_text = "❌ DEMASIADAS"

    marker = "👉" if 0.06 <= thresh <= 0.07 else "  "
    print(f"{marker} {thresh:<8.3f} {count:>10,}     {pct:>6.2f}%        {eval_text}")

print("=" * 80)
print("\n🎯 RECOMENDACIÓN:")
print("   Umbral óptimo: 0.065 - 0.070")
print("   Genera ~4,000-8,000 predicciones (~2.5-4.5% del universo)")
print("   Esto debería dar F1 mucho mejor que 0.03")
print("=" * 80)

# Análisis del CSV actual
csv = pd.read_csv("airflow/predictions/prediccion_2025-12-03.csv")
print(f"\n📄 CSV ACTUAL (enviado a CodaLab):")
print(f"   Predicciones: {len(csv):,}")
print(f"   Porcentaje: {100*len(csv)/len(pq):.3f}%")
print(f"   F1 obtenido: 0.03 ❌")
print(f"\n   🔴 PROBLEMA: Solo enviaste {len(csv)} predicciones")
print(f"   🔴 Necesitas ~4,400 para mejor F1")
print("=" * 80)
