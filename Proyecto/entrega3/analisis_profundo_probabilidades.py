"""
Análisis Profundo: ¿Por qué las probabilidades son tan bajas?
==============================================================
"""

import numpy as np
import pandas as pd

# Cargar datos
preds = pd.read_parquet("airflow/predictions/predictions_2025-12-03.parquet")
train = pd.read_parquet("airflow/data/processed/train_data.parquet")

print("=" * 80)
print("ANÁLISIS: ¿LAS PROBABILIDADES BAJAS INDICAN MAL MODELO?")
print("=" * 80)

# 1. Comparar distribución de probabilidades vs tasa real de compra
print("\n📊 COMPARACIÓN: PROBABILIDADES vs REALIDAD")
print("-" * 80)
real_buy_rate = train["bought"].mean()
pred_mean = preds["probability"].mean()
pred_median = preds["probability"].median()

print(
    f"Tasa real de compra (train):        {real_buy_rate:.4f} ({100*real_buy_rate:.2f}%)"
)
print(f"Probabilidad media (predicciones):  {pred_mean:.4f} ({100*pred_mean:.2f}%)")
print(
    f"Probabilidad mediana (predicciones): {pred_median:.4f} ({100*pred_median:.2f}%)"
)

ratio = pred_mean / real_buy_rate
print(f"\nRatio: Pred/Real = {ratio:.2f}")
if ratio > 2.5:
    print("✅ BIEN CALIBRADO: Las probabilidades están en el rango correcto")
elif ratio > 1.5:
    print("⚠️  ACEPTABLE: Ligeramente sobre-estimado pero razonable")
elif ratio < 0.5:
    print("🔴 MAL CALIBRADO: Probabilidades demasiado bajas")
else:
    print("⚠️  REVISAR: Calibración dudosa")

# 2. Analizar dispersión
print("\n📈 DISPERSIÓN DE PROBABILIDADES")
print("-" * 80)
std = preds["probability"].std()
cv = std / pred_mean  # Coefficient of variation
print(f"Desviación estándar:     {std:.6f}")
print(f"Coeficiente variación:   {cv:.4f}")
print(
    f"Rango:                   {preds['probability'].min():.6f} - {preds['probability'].max():.6f}"
)

if cv < 0.01:
    print("\n🔴 PROBLEMA GRAVE: Casi sin variación (modelo no discrimina)")
elif cv < 0.05:
    print("\n⚠️  PROBLEMA: Poca variación (modelo discrimina débilmente)")
elif cv < 0.15:
    print("\n⚠️  MEJORABLE: Variación moderada")
else:
    print("\n✅ BIEN: Buena variación")

# 3. Verificar si hay problema de "collapsed probabilities"
print("\n🔍 ANÁLISIS DE COLAPSO DE PROBABILIDADES")
print("-" * 80)
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print("Percentiles de probabilidad:")
for p in percentiles:
    val = preds["probability"].quantile(p / 100)
    print(f"  {p:2d}%: {val:.6f}")

# Calcular "spread" entre percentiles
p10_90_spread = preds["probability"].quantile(0.90) - preds["probability"].quantile(
    0.10
)
p25_75_spread = preds["probability"].quantile(0.75) - preds["probability"].quantile(
    0.25
)

print(f"\nSpread P10-P90: {p10_90_spread:.6f}")
print(f"Spread P25-P75: {p25_75_spread:.6f}")

if p10_90_spread < 0.005:
    print("🔴 COLAPSO SEVERO: 80% de predicciones en rango <0.5%")
elif p10_90_spread < 0.01:
    print("⚠️  COLAPSO MODERADO: 80% de predicciones en rango <1%")
else:
    print("✅ SIN COLAPSO: Distribución saludable")

# 4. Identificar si el modelo es demasiado conservador
print("\n🎯 EVALUACIÓN DE CONSERVADURISMO DEL MODELO")
print("-" * 80)
above_50pct = (preds["probability"] >= 0.5).sum()
above_30pct = (preds["probability"] >= 0.3).sum()
above_20pct = (preds["probability"] >= 0.2).sum()
above_10pct = (preds["probability"] >= 0.1).sum()

total = len(preds)
print(f"Predicciones >= 50%: {above_50pct:>7,} ({100*above_50pct/total:>6.2f}%)")
print(f"Predicciones >= 30%: {above_30pct:>7,} ({100*above_30pct/total:>6.2f}%)")
print(f"Predicciones >= 20%: {above_20pct:>7,} ({100*above_20pct/total:>6.2f}%)")
print(f"Predicciones >= 10%: {above_10pct:>7,} ({100*above_10pct/total:>6.2f}%)")

if above_10pct == 0:
    print("\n🔴 EXTREMADAMENTE CONSERVADOR: Ninguna predicción > 10%")
elif above_10pct < total * 0.001:
    print("\n🔴 MUY CONSERVADOR: <0.1% de predicciones > 10%")
elif above_20pct == 0:
    print("\n⚠️  CONSERVADOR: Ninguna predicción > 20%")
else:
    print("\n✅ RAZONABLE: Algunas predicciones con alta confianza")

# 5. Diagnóstico final
print("\n" + "=" * 80)
print("💡 DIAGNÓSTICO Y RECOMENDACIONES")
print("=" * 80)

problems = []
if cv < 0.05:
    problems.append("Poca variación en probabilidades")
if pred_mean < real_buy_rate * 0.7:
    problems.append("Probabilidades sistemáticamente bajas")
if above_10pct < total * 0.01:
    problems.append("Modelo extremadamente conservador")
if p10_90_spread < 0.01:
    problems.append("Colapso de probabilidades")

if len(problems) == 0:
    print("\n✅ MODELO SALUDABLE")
    print("Las probabilidades bajas son apropiadas dado el desbalance extremo.")
else:
    print("\n🔴 PROBLEMAS DETECTADOS:")
    for i, p in enumerate(problems, 1):
        print(f"  {i}. {p}")

    print("\n📋 RECOMENDACIONES:")
    print("\n1. INMEDIATO - Verificar scale_pos_weight:")
    scale_pos = 36.89
    print(f"   - Desbalance: 36.89:1")
    print(f"   - scale_pos_weight debería ser ~{scale_pos:.1f}")
    print(f"   - Verificar en logs de MLflow que se esté usando")

    print("\n2. Features probablemente débiles:")
    print("   - Recency/Frequency dependen de historial (muchos sin datos)")
    print("   - product_id como numérico NO captura patrones de compra")
    print("   - Faltan features temporales (estacionalidad)")

    print("\n3. Probar ajustes inmediatos:")
    print("   a) Reducir scale_pos_weight a 15-20 (menos conservador)")
    print("   b) Aumentar learning_rate (modelo aprende mejor)")
    print("   c) Reducir regularización (gamma, reg_alpha, reg_lambda)")

    print("\n4. Mejoras de features:")
    print("   - Target encoding para product_id")
    print("   - Features temporales (mes, día semana)")
    print("   - Tasa de compra por categoría/marca")
    print("   - Últimas N compras del cliente (no solo frecuencia)")

print("\n" + "=" * 80)
