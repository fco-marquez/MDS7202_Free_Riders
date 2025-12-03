# 🔴 ANÁLISIS COMPLETO: Por Qué Tu Modelo Tiene Performance Pésima

## Resumen Ejecutivo

**F1 en CodaLab: 0.03** → **PÉSIMO**

Tienes razón: las probabilidades bajas NO son solo por desbalance. Hay **problemas graves** con el modelo.

---

## 🔍 PROBLEMAS IDENTIFICADOS (En Orden de Gravedad)

### 1. 🔴 COLAPSO DE PROBABILIDADES (MUY GRAVE)

**Evidencia:**

```
80% de predicciones en rango de solo 0.5% (0.062-0.065)
Spread P10-P90: 0.002951 (menos de 0.3%)
Coeficiente de variación: 0.035 (CRÍTICO)
```

**¿Qué significa?**

- El modelo NO está discriminando entre pares
- Casi todos los pares reciben ~6.3% de probabilidad
- Es como si el modelo solo aprendió la media y nada más

**Causa raíz:** `scale_pos_weight` demasiado alto (36.9) hace al modelo EXTREMADAMENTE conservador

---

### 2. 🔴 MODELO ULTRA-CONSERVADOR (MUY GRAVE)

**Evidencia:**

```
Predicciones >= 50%: 0
Predicciones >= 30%: 0
Predicciones >= 20%: 0
Predicciones >= 10%: 40 (0.02% del total)
Probabilidad máxima: 12.8%
```

**¿Qué significa?**

- El modelo NUNCA está seguro de nada
- Ni siquiera para las compras más obvias supera 13%
- Esto NO es normal, incluso con desbalance extremo

**Causa raíz:** Combinación de:

- `scale_pos_weight = 36.9` (demasiado alto)
- Regularización excesiva (`gamma`, `reg_alpha`, `reg_lambda`)
- Learning rate bajo (modelo no aprende agresivamente)

---

### 3. ⚠️ FEATURES DÉBILES (GRAVE)

**Problemas detectados:**

#### a) `product_id` como numérico

```python
numerical_features = [
    "product_id",  # ❌ Tratado como número ordinal
]
```

**Problema:** El modelo cree que product_id=100 y product_id=101 son similares (¡FALSO!)

**Impacto:** NO puede aprender patrones de compra por producto

#### b) Recency/Frequency/Trend dependen de historial

- Si un par customer-product es nuevo → features = 0
- Muchos pares sin historial → modelo no puede discriminar

#### c) Faltan features temporales

- Sin mes/día de la semana
- Sin estacionalidad
- Sin tendencias temporales

---

### 4. ⚠️ DESBALANCE MAL MANEJADO

**Ratio real:** 36.89:1 (negativo:positivo)

**Problema:** Usando `scale_pos_weight = 36.9` directamente hace que:

- El modelo penaliza DEMASIADO los falsos positivos
- Prefiere NO predecir compras (más seguro)
- Genera probabilidades artificialmente bajas

**Literatura:** Con desbalance >20:1, usar el ratio completo en `scale_pos_weight` suele colapsar las probabilidades. Mejor usar valores moderados (10-20).

---

## ✅ SOLUCIONES IMPLEMENTADAS (AHORA)

### Ajuste 1: `scale_pos_weight` con tope en 20

```python
# ANTES: Usaba 36.9 directamente
scale_pos_weight = n_negative / n_positive  # = 36.9

# AHORA: Tope en 20
scale_pos_weight = min(n_negative / n_positive, 20.0)
```

**Impacto esperado:** Modelo menos conservador, probabilidades más dispersas

---

### Ajuste 2: Learning rate más agresivo

```python
# ANTES: 0.01 - 0.3
"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)

# AHORA: 0.01 - 0.5
"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True)
```

**Impacto esperado:** Modelo aprende patrones más rápido

---

### Ajuste 3: Regularización reducida

```python
# ANTES:
"gamma": trial.suggest_float("gamma", 0, 2)        # Hasta 2.0
"reg_alpha": trial.suggest_float("reg_alpha", 0, 1)  # Hasta 1.0
"reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2)  # Mínimo 0.5

# AHORA:
"gamma": trial.suggest_float("gamma", 0.0, 1.5)      # Hasta 1.5
"reg_alpha": trial.suggest_float("reg_alpha", 0, 0.5)  # Hasta 0.5
"reg_lambda": trial.suggest_float("reg_lambda", 0.1, 1.5)  # Mínimo 0.1
```

**Impacto esperado:** Modelo más expresivo, menos penalizado

---

### Ajuste 4: Submuestreo menos agresivo

```python
# ANTES: Usaba IMBALANCE_RATIO_THRESHOLD (15:1)
n_majority_to_keep = int(n_minority * IMBALANCE_RATIO_THRESHOLD)

# AHORA: Tope en 15:1 incluso si threshold es mayor
n_majority_to_keep = int(n_minority * min(IMBALANCE_RATIO_THRESHOLD, 15))
```

---

## 📋 SOLUCIONES PENDIENTES (Siguiente Iteración)

### 1. 🎯 CRÍTICO: Target Encoding para `product_id`

En vez de tratar product_id como número:

```python
# Calcular tasa de compra por producto
product_buy_rate = train.groupby('product_id')['bought'].mean()

# Aplicar al dataset
X['product_buy_rate'] = X['product_id'].map(product_buy_rate)
X.drop('product_id', axis=1, inplace=True)  # Eliminar product_id numérico
```

**Impacto esperado:** **MEJORA MASIVA** - El modelo aprenderá qué productos se compran más

---

### 2. 🎯 IMPORTANTE: Features temporales

```python
# Agregar estacionalidad
X['month'] = X['week'] % 4  # Aproximación de mes
X['week_of_month'] = X['week'] % 4
X['is_month_end'] = (X['week'] % 4 == 3).astype(int)
```

---

### 3. 🎯 IMPORTANTE: Features de popularidad

```python
# Tasa de compra por categoría
category_buy_rate = train.groupby('category')['bought'].mean()
X['category_buy_rate'] = X['category'].map(category_buy_rate)

# Tasa de compra por marca
brand_buy_rate = train.groupby('brand')['bought'].mean()
X['brand_buy_rate'] = X['brand'].map(brand_buy_rate)
```

---

### 4. ⚠️ OPCIONAL: Probar SMOTE

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.1, random_state=42)  # Llegar a 10:1
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

---

## 🚀 PLAN DE ACCIÓN RECOMENDADO

### AHORA (Ya implementado):

1. ✅ Re-entrenar con `scale_pos_weight` tope en 20
2. ✅ Re-entrenar con learning rate hasta 0.5
3. ✅ Re-entrenar con regularización reducida

**Comando:**

```bash
# Re-trigger DAG en Airflow
docker-compose up -d
# O entrenar directamente
cd airflow/dags
python train_module.py
```

---

### DESPUÉS DEL RE-ENTRENAMIENTO:

**Si F1 mejora a >0.15:**

- Los ajustes funcionaron
- Continuar con features (target encoding, temporales)

**Si F1 sigue <0.10:**

- El problema es principalmente las features débiles
- PRIORIDAD: Implementar target encoding para product_id
- Agregar features temporales

---

## 📊 MEJORAS ESPERADAS

### Con ajustes de hiperparámetros (ahora):

```
Probabilidad máxima: 0.128 → 0.25-0.40
Spread P10-P90: 0.003 → 0.01-0.02
Predicciones >10%: 40 → 500-1000
F1 esperado: 0.03 → 0.08-0.15
```

### Con target encoding + features (siguiente):

```
F1 esperado: 0.15 → 0.25-0.35
```

---

## ⚠️ CONCLUSIÓN: TENÍAS RAZÓN

**Sí, las probabilidades bajas SÍ indican problemas del modelo:**

1. **Colapso de probabilidades** (80% en rango de 0.3%)
2. **Modelo ultra-conservador** (max 12.8%)
3. **Features débiles** (product_id mal codificado)
4. **scale_pos_weight excesivo** (36.9 colapsa predicciones)

El desbalance 36.89:1 es real, PERO un modelo bien entrenado debería:

- Generar probabilidades dispersas (no colapsadas)
- Tener algunas predicciones >30% para compras obvias
- Coeficiente de variación >0.1

**Acción inmediata:** Re-entrenar con los ajustes implementados y evaluar.

---

**Fecha:** 3 de diciembre de 2025  
**Estado:** Cambios implementados, listo para re-entrenamiento
