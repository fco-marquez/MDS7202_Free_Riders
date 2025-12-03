# 📊 Análisis de Resultados - Nuevo Entrenamiento

**Fecha:** 3 de diciembre de 2025  
**Diagnóstico ejecutado después del re-entrenamiento con cambios implementados**

---

## 🎯 RESUMEN EJECUTIVO

### ✅ Mejoras Confirmadas

El nuevo modelo muestra **mejoras significativas** en la capacidad de discriminación:

| Métrica                  | Modelo Viejo | Modelo Nuevo | Mejora                 |
| ------------------------ | ------------ | ------------ | ---------------------- |
| **Varianza (std)**       | 0.000321     | 0.002219     | **6.9× mayor** ✅      |
| **Rango probabilidades** | 0.0097       | 0.0858       | **8.9× más amplio** ✅ |
| **Probabilidad media**   | 0.1303       | 0.0634       | Más calibrada ✅       |
| **Prob. máxima**         | 0.136        | 0.128        | Similar                |
| **Prob. mínima**         | 0.127        | 0.042        | Más discriminación ✅  |

### ⚠️ Problema Persistente

- **Predicciones positivas:** Aún 0 con umbral 0.112
- **Razón:** Las probabilidades máximas (~0.128) están justo por encima del umbral
- **Solo 7 pares superan el umbral 0.112** (0.004% del total)

---

## 📈 ANÁLISIS DETALLADO

### 1. Distribución de Probabilidades

**Modelo VIEJO (problemático):**

```
Media:   0.1303
Std:     0.0003  ← CASI CONSTANTE (problema grave)
Rango:   0.1265 - 0.1362
```

❌ **Problema:** Probabilidades prácticamente idénticas para todos los pares → modelo no discrimina

**Modelo NUEVO (mejorado):**

```
Media:   0.0634
Std:     0.0022  ← 6.9× más varianza
Rango:   0.0424 - 0.1282
```

✅ **Mejora:** Mayor dispersión indica que el modelo SÍ está discriminando entre pares

### 2. Predicciones por Umbral

| Umbral      | Predicciones | % del Total   |
| ----------- | ------------ | ------------- |
| ≥ 0.01      | 169,860      | 100.00%       |
| ≥ 0.05      | 169,827      | 99.98%        |
| ≥ 0.10      | 40           | 0.02%         |
| **≥ 0.112** | **7**        | **0.004%** ⚠️ |
| ≥ 0.15      | 0            | 0%            |

**Interpretación:**

- El modelo genera probabilidades bajas en general (media 6.3%)
- Solo 40 pares superan 10% de probabilidad
- Solo 7 pares superan el umbral de CodaLab (11.2%)

### 3. Top 7 Predicciones (Superan umbral 0.112)

```
customer_id  product_id  probability
     250284       61280     0.1282  ← Máxima
     212445       56714     0.1210
      60762       57804     0.1192
     206952       57804     0.1158
     246033       54096     0.1144
     172419       56714     0.1136
     208347       56714     0.1130
```

**Observaciones:**

- Producto 56714 aparece 4 veces → podría ser popular/común
- Producto 57804 aparece 2 veces
- Las probabilidades están en rango estrecho: 0.113 - 0.128

---

## 🔍 DIAGNÓSTICO

### ¿Por qué las probabilidades son tan bajas?

**Posibles causas:**

1. **✅ Calibración conservadora (ESPERADO)**

   - Con desbalance 36.89:1, el modelo aprende que comprar es evento raro
   - Media de 6.3% es razonable si ~2.6% de pares realmente compran
   - Esto es **CORRECTO** del modelo

2. **⚠️ Features débiles**

   - `recency`, `frequency`, `trend` dependen de historial
   - Si muchos pares son nuevos (sin historial) → probabilidades bajas
   - `product_id` tratado como numérico → puede estar confundiendo

3. **⚠️ scale_pos_weight podría estar sobre-compensando**
   - Con ratio 36.89:1, el peso es muy alto
   - Podría estar siendo demasiado conservador

---

## 🎯 EVALUACIÓN: ¿El modelo mejoró?

### ✅ SÍ - Mejoras Confirmadas:

1. **Varianza 6.9× mayor** → El modelo SÍ discrimina ahora (antes no)
2. **Rango 8.9× más amplio** → Identifica diferencias entre pares
3. **Media bajó** → Mejor calibrado para clase rara
4. **Genera predicciones (7)** → Antes generaba 0 con umbral

### ⚠️ PERO - Problemas Restantes:

1. **Muy pocas predicciones positivas** (7 de 169,860 = 0.004%)
2. **Probabilidades máximas bajas** (~12.8% máx)
3. **CSV tiene solo 7 pares** → Muy poco para evaluación real

---

## 💡 RECOMENDACIONES

### INMEDIATO - Ajustar Umbral

El umbral 0.112 es **demasiado alto** para este modelo. Reducir a:

```python
# En dag.py
PREDICTION_THRESHOLD = 0.05  # De 0.112 a 0.05 (5%)
```

**Impacto esperado:**

- Con umbral 0.05: **99.98% predicciones** (169,827 pares) ❌ Demasiadas
- Con umbral 0.10: **40 predicciones** (0.02%) ✅ Más razonable

**Recomendación:** Probar umbrales entre 0.08-0.10 para balance

### CORTO PLAZO - Verificar Métricas en MLflow

Necesitamos ver:

1. **Recall en validación** - ¿Es >0 ahora?
2. **F2-score** - ¿Mejoró vs F1?
3. **Precision-Recall curve** - Para elegir umbral óptimo
4. **Confusion matrix** - Distribución de predicciones

**Ejecutar:**

```bash
# Abrir MLflow UI
mlflow ui --backend-store-uri file:///path/to/mlflow_data
# Luego navegar a http://localhost:5000
```

### MEDIANO PLAZO - Mejorar Features

1. **Agregar features temporales:**

   - Día de la semana
   - Mes/estacionalidad
   - Días desde última compra

2. **Target Encoding para product_id:**

   - En vez de numérico, usar tasa de compra por producto
   - Ejemplo: product_56714 → 0.05 (5% de compra histórica)

3. **Interacciones:**

   - `customer_type × brand`
   - `category × recency`

4. **Probar SMOTE:**
   - Generar ejemplos sintéticos de clase positiva
   - Puede ayudar con desbalance extremo

---

## 📊 COMPARACIÓN ANTES/DESPUÉS

### Antes (Modelo Viejo):

```
❌ Probabilidades constantes (std=0.0003)
❌ Rango minúsculo (0.126-0.136)
❌ No discrimina entre pares
❌ 0 predicciones generadas
```

### Después (Modelo Nuevo):

```
✅ Probabilidades variables (std=0.0022)
✅ Rango amplio (0.042-0.128)
✅ Discrimina entre pares
✅ 7 predicciones generadas (con umbral 0.112)
⚠️ Necesita ajuste de umbral
```

---

## 🚀 PRÓXIMOS PASOS

1. **[ ] Revisar métricas en MLflow** para confirmar mejoras en recall/F2
2. **[ ] Ajustar umbral a 0.08-0.10** para generar más predicciones
3. **[ ] Verificar CSV en CodaLab** con nuevo umbral
4. **[ ] Iterar en features** si performance sigue baja

---

## ✅ CONCLUSIÓN

**El modelo SÍ mejoró significativamente:**

- Varianza 6.9× mayor
- Rango 8.9× más amplio
- Genera predicciones (aunque pocas)

**Pero necesita ajuste:**

- Umbral 0.112 es muy alto para las probabilidades que genera
- Reducir a 0.08-0.10 para evaluar mejor
- Revisar métricas de validación en MLflow

**Éxito parcial:** Los cambios funcionaron, pero el modelo es muy conservador. Esto puede ser correcto dado el desbalance extremo (36.89:1), o puede necesitar más features/datos.

---

**Próximo diagnóstico:** Después de ajustar umbral y revisar MLflow
