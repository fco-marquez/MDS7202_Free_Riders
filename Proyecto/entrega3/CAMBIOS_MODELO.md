# 🔧 Cambios Implementados para Mejorar el Modelo

**Fecha:** 2 de diciembre de 2025  
**Problema:** El modelo predice siempre `bought=0` (clase negativa)

---

## ✅ Cambios Implementados

### 🚨 **Cambio 1: CRÍTICO - Corrección de `scale_pos_weight`**

**Archivo:** `airflow/dags/train_module.py` (línea ~286)

**Antes:**

```python
scale_pos_weight = IMBALANCE_RATIO_THRESHOLD  # ❌ Valor fijo 8.0
```

**Después:**

```python
scale_pos_weight = scale_pos_weight  # ✅ Usa el valor calculado del desbalance real
```

**Impacto:** XGBoost ahora penaliza correctamente los errores en la clase minoritaria (bought=1) según el desbalance real de los datos, no un valor fijo.

---

### 📊 **Cambio 2: Ampliación de Hiperparámetros**

**Archivo:** `airflow/dags/train_module.py` (función `objective()`)

| Hiperparámetro     | Antes      | Después             | Razón                                             |
| ------------------ | ---------- | ------------------- | ------------------------------------------------- |
| `max_depth`        | 3-6        | **5-10**            | Árboles más profundos capturan patrones complejos |
| `learning_rate`    | 0.01-0.1   | **0.01-0.3**        | Permite aprendizaje más agresivo si necesario     |
| `min_child_weight` | 1-7        | **1-5**             | Menos restrictivo para divisiones                 |
| `subsample`        | 0.6-0.9    | **0.6-1.0**         | Permite usar todos los datos si es óptimo         |
| `colsample_bytree` | 0.6-0.9    | **0.6-1.0**         | Más features disponibles por árbol                |
| `gamma`            | 1.0 (fijo) | **0-2 (tunable)**   | Optuna decide la penalización óptima              |
| `reg_alpha`        | 0.5 (fijo) | **0-1 (tunable)**   | Regularización L1 ajustable                       |
| `reg_lambda`       | 1.0 (fijo) | **0.5-2 (tunable)** | Regularización L2 ajustable                       |

**Impacto:** El modelo tiene más capacidad para aprender patrones complejos y Optuna puede encontrar mejor configuración.

---

### 📈 **Cambio 3: Incremento de Datos de Entrenamiento**

**Archivo:** `airflow/dags/train_module.py` (líneas ~58-61)

| Parámetro                   | Antes     | Después       | Impacto                              |
| --------------------------- | --------- | ------------- | ------------------------------------ |
| `TRAIN_SAMPLE_FRAC`         | 0.2 (20%) | **0.6 (60%)** | 3× más datos para entrenar           |
| `IMBALANCE_RATIO_THRESHOLD` | 8:1       | **15:1**      | Mantiene casi 2× más datos negativos |

**Impacto:** El modelo entrena con **significativamente más datos**, mejorando su capacidad de generalización.

---

### 🎯 **Cambio 4: Optimización con F2-Score**

**Archivo:** `airflow/dags/train_module.py` (función `objective()`)

**Antes:**

```python
return f1  # ❌ F1 puede ser alto incluso sin detectar positivos
```

**Después:**

```python
# ✅ F2-score: Recall es 2× más importante que precision
beta = 2
f2 = ((1 + beta**2) * precision * recall / (beta**2 * precision + recall)) if (precision + recall) > 0 else 0
return f2
```

**Impacto:** Optuna prioriza modelos que **detectan más compras** (recall alto), aunque la precision baje ligeramente. Esto es ideal para clases desbalanceadas.

---

## 🔍 Diagnóstico Adicional

Se creó el script `diagnostico_modelo.py` para verificar:

- Balance de clases en train/validation
- Distribución de probabilidades
- Estadísticas de predicciones

**Uso:**

```bash
python diagnostico_modelo.py
```

---

## 📋 Próximos Pasos

### 1. **Inmediato - Re-entrenar el modelo**

```bash
# Desde el directorio del proyecto
docker-compose up -d
# Esperar a que Airflow ejecute el DAG o triggerearlo manualmente
```

### 2. **Verificar mejoras**

- Ejecutar `python diagnostico_modelo.py`
- Revisar métricas en MLflow (buscar F2-score, recall, precision)
- Verificar distribución de probabilidades en predicciones

### 3. **Ajustar umbral si necesario**

Si las probabilidades siguen siendo muy bajas (<0.1), modificar en `airflow/dags/dag.py`:

```python
PREDICTION_THRESHOLD = 0.05  # Reducir de 0.112 a 0.05
```

### 4. **Mejoras adicionales (futuro)**

- [ ] Agregar features temporales (mes, día semana, estacionalidad)
- [ ] Probar SMOTE para generar ejemplos sintéticos de clase minoritaria
- [ ] Considerar LightGBM o CatBoost como alternativas a XGBoost
- [ ] Implementar ensemble (combinar múltiples modelos)
- [ ] Target encoding para `product_id` en vez de tratarlo como numérico

---

## ⚠️ Notas Importantes

1. **Paciencia:** Con 60% de datos y Optuna explorando más hiperparámetros, el entrenamiento tomará **más tiempo** (esperado).

2. **Recursos:** Si hay problemas de memoria, reducir:

   - `TRAIN_SAMPLE_FRAC` a 0.4
   - `n_trials` en Optuna (default 50)

3. **Monitoreo:** Revisar logs de Airflow y MLflow para detectar errores o warnings.

---

## 📞 Soporte

Si los cambios no mejoran el modelo:

1. Ejecutar diagnóstico: `python diagnostico_modelo.py`
2. Revisar MLflow para comparar runs antes/después
3. Verificar distribución de probabilidades en predicciones
4. Considerar que el problema puede ser en los **datos** (features débiles, ruido, etc.)

---

**Última actualización:** 2 de diciembre de 2025
