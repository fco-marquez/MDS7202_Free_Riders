# 🎯 Resumen del Diagnóstico del Modelo

## 📊 Estado Actual (ANTES de los cambios)

### Datos de Entrenamiento

- **Total registros:** 7,338,575
- **Clase 0 (no compra):** 7,144,896 (97.36%)
- **Clase 1 (compra):** 193,679 (2.64%)
- **Ratio de desbalance:** **36.89:1** ⚠️ SEVERO

### Datos de Validación

- **Total registros:** 1,711,595
- **Clase 0:** 1,651,224 (96.47%)
- **Clase 1:** 60,371 (3.53%)

### Predicciones Actuales (Modelo Viejo)

- **Total predicciones:** 169,860
- **Predicción = 1:** 0 (0%) ❌ **NINGUNA compra predicha**
- **Predicción = 0:** 169,860 (100%)

### Probabilidades (Modelo Viejo)

- **Mínima:** 0.1265
- **Máxima:** 0.1362
- **Media:** 0.1303
- **Desviación:** 0.0003 ⚠️ **CASI CONSTANTE**

**🔴 PROBLEMA CONFIRMADO:** El modelo genera probabilidades prácticamente constantes (~13%) para TODOS los pares, indicando que **no está discriminando** entre compras y no-compras.

---

## ✅ Cambios Implementados

### 1. **scale_pos_weight CORREGIDO**

- Antes: Fijo en 8.0 (inadecuado para ratio 36.89:1)
- Después: Se calcula dinámicamente del desbalance real

### 2. **Hiperparámetros Ampliados**

- `max_depth`: 3-6 → **5-10**
- `learning_rate`: 0.01-0.1 → **0.01-0.3**
- `gamma`, `reg_alpha`, `reg_lambda`: Ahora **tunables**

### 3. **Más Datos de Entrenamiento**

- Sample fraction: 20% → **60%**
- Imbalance threshold: 8:1 → **15:1**

### 4. **Optimización con F2-score**

- Prioriza recall (detectar compras) sobre precision

---

## 🚀 Próximos Pasos

### INMEDIATO: Re-entrenar el modelo

Con los cambios implementados, el modelo debería:

1. Usar el `scale_pos_weight` correcto (~37 en vez de 8)
2. Entrenar con 3× más datos
3. Explorar hiperparámetros más agresivos
4. Optimizar para detectar más compras (F2-score)

### Ejecutar re-entrenamiento:

```bash
# Opción 1: Con Docker/Airflow
docker-compose up -d
# Luego triggerea el DAG manualmente en Airflow UI

# Opción 2: Entrenamiento directo (sin Airflow)
cd airflow/dags
python -c "from train_module import run_full_training; run_full_training('../../data/processed/train_data.parquet', '../../data/processed/val_data.parquet', n_trials=30, output_model_path='../../models/best_model_v2.pkl')"
```

### Verificar mejoras esperadas:

**Antes del re-entrenamiento:**

- Probabilidades: 0.126-0.136 (rango de 0.01, casi constante)
- Predicciones positivas: 0

**Esperado DESPUÉS:**

- Probabilidades: Mayor dispersión (0.01-0.90)
- Predicciones positivas: >0 (idealmente 2-5% del total)
- Recall en validación: >0 (antes era 0)

---

## 📈 Métricas a Monitorear en MLflow

1. **val_recall** - Debe ser >0 (antes era 0)
2. **val_f2** - Nueva métrica optimizada
3. **val_auc_pr** - Area bajo curva Precision-Recall
4. **Distribución de probabilidades** - Debe tener varianza

---

## ⚠️ Si Aún No Mejora

Si después del re-entrenamiento el modelo sigue prediciendo todo 0:

### Diagnóstico Avanzado:

1. Verificar que `scale_pos_weight` se esté usando (revisar logs)
2. Revisar features: ¿Recency/Frequency tienen varianza?
3. Probar reducir `PREDICTION_THRESHOLD` a 0.05
4. Considerar SMOTE para balance sintético

### Plan B - Estrategias Alternativas:

1. **Undersampling más agresivo:** Ratio 5:1 en vez de 15:1
2. **Algoritmo alternativo:** LightGBM con `is_unbalance=True`
3. **Two-stage model:** Primero detectar "clientes activos", luego productos
4. **Ensemble:** Combinar múltiples modelos con diferentes balances

---

**Fecha de diagnóstico:** 2 de diciembre de 2025
**Próxima revisión:** Después del re-entrenamiento
