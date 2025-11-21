# Pipeline de Predicción SodAI

## Descripción del DAG

Pipeline automatizado que procesa datos de transacciones, detecta cambios en las distribuciones (drift), y genera predicciones semanales de compra por cliente-producto usando XGBoost optimizado con Optuna y trackeado en MLflow.

**DAG ID**: `sodai_prediction_pipeline`
**Schedule**: Manual (configurar `@weekly` para producción)
**Autor**: Free Riders Team

---

## Diagrama de Flujo

```
START
  │
  ▼
ingest_and_preprocess
  │ (Carga y limpia datos, crea universo cliente-producto-semana)
  │
  ▼
detect_drift_and_decide
  │ (Detecta drift estadístico y decide si reentrenar)
  │
  ├──────────┬──────────┐
  │          │          │
  ▼          ▼          ▼
  │    split_and_train  skip_retrain
  │    (Optuna + XGBoost)   (usa modelo existente)
  │          │          │
  └──────────┴──────────┘
  │
  ▼
generate_predictions
  │ (Predice semana N+1)
  │
  ▼
END
```

**Vista en Airflow UI**: El DAG muestra un flujo lineal con un branch en `detect_drift_and_decide` que dirige a training o skip según la detección de drift.

---

## Tareas del Pipeline

### 1. `ingest_and_preprocess`
**Archivo**: `load_and_preprocess.py`

Carga y procesa datos:
- Lee archivos parquet de `data/raw/` (históricos + nuevos fragmentos si existen)
- Carga catálogos estáticos (`clientes.parquet`, `productos.parquet`)
- Limpia transacciones (duplicados, items=0)
- Optimiza tipos de datos (int64→int32, float64→float32)
- Crea variable `week` y objetivo `bought` (1 si compró, 0 si no)
- Genera universo completo: clientes × productos × semanas

**Output**: `current_data.parquet` (primera vez) o `final_data.parquet` (ejecuciones posteriores)

---

### 2. `detect_drift_and_decide` (BranchPythonOperator)
**Archivo**: `drift_detector.py`

Lógica de decisión en 4 pasos:

```python
if no existe modelo:
    return "split_and_train"  # Primera ejecución

if no llegaron nuevos datos:
    return "skip_retrain"  # Usar modelo actual

# Ejecutar detección de drift
drift_detected = run_drift_detection()

if drift_detected:
    actualizar current_data.parquet
    return "split_and_train"
else:
    return "skip_retrain"
```

**Detección de drift**:
- Features numéricas (8): Kolmogorov-Smirnov test (p < 0.05)
- Features categóricas (6): Chi-cuadrado test (p < 0.05)
- Decisión: Si >30% de features tienen drift → Reentrenar

**Output**: `drift_report_{fecha}.json`

---

### 3a. `split_and_train`
**Archivos**: `pipeline.py`, `train_module.py`

Entrenamiento completo:

1. **Split temporal**: 80% train / 20% validation (sin test set)
2. **Optimización Optuna**: 50 trials maximizando F1-score
3. **Feature engineering**:
   - Recency: semanas desde última compra
   - Frequency: compras en últimas 6 semanas
   - Trend: tendencia reciente vs pasada
   - Clustering geográfico (KMeans 2 clusters)
4. **Training XGBoost** con mejores hiperparámetros
5. **Logging en MLflow**:
   - Métricas: precision, recall, F1, AUC-PR
   - Plots: confusion matrix, PR curve, SHAP
   - Modelo completo (pipeline)

**Output**: `best_model.pkl` + registro en MLflow

---

### 3b. `skip_retrain`
Placeholder vacío que marca la rama donde no se reentrena.

---

### 4. `generate_predictions`
**Archivo**: `predict_module.py`

Genera predicciones para semana siguiente:

1. Carga mejor modelo (desde MLflow o local)
2. Determina semana a predecir: `max(week) + 1`
3. Crea universo de predicción (clientes × productos × semana_N+1)
4. Aplica feature engineering
5. Genera predicciones en batches

**Output**: `predictions_{fecha}.parquet` con columnas:
- `customer_id`, `product_id`, `week`, `year`, `probability_purchase`

---

## Integración MLflow, Optuna y SHAP

### MLflow
- **Tracking URI**: `file:///opt/airflow/mlflow_data`
- **Experiment**: `sodai_drinks_prediction`
- Registra cada trial de Optuna como nested run
- Guarda modelo final con todos los artefactos

### Optuna
- 50 trials con TPE sampler
- Optimiza F1-score
- Espacio de búsqueda: max_depth, learning_rate, subsample, etc.
- Genera plots de optimization history y parameter importances

### SHAP
- Calcula importancia de features en muestra de 1000 observaciones
- Genera summary plot y bar plot
- Identifica features más relevantes para predicciones

---

## Lógica de Drift y Reentrenamiento

### ¿Cuándo se reentrena?

1. **Primera ejecución**: Siempre (no hay modelo)
2. **Sin nuevos datos**: Nunca (usa modelo existente)
3. **Nuevos datos + drift >30%**: Sí
4. **Nuevos datos + drift ≤30%**: No

### ¿Cómo se detecta drift?

Comparamos `current_data.parquet` (referencia histórica) vs `final_data.parquet` (datos nuevos):

- Test KS para numéricas: size, recency, frequency, etc.
- Test Chi² para categóricas: brand, category, segment, etc.
- Si p-value < 0.05 → Feature tiene drift
- Si drift_ratio > 0.30 → Reentrenar

### Actualización de referencia

Solo cuando hay drift y se reentrena:
```python
shutil.copyfile(final_data.parquet, current_data.parquet)
```

Esto mantiene una baseline estable para futuras comparaciones.

---

## Integración de Nuevos Datos

El pipeline está diseñado para recibir fragmentos incrementales:

### Ejecución con nuevos datos

**Desde Airflow UI**:
```json
{
  "new_parquet_paths": [
    "/opt/airflow/data/raw/transacciones_2025_01.parquet"
  ]
}
```

**Desde CLI**:
```bash
airflow dags trigger sodai_prediction_pipeline \
  --conf '{"new_parquet_paths": ["/path/to/new_data.parquet"]}'
```

El sistema:
1. Copia fragmentos a `data/raw/`
2. Los concatena con datos existentes
3. Procesa todo junto
4. Evalúa drift
5. Reentrena si es necesario
6. Genera predicciones para siguiente semana

---

## Configuración

### Variables de entorno
```bash
export N_OPTUNA_TRIALS=50
export MLFLOW_EXPERIMENT_NAME=sodai_drinks_prediction
export DRIFT_THRESHOLD=0.3
export TRAIN_SAMPLE_FRAC=0.2  # Sampling para desarrollo
export VAL_SAMPLE_FRAC=0.3
```

### Estructura de datos
```
airflow/
├── dags/
│   ├── dag.py                     # DAG principal
│   ├── load_and_preprocess.py     # Preprocesamiento
│   ├── pipeline.py                # Feature engineering
│   ├── train_module.py            # Entrenamiento
│   ├── predict_module.py          # Predicciones
│   ├── drift_detector.py          # Detección drift
│   └── mlflow_config.py           # Config MLflow
├── data/
│   ├── raw/                       # Datos crudos
│   ├── static/                    # Clientes, productos
│   └── processed/                 # Datos procesados
├── models/
│   └── best_model.pkl
├── predictions/
├── drift_reports/
└── mlflow_data/
```

---

## Supuestos

1. **Nuevos datos**: Misma estructura que datos históricos (schema idéntico)
2. **Catálogos estáticos**: Clientes y productos no cambian
3. **Frecuencia**: Datos llegan semanalmente
4. **Predicción**: Siempre para la semana N+1 (siguiente a la última en datos)
5. **Universo**: Solo clientes y productos con al menos 1 transacción histórica

---

## Acceso a MLflow UI

```bash
mlflow ui --backend-store-uri file:///opt/airflow/mlflow_data --port 5000
# Abrir: http://localhost:5000
```
