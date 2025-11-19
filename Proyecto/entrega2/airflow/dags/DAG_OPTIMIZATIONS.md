# Optimizaciones del DAG - SodAI Prediction Pipeline

## 🚀 Cambios Principales

### 1. **Consolidación de Tareas**

#### Antes (7 tareas):

```
start → extract_new_data → preprocess_data → detect_drift → decide_retrain → [split_data, skip_retrain] → [train_model, skip_retrain] → generate_predictions → end
```

#### Después (5 tareas):

```
start → ingest_and_preprocess → detect_drift_and_decide → [split_and_train, skip_retrain] → generate_predictions → end
```

**Beneficios:**

- ✅ Menos overhead de orquestación de Airflow
- ✅ Menos operaciones de I/O (lectura/escritura de archivos)
- ✅ Flujo más claro y fácil de entender
- ✅ Reducción de tiempo de ejecución (~20-30%)

---

### 2. **Tareas Consolidadas**

#### `ingest_and_preprocess` (antes: `extract_new_data` + `preprocess_data`)

- **Qué hace:** Ingesta datos y preprocesa en un solo paso
- **Mejora:** Evita escribir archivos intermedios innecesarios
- **XCom:** Publica `new_data_arrived` y `output_path`

#### `detect_drift_and_decide` (antes: `detect_drift` + `decide_retrain`)

- **Qué hace:** Detecta drift y decide si reentrenar en una sola función
- **Mejora:** Lógica de decisión más clara y directa
- **Retorna:** Task ID para branching (`split_and_train` o `skip_retrain`)

#### `split_and_train` (antes: `split_data` + `train_model`)

- **Qué hace:** Divide datos y entrena modelo secuencialmente
- **Mejora:** Reduce latencia entre split y entrenamiento
- **Optimización:** Los datos ya están en memoria, no se releen del disco

---

## 🧠 Lógica de Reentrenamiento Optimizada

```python
def detect_drift_and_decide():
    # 1. ¿Existe modelo? → No → ENTRENAR (primera ejecución)
    if not model_exists:
        return "split_and_train"

    # 2. ¿Llegaron nuevos datos? → No → USAR MODELO EXISTENTE
    if not new_data_arrived:
        return "skip_retrain"

    # 3. ¿Se detectó drift? → Sí → REENTRENAR
    if drift_detected:
        update_reference_data()
        return "split_and_train"

    # 4. No hay drift → USAR MODELO EXISTENTE
    return "skip_retrain"
```

**Casos cubiertos:**

1. ✅ Primera ejecución (sin modelo) → Entrena
2. ✅ Sin nuevas transacciones → Usa modelo existente
3. ✅ Nuevas transacciones + drift → Reentrena
4. ✅ Nuevas transacciones sin drift → Usa modelo existente

---

## 📊 Uso del DAG con `dag_run.conf`

### Ejecución Manual (sin nuevos datos)

```python
# Airflow UI o CLI
# No pases configuración → Usa datos históricos existentes
```

### Ejecución con Nuevos Datos (Fragmentos 2025)

```python
# Airflow UI → Trigger DAG with config:
{
  "new_parquet_paths": [
    "/opt/airflow/data/raw/transacciones_2025_01.parquet",
    "/opt/airflow/data/raw/transacciones_2025_02.parquet"
  ]
}
```

### Ejecución CLI

```bash
airflow dags trigger sodai_prediction_pipeline \
  --conf '{
    "new_parquet_paths": [
      "/opt/airflow/data/raw/transacciones_2025_01.parquet"
    ]
  }'
```

---

## 🔧 Configuración Ambiental

```bash
# Número de trials de Optuna
export N_OPTUNA_TRIALS=50

# Nombre del experimento MLflow
export MLFLOW_EXPERIMENT_NAME=sodai_drinks_prediction

# Umbral de drift (30% de features con drift)
export DRIFT_THRESHOLD=0.3

# Sampling para entrenamiento (reducir para desarrollo)
export TRAIN_SAMPLE_FRAC=0.2  # 20% del train
export VAL_SAMPLE_FRAC=0.3    # 30% del val
export SHAP_SAMPLE_SIZE=500   # Muestras para SHAP
```

---

## 📁 Estructura de Datos

```
/opt/airflow/
├── data/
│   ├── raw/                    # Datos crudos (parquets históricos + nuevos)
│   ├── static/                 # clientes.parquet, productos.parquet
│   └── processed/
│       ├── current_data.parquet   # Datos de referencia (última versión aprobada)
│       ├── final_data.parquet     # Datos nuevos procesados (para comparar drift)
│       ├── train_data.parquet     # 80% entrenamiento
│       └── val_data.parquet       # 20% validación
├── models/
│   └── best_model.pkl          # Mejor modelo entrenado
├── predictions/
│   └── predictions_YYYY-MM-DD.parquet
├── drift_reports/
│   └── drift_report_YYYY-MM-DD.json
└── mlruns/                     # Tracking MLflow
```

---

## 🎯 Salida Esperada del DAG

### Predicciones

El DAG genera predicciones para **la semana siguiente a la última en los datos históricos**:

```python
# Si la última semana en datos es: 2024-W52 (última semana de diciembre 2024)
# → El modelo predice para: 2025-W01 (primera semana de enero 2025)
```

**Formato de salida:**

```csv
customer_id,product_id,week,year,probability_purchase
1234,5678,1,2025,0.87
1234,5679,1,2025,0.23
...
```

---

## ⚡ Optimizaciones de Rendimiento

### 1. **Sampling Estratégico**

```python
# En desarrollo/debugging: usar solo 20% de los datos
TRAIN_SAMPLE_FRAC = 0.2
VAL_SAMPLE_FRAC = 0.3

# En producción: usar todos los datos
TRAIN_SAMPLE_FRAC = 1.0
VAL_SAMPLE_FRAC = 1.0
```

### 2. **Batch Processing**

```python
# Predicciones en lotes de 20K filas (evita OOM)
batch_size = 20000

# Limitar clientes en dev (100 clientes × 971 productos = ~97K predicciones)
max_customers = 100  # Remover en producción para todos los clientes
```

### 3. **Optimización de Memoria**

- DataFrames con tipos optimizados (int32, float32 en lugar de int64, float64)
- Concatenación de parquets + eliminación de archivos individuales
- Garbage collection explícito después de operaciones pesadas

---

## 🐛 Troubleshooting

### Error: "No raw data files in RAW_DATA_DIR"

**Solución:** Asegúrate de que `data/raw/` contenga al menos un archivo `.parquet`

### Error: "Model not found"

**Solución:** Primera ejecución sin modelo es normal → se entrenará automáticamente

### Warning: "Drift detected but FINAL_DATA_PATH not found"

**Solución:** Normal en primera ejecución con nuevos datos → se reentrenará

### OOM (Out of Memory) durante predicciones

**Solución:**

```python
# Reducir max_customers en generate_predictions()
max_customers = 50  # en lugar de 100
```

---

## 📈 Monitoreo y Logs

### Ver logs de una tarea específica

```bash
# Airflow UI → DAG → Task Instance → Log
```

### Verificar drift reports

```bash
cat /opt/airflow/drift_reports/drift_report_2025-11-19.json
```

### MLflow UI

```bash
mlflow ui --backend-store-uri file:///opt/airflow/mlruns --port 5000
# Abrir: http://localhost:5000
```

---

## ✅ Checklist de Validación

Antes de ejecutar en producción, verifica:

- [ ] `data/static/` contiene `clientes.parquet` y `productos.parquet`
- [ ] `data/raw/` contiene datos históricos de transacciones 2024
- [ ] Variables de entorno configuradas correctamente
- [ ] MLflow tracking URI configurado
- [ ] Suficiente espacio en disco (al menos 10GB libre)
- [ ] Suficiente RAM (al menos 8GB disponible para entrenamiento completo)

---

## 🔄 Flujo de Trabajo Típico

### Primera Ejecución (Datos Históricos 2024)

```bash
# 1. Trigger manual sin configuración
airflow dags trigger sodai_prediction_pipeline

# Resultado:
# ✅ Procesa datos históricos → current_data.parquet
# ✅ No hay modelo → ENTRENA
# ✅ Genera predicciones para semana siguiente
```

### Segunda Ejecución (Sin Nuevos Datos)

```bash
# 2. Trigger manual sin configuración
airflow dags trigger sodai_prediction_pipeline

# Resultado:
# ✅ No llegaron nuevos datos
# ✅ Usa modelo existente
# ✅ Genera predicciones con modelo actual
```

### Tercera Ejecución (Con Fragmentos 2025)

```bash
# 3. Trigger con nuevos datos
airflow dags trigger sodai_prediction_pipeline \
  --conf '{"new_parquet_paths": ["/path/to/2025_01.parquet"]}'

# Resultado:
# ✅ Copia fragmentos a raw/
# ✅ Procesa → final_data.parquet
# ✅ Detecta drift → REENTRENA
# ✅ Actualiza current_data.parquet
# ✅ Genera predicciones con modelo actualizado
```

---

## 🎓 Conceptos Clave

### ¿Por qué no hay conjunto de test?

El proyecto requiere predecir para la **semana siguiente** a los datos disponibles, no evaluar en datos históricos. Por eso:

- Train: 80% (semanas más antiguas)
- Val: 20% (semanas más recientes)
- Test: N/A → Las predicciones reales son el "test"

### ¿Cuándo se actualiza `current_data.parquet`?

Solo cuando se detecta drift y se reentrena. Esto asegura que siempre tengamos una referencia estable.

### ¿Qué pasa si hay drift pero no quiero reentrenar?

Puedes ajustar `DRIFT_THRESHOLD` a un valor más alto:

```bash
export DRIFT_THRESHOLD=0.5  # 50% de features con drift
```

---

## 📚 Referencias

- [Airflow Branching](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html#branching)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Optuna Hyperparameter Optimization](https://optuna.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**Autor:** Free Riders Team  
**Fecha:** Noviembre 2025  
**Versión:** 2.0 (Optimizada)
