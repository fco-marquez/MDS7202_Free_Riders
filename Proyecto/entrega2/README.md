# SodAI Drinks - ML Prediction Pipeline

**Equipo:** Free Riders
**Curso:** MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos
**Entrega:** 2 - Pipeline Automatizado con Airflow

---

## Descripción

Pipeline automatizado de Machine Learning para predecir compras de productos por cliente utilizando Apache Airflow. Incluye detección automática de drift, reentrenamiento condicional, optimización de hiperparámetros con Optuna, y tracking con MLflow.

**Características:**
- Detección automática de drift (KS-test + Chi-square)
- Reentrenamiento condicional (solo cuando drift > 30%)
- Optimización con Optuna (20 trials)
- Interpretabilidad con SHAP
- FastAPI backend + Gradio frontend para predicciones interactivas

---

## Quick Start

### Prerrequisitos
- Docker Desktop instalado: https://www.docker.com/products/docker-desktop/
- 8GB RAM disponibles

### Iniciar el Sistema

```bash
# 1. Navegar al directorio del proyecto
cd C:\Users\fmarq\DCC\MDS\MDS7202_Laboratorio\MDS7202_Free_Riders\Proyecto\entrega2

# 2. Iniciar todos los servicios
docker-compose up -d

# 3. Verificar estado (esperar ~60 segundos)
docker-compose ps
```

### Acceso a Interfaces

- **Airflow UI**: http://localhost:8080 (admin / admin)
- **MLflow UI**: http://localhost:5000
- **Prediction API**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:7860

### Ejecutar Pipeline

1. Abrir Airflow UI: http://localhost:8080
2. Buscar DAG `sodai_prediction_pipeline`
3. Activar (toggle ON)
4. Click en "Trigger DAG" ▶️

### Comandos Útiles

```bash
# Ver logs
docker-compose logs -f

# Logs de servicio específico
docker-compose logs -f backend
docker-compose logs -f airflow

# Detener servicios
docker-compose stop

# Reiniciar servicios
docker-compose start

# Limpiar todo
docker-compose down

# Reconstruir servicio
docker-compose up -d --build backend
```

---

## Arquitectura

### Flujo del Pipeline

```
Extract Data → Preprocess → Split → Detect Drift → Branch
                                                      ├→ Train Model (if drift > 30%)
                                                      └→ Skip Retrain (if drift ≤ 30%)
                                                                  ↓
                                                        Generate Predictions
```

### Servicios

| Servicio | Puerto | Descripción |
|----------|--------|-------------|
| PostgreSQL | 5432 | Base de datos de Airflow |
| Airflow | 8080 | Orquestador del pipeline |
| MLflow | 5000 | Tracking de experimentos |
| Backend | 8000 | API de predicción (FastAPI) |
| Frontend | 7860 | UI de predicción (Gradio) |

Todos los servicios se comunican vía `sodai_network`.

---

## Pipeline Tasks

### 1. extract_new_data
Valida existencia de datos raw (`clientes.parquet`, `productos.parquet`, `transacciones.parquet`).

### 2. preprocess_data
- Limpia transacciones (duplicados, items inválidos)
- Crea variable `week` y target `bought`
- Genera universo cliente × producto × semana
- Output: `data/processed/final_data.parquet`

### 3. split_data
División temporal (no aleatoria) para evitar data leakage:
- Train: 70% (semanas antiguas)
- Validation: 15% (semanas intermedias)
- Test: 15% (semanas recientes)

### 4. detect_drift
Detecta cambios estadísticos en distribuciones:
- **KS-test** para variables numéricas (size, recency, frequency, etc.)
- **Chi-square** para categóricas (customer_type, brand, category, etc.)
- Threshold: p-value < 0.05 indica drift
- Decisión: Si > 30% features con drift → reentrenar

### 5. decide_retrain
Branching basado en drift detection.

### 6a. train_model (condicional)
Si hay drift:
1. **Optimización Optuna**: 50 trials, optimiza Recall
2. **Entrenamiento**: XGBoost con feature engineering (clustering geográfico + RFM)
3. **Evaluación**: Métricas + SHAP values
4. **Tracking**: Todo registrado en MLflow
5. **Persistencia**: Guarda modelo en `/models/best_model.pkl`

### 6b. skip_retrain (condicional)
Si no hay drift: Usa modelo existente.

### 7. generate_predictions
- Carga mejor modelo
- Crea universo para próxima semana
- Genera predicciones con probabilidades
- Output: `predictions/predictions_{date}.parquet`

---

## Detección de Drift

### Tests Estadísticos

**Kolmogorov-Smirnov (numéricas)**
- Compara distribuciones de datos históricos vs nuevos
- p-value < 0.05 → Drift detectado

**Chi-Square (categóricas)**
- Compara frecuencias de categorías
- p-value < 0.05 → Drift detectado

### Decisión de Reentrenamiento

```python
if (features_with_drift / total_features) > 0.3:
    needs_retrain = True
```

### Reporte

Generado en `drift_reports/drift_report_{date}.json`:

```json
{
  "needs_retrain": true,
  "drift_ratio": 0.42,
  "features_with_drift": 5,
  "feature_statistics": {
    "recency": {
      "test": "kolmogorov_smirnov",
      "p_value": 0.001,
      "drift_detected": true
    }
  }
}
```

---

## Estructura de Archivos

```
entrega2/
├── docker-compose.yml          # Orquestación de todos los servicios
├── .env                        # Configuración
│
├── airflow/                    # Pipeline ML
│   ├── Dockerfile
│   ├── dags/
│   │   ├── dag.py              # DAG principal
│   │   ├── load_and_preprocess.py
│   │   ├── pipeline.py         # Feature engineering
│   │   ├── drift_detector.py   # Drift detection
│   │   ├── train_module.py     # Entrenamiento + Optuna
│   │   └── predict_module.py   # Predicciones
│   ├── data/
│   │   ├── raw/                # Datos de entrada
│   │   ├── processed/          # Datos procesados
│   │   └── static/             # Datos estáticos (clientes, productos)
│   ├── models/                 # Modelos entrenados
│   ├── predictions/            # Predicciones generadas
│   ├── drift_reports/          # Reportes de drift
│   ├── mlflow_data/            # Experimentos MLflow
│   └── generate_test_data.py   # Script para generar datos de prueba
│
└── app/                        # Aplicación web
    ├── backend/                # FastAPI
    │   ├── Dockerfile
    │   ├── main.py             # API endpoints
    │   ├── model_loader.py     # Carga modelos desde MLflow
    │   ├── predictor.py        # Lógica de predicción
    │   └── recommender.py      # Sistema de recomendación
    └── frontend/               # Gradio
        ├── Dockerfile
        └── app.py              # UI interactiva
```

---

## Uso

### 1. Pipeline de Entrenamiento (Airflow)

```bash
# Generar datos de prueba (opcional)
docker-compose exec airflow python generate_test_data.py --mode add_weeks --weeks 2 --noise 0.3

# Ejecutar DAG vía UI
# http://localhost:8080 → sodai_prediction_pipeline → Trigger
```

### 2. Predicciones vía API

```bash
# Health check
curl http://localhost:8000/health

# Predicción individual
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 256017, "product_id": 84968}'

# Recomendaciones (Top-N productos)
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"customer_id": 256017, "top_n": 5}'
```

### 3. Predicciones vía Frontend

Abrir http://localhost:7860 y usar las pestañas:
- **Predicción Individual**: Seleccionar cliente y producto
- **Recomendaciones**: Seleccionar cliente y ver Top-N productos

---

## Generar Datos de Prueba

Para demostrar el pipeline:

```bash
# Ver datos actuales
python generate_test_data.py --mode summary

# Agregar nuevas semanas (simula datos nuevos)
python generate_test_data.py --mode add_weeks --weeks 2 --noise 0.3

# Restaurar datos originales
python generate_test_data.py --mode restore
```

Modos disponibles:
- `add_weeks`: Agrega semanas nuevas con ruido
- `sample`: Reduce tamaño del dataset
- `synthetic`: Genera datos sintéticos
- `restore`: Restaura backup
- `summary`: Muestra resumen de datos

---

## Resultados

### Datos Procesados
`data/processed/`:
- `final_data.parquet`: Universo completo
- `train_data.parquet`, `val_data.parquet`, `test_data.parquet`

Columnas: customer_id, product_id, week, customer_type, X, Y, num_deliver_per_week, brand, category, sub_category, segment, package, size, recency, frequency, customer_product_share, trend, cluster, bought

### Predicciones
`predictions/predictions_{date}.parquet`:
- customer_id, product_id, week, prediction (0/1), probability (0-1)

### MLflow
Experiments en http://localhost:5000:
- Optimización: 50 trials de Optuna
- Modelo final: métricas, parámetros, SHAP plots, confusion matrix
- Artifacts: modelo, gráficos

---

## Troubleshooting

### Puerto ocupado
```bash
# Ver qué usa el puerto
netstat -ano | findstr :8080

# Cambiar puerto en docker-compose.yml
ports:
  - "8081:8080"
```

### Servicios no inician
```bash
# Ver logs
docker-compose logs

# Reiniciar servicio específico
docker-compose restart airflow

# Limpiar y reiniciar
docker-compose down
docker-compose up -d
```

### Backend no encuentra modelo
```bash
# Verificar que modelo existe
ls airflow/models/best_model.pkl

# Si no existe, ejecutar DAG de Airflow primero
```

### DAG no aparece en Airflow
```bash
# Ver errores de importación
docker-compose exec airflow airflow dags list-import-errors

# Verificar sintaxis
docker-compose exec airflow python dags/dag.py
```

### MLflow unhealthy
```bash
# Reiniciar MLflow
docker-compose restart mlflow

# Esperar 30 segundos para health check
docker-compose ps
```

### Frontend no conecta con backend
```bash
# Verificar que backend está corriendo
curl http://localhost:8000/health

# Ver logs del frontend
docker-compose logs frontend
```

---

## Configuración

Editar `.env` para cambiar:
- Credenciales de base de datos
- Usuario admin de Airflow
- Hiperparámetros del modelo
- Límites de recursos

---

## Métricas del Modelo

**Objetivo principal:** Maximizar Recall (detectar compras)

**Métricas trackeadas:**
- Accuracy, Precision, Recall, F1-score
- AUC-PR (Area Under Precision-Recall Curve)
- Confusion Matrix
- SHAP Feature Importance

**Feature Engineering:**
- Clustering geográfico (KMeans, n=2)
- RFM: Recency, Frequency, Monetary (customer_product_share)
- Trend: cambio reciente en comportamiento

---

## Consideraciones de Producción

**Escalabilidad:**
- Usar Spark para datasets grandes
- Paralelizar Optuna trials con Dask

**Monitoring:**
- Performance del modelo en producción
- Tasa de drift detection
- Tiempo de ejecución de tareas

**Mejoras futuras:**
- Procesamiento incremental de datos
- A/B testing de modelos
- Feature store centralizado
- CI/CD con GitHub Actions

---

## Referencias

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**Última Actualización:** Noviembre 2025
**Versión:** 2.0
