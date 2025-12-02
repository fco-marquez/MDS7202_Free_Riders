# SODAI Drinks - Aplicación Web de Predicción

Aplicación web completa para predicción de compras y recomendación de productos, construida con FastAPI y Gradio.

## Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                  SODAI Drinks App                        │
│                                                          │
│  ┌──────────────────┐         ┌──────────────────┐    │
│  │   Frontend       │────────▶│   Backend        │    │
│  │   (Gradio)       │  HTTP   │   (FastAPI)      │    │
│  │   Port 7860      │         │   Port 8000      │    │
│  └──────────────────┘         └──────────────────┘    │
│                                        │                │
│                                        ▼                │
│                    ┌────────────────────────────────┐  │
│                    │  Datos y Modelos de Airflow    │  │
│                    │  - /data (raw, processed)       │  │
│                    │  - /models (best_model.pkl)     │  │
│                    │  - /mlflow (experiments)        │  │
│                    └────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Características

### Backend (FastAPI)

- **POST /predict**: Predicción individual (customer_id + product_id)
- **POST /recommend**: Top N recomendaciones para un cliente
- **GET /health**: Estado del servicio y modelo
- **GET /model/info**: Información del modelo cargado
- **POST /model/reload**: Recargar modelo (útil después de reentrenamiento)
- **GET /customers/sample**: Obtener IDs de ejemplo de clientes
- **GET /products/sample**: Obtener IDs de ejemplo de productos

#### Estrategia de Carga de Modelo

El backend implementa una estrategia de fallback robusta:

1. **Intenta cargar desde MLflow** (`http://mlflow:5000`)
   - Busca el mejor modelo por métrica `val_recall`
   - Carga modelo completo con metadatos
2. **Si falla, carga desde archivo local** (`/models/best_model.pkl`)
3. **Si ambos fallan, retorna error** con mensaje descriptivo

### Frontend (Gradio)

#### Tab 1: Predicción Individual
- Input: Customer ID, Product ID
- Output: Predicción (comprará / no comprará), probabilidad, detalles

#### Tab 2: Recomendaciones
- Input: Customer ID, número de recomendaciones (1-20)
- Output: Tabla con Top N productos ordenados por probabilidad

## Requisitos Previos

1. **Airflow pipeline ejecutado al menos una vez**:
   - Debe existir `airflow/data/raw/` con datos
   - Debe existir `airflow/data/processed/final_data.parquet`
   - Debe existir `airflow/models/best_model.pkl` o experimentos en MLflow

2. **Red Docker de Airflow**:
   - La red `sodai_network` debe existir (creada por docker-compose de Airflow)

3. **Docker y Docker Compose** instalados

## Instalación y Ejecución

### Opción 1: Usando Docker Compose (Recomendado)

```bash
# Navegar a la carpeta app/
cd app/

# Levantar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

### Opción 2: Ejecución Local (Desarrollo)

#### Backend

```bash
cd app/backend/

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
export MLFLOW_TRACKING_URI=http://localhost:5000
export MODEL_PATH=../../airflow/models/best_model.pkl

# Ejecutar
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

#### Frontend

```bash
cd app/frontend/

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
export BACKEND_URL=http://localhost:8000

# Ejecutar
python app.py
```

## Acceso

Una vez ejecutándose:

- **Frontend (Gradio)**: http://localhost:7860
- **Backend (FastAPI)**: http://localhost:8000
- **Documentación API (Swagger)**: http://localhost:8000/docs
- **Documentación API (ReDoc)**: http://localhost:8000/redoc

## Estructura de Archivos

```
app/
├── backend/
│   ├── main.py              # Aplicación FastAPI
│   ├── model_loader.py      # Carga de modelo (MLflow + fallback)
│   ├── predictor.py         # Lógica de predicción individual
│   ├── recommender.py       # Sistema de recomendación (Top N)
│   ├── requirements.txt     # Dependencias Python
│   └── Dockerfile           # Imagen Docker
│
├── frontend/
│   ├── app.py               # Interfaz Gradio
│   ├── requirements.txt     # Dependencias Python
│   └── Dockerfile           # Imagen Docker
│
├── docker-compose.yml       # Orquestación de servicios
└── README.md                # Esta documentación
```

## Variables de Entorno

### Backend

| Variable | Descripción | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | URI del servidor MLflow | `http://mlflow:5000` |
| `MLFLOW_EXPERIMENT_NAME` | Nombre del experimento | `sodai_drinks_prediction` |
| `MODEL_PATH` | Ruta al modelo local | `/models/best_model.pkl` |

### Frontend

| Variable | Descripción | Default |
|----------|-------------|---------|
| `BACKEND_URL` | URL del backend FastAPI | `http://backend:8000` |

## Uso de la API

### Ejemplo: Predicción Individual

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 1001,
    "product_id": 2001
  }'
```

Respuesta:
```json
{
  "customer_id": 1001,
  "product_id": 2001,
  "prediction": 1,
  "probability": 0.8523,
  "week": 53,
  "customer_type": "PREMIUM",
  "product_brand": "COCA-COLA",
  "product_category": "GASEOSAS"
}
```

### Ejemplo: Recomendaciones

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 1001,
    "top_n": 5
  }'
```

Respuesta:
```json
{
  "customer_id": 1001,
  "total_recommendations": 5,
  "recommendations": [
    {
      "rank": 1,
      "product_id": 2015,
      "probability": 0.9234,
      "brand": "COCA-COLA",
      "category": "GASEOSAS",
      "sub_category": "COLA",
      "segment": "PREMIUM",
      "package": "BOTELLA PET",
      "size": 2.5
    },
    ...
  ]
}
```

## Troubleshooting

### Error: "Model not loaded"

**Causa**: El pipeline de Airflow no ha ejecutado o no hay modelo guardado.

**Solución**:
1. Ejecutar el DAG de Airflow al menos una vez
2. Verificar que existe `airflow/models/best_model.pkl`
3. O verificar que hay experimentos en MLflow

### Error: "Customer/Product not found"

**Causa**: El ID proporcionado no existe en los datos.

**Solución**:
1. Usar el endpoint `/customers/sample` para obtener IDs válidos
2. Usar el endpoint `/products/sample` para obtener IDs válidos
3. En la UI de Gradio, usar el botón "Obtener ejemplos"

### Error: "Connection refused" (backend → MLflow)

**Causa**: La red Docker no está configurada correctamente o MLflow no está ejecutándose.

**Solución**:
1. Verificar que docker-compose de Airflow está ejecutándose
2. Verificar que el contenedor MLflow está saludable: `docker ps | grep mlflow`
3. El backend tiene fallback a archivo local, debería funcionar igual

### Frontend no se conecta al backend

**Causa**: Variable de entorno `BACKEND_URL` incorrecta.

**Solución**:
- En Docker: Debe ser `http://backend:8000` (nombre del servicio)
- En local: Debe ser `http://localhost:8000`

## Integración con Airflow

La aplicación comparte datos con Airflow mediante volúmenes Docker:

```yaml
volumes:
  - ../airflow/data:/data:ro          # Datos (read-only)
  - ../airflow/models:/models:ro      # Modelos (read-only)
  - ../airflow/mlflow_data:/mlflow:ro # MLflow (read-only)
```

**Importante**: Los volúmenes son de solo lectura (`:ro`) para evitar que la aplicación modifique los datos de entrenamiento.

### Workflow Completo

1. **Airflow ejecuta el pipeline**:
   - Preprocesa datos
   - Detecta drift
   - Entrena modelo (si es necesario)
   - Guarda modelo en `/models/` y MLflow

2. **Backend carga el modelo**:
   - Automáticamente al iniciar
   - Manualmente con `POST /model/reload`

3. **Usuario interactúa con la aplicación**:
   - Hace predicciones individuales
   - Obtiene recomendaciones

## Escalabilidad

### Para Producción

1. **Usar base de datos en lugar de Parquet**:
   ```python
   # Reemplazar pd.read_parquet con queries SQL
   engine = create_engine("postgresql://...")
   df = pd.read_sql("SELECT * FROM customers", engine)
   ```

2. **Caché de predicciones**:
   - Usar Redis para cachear predicciones frecuentes
   - Reducir latencia para requests repetidos

3. **Asincronía**:
   - Usar `async def` en endpoints de FastAPI
   - Procesar batch predictions en background con Celery

4. **Load Balancing**:
   - Desplegar múltiples instancias del backend
   - Usar nginx o Traefik como reverse proxy

5. **Monitoreo**:
   - Agregar Prometheus para métricas
   - Grafana para dashboards
   - Sentry para tracking de errores

## Tecnologías Utilizadas

- **Backend**: FastAPI, Pydantic, MLflow, scikit-learn, XGBoost
- **Frontend**: Gradio, Requests
- **Containerización**: Docker, Docker Compose
- **Modelo**: XGBoost con pipeline de sklearn (GeoClusterer + FeatureEngineer + Preprocessor)

## Créditos

Proyecto desarrollado para **MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos**

**Entrega 2**: Pipelines Productivos y Aplicación Web

---

Para más información sobre el pipeline de Airflow, consultar el README principal en la raíz del proyecto.
