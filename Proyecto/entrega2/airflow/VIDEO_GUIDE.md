# Guía para Grabación del Video - Pipeline de Airflow

**Entrega 2 - MDS7202**
**Equipo:** Free Riders

---

## 📋 Requisitos del Video

Según el enunciado, el video debe:

✅ Mostrar la ejecución del pipeline de Airflow **de principio a fin**
✅ Ejecutar con un **nuevo conjunto de datos** de entrada
✅ Mostrar cómo los datos pasan por la **etapa de reentrenamiento**
✅ Subir a YouTube u otra plataforma y compartir link (no subirlo al repositorio)

**Duración recomendada:** 5-10 minutos

---

## 🎬 Estructura del Video

### Sección 1: Introducción (30-60 seg)
- Presentación del equipo
- Descripción breve del proyecto
- Objetivo del pipeline

### Sección 2: Preparación de Datos (1-2 min)
- Mostrar datos actuales
- Generar nuevos datos de prueba
- Explicar qué cambios se introdujeron

### Sección 3: Ejecución del DAG (3-5 min)
- Iniciar Airflow
- Activar y ejecutar el DAG
- Mostrar progreso de cada tarea
- Explicar qué hace cada paso

### Sección 4: Resultados (2-3 min)
- Mostrar drift report
- Mostrar que se reentrenó el modelo
- Mostrar predicciones generadas
- Mostrar experimentos en MLflow

### Sección 5: Cierre (30 seg)
- Resumen de lo demostrado
- Conclusiones

---

## 🛠️ Preparación Antes de Grabar

### 1. Verificar Instalación

```bash
# Verificar Python y paquetes
python --version
pip list | grep airflow
pip list | grep mlflow

# Verificar que el DAG está registrado
airflow dags list | grep sodai
```

### 2. Limpiar Estado Anterior

```bash
# Detener Airflow si está corriendo
# Ctrl+C en las terminales de scheduler y webserver

# Limpiar runs anteriores (opcional, para video limpio)
rm -rf mlruns/*
rm -rf drift_reports/*
rm -rf predictions/*

# Reinicializar DB de Airflow
airflow db reset
airflow db init

# Recrear usuario
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### 3. Preparar Ventanas

Tener abiertas y organizadas:
- **Ventana 1:** Terminal para comandos
- **Ventana 2:** Navegador con Airflow UI (http://localhost:8080)
- **Ventana 3:** Navegador con MLflow UI (http://localhost:5000) - opcional
- **Ventana 4:** Explorador de archivos mostrando carpeta del proyecto

---

## 🎥 Script de Grabación Paso a Paso

### SECCIÓN 1: Introducción (GRABANDO)

**[Pantalla: Terminal o presentación con título]**

```
🎤 NARRACIÓN:
"Hola, somos el equipo Free Riders y les presentamos nuestro pipeline
automatizado de Machine Learning para predicción de compras de productos.

Este pipeline utiliza Apache Airflow para orquestar todo el flujo, desde
la extracción de datos hasta la generación de predicciones. Incluye
detección automática de drift y reentrenamiento condicional del modelo.

Utilizamos MLflow para tracking de experimentos, Optuna para optimización
de hiperparámetros, y SHAP para interpretabilidad."
```

---

### SECCIÓN 2: Preparación de Datos

#### Paso 1: Mostrar Datos Actuales

**[Pantalla: Terminal]**

```bash
# Navegar al directorio
cd C:/Users/fmarq/DCC/MDS/MDS7202_Laboratorio/MDS7202_Free_Riders/Proyecto/entrega2/airflow

# Mostrar resumen de datos actuales
python generate_test_data.py --mode summary
```

**[Mostrar output en pantalla]**

```
🎤 NARRACIÓN:
"Primero, vamos a ver el estado actual de nuestros datos. Como pueden ver,
tenemos X clientes, Y productos, y Z transacciones.

Las transacciones van desde la semana [primera semana] hasta la semana
[última semana]."
```

#### Paso 2: Generar Nuevos Datos

**[Pantalla: Terminal]**

```bash
# Generar 2 nuevas semanas con ruido considerable
python generate_test_data.py --mode add_weeks --weeks 2 --noise 0.3
```

**[Mostrar output del script]**

```
🎤 NARRACIÓN:
"Ahora vamos a simular la llegada de datos nuevos. Agregamos 2 semanas
adicionales de transacciones con un factor de ruido del 30% para
introducir variabilidad.

Esto hará backup de los datos originales y creará nuevas transacciones
basadas en patrones recientes pero con variaciones aleatorias.

[ESPERAR A QUE TERMINE]

Como pueden ver, se agregaron [N] nuevas transacciones. Ahora tenemos
datos hasta la semana [nueva última semana]."
```

#### Paso 3: Verificar Nuevos Datos

```bash
# Ver resumen actualizado
python generate_test_data.py --mode summary
```

```
🎤 NARRACIÓN:
"Perfecto, ahora tenemos nuestro nuevo conjunto de datos que simula la
llegada de información de las siguientes semanas."
```

---

### SECCIÓN 3: Ejecución del DAG

#### Paso 1: Iniciar Airflow

**OPCIÓN A: Con Docker (RECOMENDADO) 🐳**

**[Pantalla: Terminal]**

```bash
# Navegar al directorio
cd C:/Users/fmarq/DCC/MDS/MDS7202_Laboratorio/MDS7202_Free_Riders/Proyecto/entrega2/airflow

# Iniciar todos los servicios con un solo comando
docker-compose up -d

# Ver logs en tiempo real (opcional)
docker-compose logs -f
```

```
🎤 NARRACIÓN:
"Iniciamos el pipeline completo con Docker. Con un solo comando,
docker-compose levanta Airflow, MLflow y PostgreSQL.

[ESPERAR ~30-60 segundos]

Los servicios están arrancando. Docker está creando los contenedores,
inicializando la base de datos de Airflow, y creando el usuario admin
automáticamente.

[Mostrar docker-compose ps para ver el estado]

Perfecto, todos los servicios están saludables. Ahora accedemos a la
interfaz web en localhost:8080."
```

**OPCIÓN B: Sin Docker (Manual)**

**[Pantalla: Terminal 1]**

```bash
# Terminal 1: Iniciar Scheduler
airflow scheduler
```

**[SPLIT SCREEN - Pantalla: Terminal 2]**

```bash
# Terminal 2: Iniciar Webserver
airflow webserver --port 8080
```

```
🎤 NARRACIÓN:
"Iniciamos Airflow manualmente. El scheduler coordina la ejecución de las tareas, y
el webserver nos da la interfaz gráfica para monitorear el pipeline.

[ESPERAR ~10-15 segundos hasta que inicie]

Ahora accedemos a la interfaz web en localhost:8080."
```

#### Paso 2: Acceder a Airflow UI

**[Pantalla: Navegador - http://localhost:8080]**

```
🎤 NARRACIÓN:
"Aquí está la interfaz de Airflow. Vamos a buscar nuestro DAG llamado
'sodai_prediction_pipeline'."
```

**[Acciones en video:]**
1. Login con admin/admin (si es necesario)
2. Buscar "sodai" en la barra de búsqueda
3. Localizar el DAG

#### Paso 3: Explicar Estructura del DAG

**[Pantalla: Click en el nombre del DAG → Tab "Graph"]**

```
🎤 NARRACIÓN:
"Este es el grafo de nuestro DAG. Como pueden ver, el flujo es:

1. START: Inicio del pipeline
2. EXTRACT NEW DATA: Validación de datos raw
3. PREPROCESS DATA: Limpieza y transformación
4. SPLIT DATA: División temporal en train/val/test
5. DETECT DRIFT: Análisis estadístico de cambios en distribuciones
6. DECIDE RETRAIN: Branching decision

En este punto, el flujo se divide:
- Si hay drift significativo → TRAIN MODEL (optimización + entrenamiento)
- Si no hay drift → SKIP RETRAIN (usar modelo existente)

Ambas ramas convergen en:
7. GENERATE PREDICTIONS: Predicciones para próxima semana
8. END: Fin del pipeline"
```

#### Paso 4: Activar y Ejecutar el DAG

**[Acciones en video:]**

1. **Activar el DAG:**
   - Toggle el switch de OFF a ON

2. **Trigger manual:**
   - Click en el botón "Trigger DAG" (icono ▶️ en la derecha)
   - Confirmar la ejecución

```
🎤 NARRACIÓN:
"Vamos a activar el DAG y ejecutarlo manualmente. Click en 'Trigger DAG'
y confirmamos.

[ESPERAR A QUE EMPIECE]

Excelente, la ejecución ha comenzado. Podemos ver el estado en tiempo real."
```

#### Paso 5: Monitorear Ejecución

**[Pantalla: Tab "Graph" o "Grid"]**

**Mientras se ejecuta, ir narrando:**

```
🎤 NARRACIÓN POR TAREA:

[CUANDO START → SUCCESS]
"El pipeline ha iniciado correctamente."

[CUANDO EXTRACT_NEW_DATA está running/success]
"La tarea de extracción está validando que los datos existan.
[Si quieres, click en la tarea → Log para mostrar los logs brevemente]
Como pueden ver, detectó los 3 archivos parquet: clientes, productos y
transacciones."

[CUANDO PREPROCESS_DATA está running]
"Ahora comienza el preprocesamiento. Esta tarea:
- Carga los datos raw
- Limpia transacciones (elimina duplicados, filtra items inválidos)
- Optimiza tipos de datos para eficiencia
- Crea la variable temporal 'week'
- Genera el universo completo de cliente × producto × semana

[Opcional: mostrar log brevemente]
Esta tarea puede tomar 30-60 segundos dependiendo del volumen de datos."

[CUANDO SPLIT_DATA está running]
"La división de datos respeta el orden temporal. 70% para entrenamiento,
15% validación, y 15% test. Esto previene data leakage."

[CUANDO DETECT_DRIFT está running - IMPORTANTE]
"Esta es una de las tareas clave: la detección de drift.

El sistema compara las distribuciones estadísticas de los datos nuevos
contra los datos históricos de entrenamiento.

Utiliza:
- Test de Kolmogorov-Smirnov para variables numéricas
- Test Chi-cuadrado para variables categóricas

Si más del 30% de las features monitoreadas muestran drift significativo,
se activa el reentrenamiento.

[Mostrar el log cuando termine para ver el resultado]
```

**[Click en detect_drift → Logs cuando termine]**

```
🎤 NARRACIÓN:
"Veamos el reporte de drift...

[LEER DEL LOG]
Como pueden ver, se detectó drift en [X] de [Y] features monitoreadas.
El ratio de drift es [Z]%, que excede el threshold del 30%.

Por lo tanto, la decisión es: REENTRENAR EL MODELO."
```

```
[CUANDO DECIDE_RETRAIN → TRAIN_MODEL (y no skip_retrain)]
"Perfecto, el branching funcionó correctamente. El sistema decidió
reentrenar porque detectó drift significativo."

[CUANDO TRAIN_MODEL está running - MUY IMPORTANTE]
"Esta es la tarea más intensiva del pipeline. Aquí sucede:

1. OPTIMIZACIÓN DE HIPERPARÁMETROS con Optuna:
   - 50 trials de búsqueda
   - Cada trial entrena un modelo XGBoost con diferentes parámetros
   - Se optimiza para maximizar Recall (detectar compras)

2. ENTRENAMIENTO DEL MODELO FINAL:
   - Se usa el mejor conjunto de hiperparámetros encontrados
   - Se aplica feature engineering: clustering geográfico, features RFM
   - Se balancea las clases con scale_pos_weight

3. INTERPRETABILIDAD:
   - Se generan SHAP values para explicar predicciones
   - Se crean gráficos de feature importance

4. TRACKING:
   - Todo se registra en MLflow: métricas, parámetros, gráficos, modelo

Esta tarea puede tomar entre 15 y 30 minutos con 50 trials de Optuna."
```

**[NOTA: Si el video es muy largo, puedes:]**

**Opción A:** Hacer fast-forward del entrenamiento
```
🎤 NARRACIÓN:
"Para no alargar el video, vamos a acelerar esta parte. El entrenamiento
está corriendo en segundo plano con los 50 trials de Optuna.

[FAST FORWARD en edición hasta que termine]

Y listo, el entrenamiento ha terminado después de [X] minutos."
```

**Opción B:** Reducir trials para el video
```python
# En dag.py cambiar temporalmente:
N_OPTUNA_TRIALS = 10  # En vez de 50
```

```
[CUANDO GENERATE_PREDICTIONS está running]
"Finalmente, generamos las predicciones para la próxima semana.

Esta tarea:
- Carga el mejor modelo (del entrenamiento o de MLflow)
- Identifica la última semana en los datos
- Crea el universo de cliente × producto para la semana siguiente
- Genera predicciones con probabilidades
- Guarda los resultados

Esto nos dice, para cada cliente y cada producto, cuál es la probabilidad
de que ese cliente compre ese producto la próxima semana."

[CUANDO END → SUCCESS]
"Excelente, el pipeline ha terminado exitosamente. Todas las tareas se
completaron correctamente."
```

---

### SECCIÓN 4: Resultados

#### Paso 1: Mostrar Drift Report

**[Pantalla: Explorador de archivos → drift_reports/]**

```bash
# O desde terminal
cat drift_reports/drift_report_*.json
```

```
🎤 NARRACIÓN:
"Veamos el reporte de drift generado.

[MOSTRAR JSON en pantalla]

Aquí podemos ver:
- El timestamp de la detección
- Qué features mostraron drift
- Los valores de los tests estadísticos
- La decisión final: needs_retrain = true"
```

#### Paso 2: Mostrar MLflow

**SI USASTE DOCKER:** MLflow ya está corriendo en http://localhost:5000 🎉

**SI NO USASTE DOCKER:**

**[Pantalla: Terminal]**

```bash
# En nueva terminal
cd Proyecto/entrega2/airflow
mlflow ui --backend-store-uri file:///C:/Users/fmarq/DCC/MDS/MDS7202_Laboratorio/MDS7202_Free_Riders/Proyecto/entrega2/airflow/mlruns
```

**[Pantalla: Navegador - http://localhost:5000]**

```
🎤 NARRACIÓN:
"Ahora veamos los experimentos registrados en MLflow.

[Si usas Docker, mencionar que MLflow UI ya estaba corriendo automáticamente]

[NAVEGAR EN MLFLOW UI]

Aquí podemos ver:
- Los 50 trials de optimización de Optuna [mostrar tabla de runs]
- Las métricas de cada trial: recall, precision, F1, AUC-PR
- Los hiperparámetros probados

[CLICK en el mejor run]

Este fue el mejor trial, con un recall de [X] en validación.

[NAVEGAR A ARTIFACTS]

Y aquí están todos los artefactos guardados:
- El modelo entrenado
- Gráficos de Optuna [mostrar optimization history]
- SHAP plots [mostrar summary plot]
- Confusion matrix
- Precision-Recall curve"
```

#### Paso 3: Mostrar Predicciones

**[Pantalla: Python/Pandas o Excel]**

```python
# En terminal Python o notebook
import pandas as pd

preds = pd.read_parquet('predictions/predictions_[fecha].parquet')
print(preds.head(20))
print(f"\nTotal predicciones: {len(preds):,}")
print(f"Predicciones positivas: {(preds['prediction'] == 1).sum():,}")
print(f"Tasa de compra predicha: {(preds['prediction'] == 1).mean():.2%}")

# Top 10 predicciones más probables
print("\nTop 10 compras más probables:")
print(preds.nlargest(10, 'probability')[['customer_id', 'product_id', 'probability']])
```

```
🎤 NARRACIÓN:
"Y por último, las predicciones generadas.

[MOSTRAR DATAFRAME]

Tenemos [N] predicciones en total, una para cada combinación de
cliente-producto para la semana [X].

El modelo predice que [M] de estas combinaciones resultarán en compras.

Aquí vemos las top 10 compras más probables según el modelo. Por ejemplo,
el cliente [ID] tiene un [XX]% de probabilidad de comprar el producto [ID]."
```

---

### SECCIÓN 5: Cierre

**[Pantalla: Resumen o conclusión]**

```
🎤 NARRACIÓN:
"Para resumir, demostramos nuestro pipeline completo de Airflow que:

✅ Procesa automáticamente datos nuevos
✅ Detecta drift usando tests estadísticos
✅ Decide de forma inteligente cuándo reentrenar
✅ Optimiza hiperparámetros con Optuna
✅ Trackea experimentos en MLflow
✅ Genera interpretabilidad con SHAP
✅ Produce predicciones para la próxima semana

El sistema está diseñado para producción: es robusto, modular, y
completamente automatizable.

Gracias por su atención. Somos el equipo Free Riders."
```

---

## 📌 Tips para Grabar

### Técnicos

1. **Resolución:** 1920x1080 (Full HD) mínimo
2. **Software de grabación:**
   - OBS Studio (gratis, recomendado)
   - Camtasia
   - Loom
   - Screen Studio (Mac)

3. **Audio:**
   - Usa un micrófono decente (no del laptop)
   - Graba en ambiente silencioso
   - Normaliza audio en edición

4. **Zoom:**
   - Haz zoom en elementos importantes (especialmente logs y JSON)
   - Usa shortcuts para no mostrar mouse innecesariamente

5. **Edición:**
   - Corta pausas largas
   - Fast-forward en el entrenamiento del modelo
   - Agrega títulos/captions para secciones
   - Música de fondo suave (opcional)

### De Contenido

1. **Ensaya antes de grabar**
   - Haz un dry-run completo
   - Cronometra cada sección
   - Ten un script escrito

2. **Mantén un ritmo ágil**
   - No te detengas mucho en una sola pantalla
   - Explica mientras haces las acciones
   - Evita silencios largos

3. **Destaca lo importante**
   - La detección de drift y decisión de reentrenamiento
   - El proceso de Optuna
   - Los resultados en MLflow
   - Las predicciones finales

4. **Sé profesional pero accesible**
   - Habla claro y pausado
   - Explica términos técnicos brevemente
   - Muestra entusiasmo por el proyecto

---

## ✅ Checklist Pre-Grabación

- [ ] Airflow instalado y funcionando
- [ ] MLflow instalado y funcionando
- [ ] Datos de prueba generados con `generate_test_data.py`
- [ ] Software de grabación configurado
- [ ] Micrófono probado
- [ ] Script de narración escrito
- [ ] Dry-run realizado
- [ ] Navegador sin tabs innecesarias
- [ ] Terminal con buen contraste (fondo oscuro, texto claro)
- [ ] Notificaciones del sistema desactivadas
- [ ] Todo lo demás cerrado (Slack, email, etc.)

---

## 📤 Después de Grabar

1. **Editar:**
   - Cortar partes lentas
   - Agregar títulos de sección
   - Agregar captions si es necesario
   - Normalizar audio

2. **Exportar:**
   - Formato: MP4
   - Codec: H.264
   - Resolución: 1920x1080
   - Bitrate: 5-8 Mbps

3. **Subir a YouTube:**
   - Crear cuenta de YouTube si no tienes
   - Subir como "Unlisted" (no público, pero accesible por link)
   - Título: "Pipeline de Airflow - Predicción de Compras - Equipo Free Riders"
   - Descripción: Incluir link al repositorio GitHub

4. **Compartir:**
   - Copiar link del video
   - Agregarlo al README.md
   - Incluirlo en la entrega

---

## 🎯 Estructura Recomendada del Video

| Sección | Duración | Contenido Clave |
|---------|----------|-----------------|
| Intro | 0:00 - 0:45 | Presentación, objetivo |
| Datos | 0:45 - 2:00 | Generar datos nuevos, explicar cambios |
| DAG | 2:00 - 6:00 | Ejecutar pipeline, explicar cada tarea |
| Drift | 6:00 - 7:00 | Mostrar drift report, decisión de retrain |
| MLflow | 7:00 - 8:30 | Experimentos, gráficos, modelo |
| Predicciones | 8:30 - 9:00 | Mostrar resultados |
| Cierre | 9:00 - 9:30 | Resumen |

**Total:** ~9-10 minutos

---

## 🚨 Troubleshooting

### El DAG no aparece en Airflow
```bash
# Verificar que el archivo está en dags/
ls dags/dag.py

# Verificar sintaxis
python dags/dag.py

# Refrescar DAGs
airflow dags list-import-errors
```

### El entrenamiento toma mucho tiempo
```python
# Reducir trials temporalmente en dag.py
N_OPTUNA_TRIALS = 10  # En vez de 50
```

### MLflow no muestra experimentos
```bash
# Verificar que mlruns existe
ls mlruns/

# Iniciar con tracking URI correcto
mlflow ui --backend-store-uri file:///[path_completo]/mlruns
```

---

**¡Buena suerte con el video!** 🎬

*Free Riders Team - MDS7202*
