# Conclusiones - Entrega 2: Pipeline MLOps para Predicción de Compras

## 1. Aprendizajes sobre MLOps

### 1.1 Orquestación con Airflow

La implementación de un pipeline completo con Apache Airflow nos permitió comprender la importancia de la **automatización y reproducibilidad** en sistemas de Machine Learning en producción. Algunos aprendizajes clave:

- **Modularización**: Separar el código en módulos independientes (preprocesamiento, feature engineering, entrenamiento, predicción) facilita el mantenimiento y testing.
- **Gestión de dependencias**: El uso de operadores en Airflow permite definir claramente el orden de ejecución y las dependencias entre tareas.
- **Branching condicional**: Implementar lógica de decisión (reentrenar o no según drift) hace que el pipeline sea adaptativo e inteligente.
- **Idempotencia**: Diseñar tareas que produzcan el mismo resultado si se ejecutan múltiples veces es crucial para robustez.

### 1.2 Tracking de Experimentos con MLflow

MLflow demostró ser una herramienta invaluable para:

- **Versionado de modelos**: Guardar cada versión del modelo con sus hiperparámetros y métricas permite comparar experimentos y realizar rollbacks si es necesario.
- **Reproducibilidad**: Registrar todos los parámetros de entrenamiento asegura que podamos recrear exactamente un modelo en el futuro.
- **Visualización de métricas**: Los gráficos de Optuna integrados en MLflow ayudan a entender el proceso de optimización de hiperparámetros.
- **Interpretabilidad**: Almacenar gráficos de SHAP junto al modelo facilita explicar las predicciones a stakeholders no técnicos.

### 1.3 Containerización con Docker

Docker resolvió múltiples desafíos:

- **Entorno consistente**: Elimina el problema de "funciona en mi máquina" al garantizar que todos trabajen con las mismas versiones de dependencias.
- **Aislamiento**: Cada servicio (Airflow, MLflow, Backend, Frontend) corre en su propio contenedor sin interferencias.
- **Escalabilidad**: Facilita el despliegue en la nube o en múltiples servidores si el sistema crece.
- **Volúmenes compartidos**: Permitió que la aplicación web acceda a los modelos entrenados por Airflow sin duplicar datos.

## 2. Desafíos del Despliegue

### 2.1 Desafíos Técnicos

1. **Consumo de memoria**:
   - **Problema**: El entrenamiento con XGBoost sobre datasets grandes consumía más de 8GB de RAM.
   - **Solución**: Implementar muestreo estratégico (20% de train, 30% de val) durante optimización de hiperparámetros, manteniendo 100% para el entrenamiento final.

2. **Feature engineering consistente**:
   - **Problema**: Las features calculadas en entrenamiento (recency, frequency, trend) debían replicarse exactamente en predicción.
   - **Solución**: Crear clases reutilizables (`FeatureEngineer`, `GeoClusterer`) que se integran en el pipeline de sklearn, garantizando consistencia.

3. **Detección de drift**:
   - **Problema**: Decidir cuándo reentrenar el modelo sin hacerlo innecesariamente.
   - **Solución**: Implementar tests estadísticos (KS-test, Chi-square) sobre features clave, con threshold del 30% de features con drift.

4. **Gestión de modelos**:
   - **Problema**: El backend debe cargar el modelo más reciente, pero MLflow puede no estar disponible siempre.
   - **Solución**: Estrategia de fallback: intentar MLflow primero, luego archivo local, asegurando alta disponibilidad.

### 2.2 Desafíos de Integración

1. **Comunicación entre contenedores**:
   - Configurar correctamente las redes de Docker para que el backend acceda a MLflow fue crucial.
   - Usar nombres de servicio (e.g., `http://mlflow:5000`) en lugar de IPs simplificó la configuración.

2. **Volúmenes compartidos**:
   - Montar correctamente los directorios de Airflow en el backend para acceder a datos y modelos requirió atención a permisos (read-only).

3. **Sincronización de datos**:
   - El frontend depende del backend, que depende de que Airflow haya ejecutado el pipeline al menos una vez.
   - Implementar health checks y mensajes de error claros mejoró la experiencia de usuario.

## 3. Valor de las Herramientas Utilizadas

### 3.1 Apache Airflow

**Aportes principales**:
- **Programación de ejecuciones**: Permite ejecutar el pipeline automáticamente (diario, semanal, mensual).
- **Monitoreo visual**: La UI muestra claramente el estado de cada tarea, facilitando debugging.
- **Gestión de errores**: Retry automático de tareas fallidas y alertas configurables.
- **Escalabilidad**: Podemos ejecutar tareas en paralelo o en múltiples workers en producción.

**Limitaciones encontradas**:
- Curva de aprendizaje inicial moderada.
- Overhead para pipelines muy simples (aunque este no es el caso).

### 3.2 MLflow

**Aportes principales**:
- **Centralización de experimentos**: Todos los experimentos en un solo lugar, fácil de comparar.
- **Modelo como artefacto**: Guardar el modelo completo (pipeline + metadatos) simplifica el deployment.
- **Integración con Optuna**: Los gráficos de optimización se registran automáticamente.
- **API simple**: Cargar un modelo es tan fácil como `mlflow.sklearn.load_model(uri)`.

**Limitaciones encontradas**:
- No incluye serving nativo robusto (por eso implementamos FastAPI).
- El backend de archivos no es ideal para producción de alta escala (se recomienda S3 o similar).

### 3.3 Docker

**Aportes principales**:
- **Portabilidad**: El proyecto se puede ejecutar en cualquier máquina con Docker instalado.
- **Versionado de infraestructura**: Los Dockerfiles y docker-compose son "infraestructura como código".
- **Desarrollo-producción similar**: Reduce significativamente bugs de deployment.

**Limitaciones encontradas**:
- Consumo de disco puede ser alto si no se hace limpieza de imágenes antiguas.
- En Windows, la compatibilidad con volúmenes puede requerir configuración adicional.

## 4. Mejoras Futuras

### 4.1 Mejoras Técnicas

1. **Optimización de rendimiento**:
   - Implementar caché de features pre-calculadas para acelerar predicciones.
   - Usar modelos más ligeros (LightGBM) para reducir latencia.
   - Paralelizar el cálculo de recomendaciones usando Dask o Ray.

2. **Robustez del sistema**:
   - Agregar logging estructurado (JSON) para facilitar análisis de errores.
   - Implementar rate limiting en el backend para prevenir sobrecarga.
   - Añadir tests unitarios y de integración automatizados.

3. **Monitoreo en producción**:
   - Integrar Prometheus + Grafana para monitorear métricas en tiempo real.
   - Detectar degradación del modelo (performance drift) comparando predicciones vs ventas reales.
   - Alertas automáticas si la API tiene alta tasa de errores.

### 4.2 Mejoras Funcionales

1. **Features adicionales**:
   - Incorporar información de stock y precios en el modelo.
   - Agregar estacionalidad (festivos, fines de semana) como features.
   - Usar embeddings de productos para capturar similitudes.

2. **Interfaz de usuario**:
   - Añadir visualizaciones de trends históricos por cliente/producto.
   - Implementar filtros por categoría, marca, región.
   - Exportar recomendaciones a CSV o PDF.

3. **Escalabilidad**:
   - Migrar backend a Kubernetes para auto-scaling.
   - Usar una base de datos (PostgreSQL) en lugar de archivos Parquet para queries más eficientes.
   - Implementar una cola de mensajes (RabbitMQ, Redis) para procesar predicciones batch asíncronamente.

### 4.3 Mejoras de Negocio

1. **A/B Testing**:
   - Implementar framework para probar nuevas versiones del modelo contra la versión actual.
   - Medir impacto real en ventas de las recomendaciones generadas.

2. **Personalización**:
   - Modelos específicos por segmento de cliente o región.
   - Incorporar feedback del usuario (¿fue útil esta recomendación?).

3. **Optimización de inventario**:
   - Usar predicciones para optimizar stock por zona geográfica.
   - Minimizar desperdicios prediciendo productos con baja demanda.

## 5. Reflexiones Finales

Este proyecto demostró que implementar un sistema MLOps completo es mucho más que entrenar un modelo:

- **Ingeniería > Modelado**: Aproximadamente 70% del tiempo se invierte en ingeniería de datos, deployment y monitoreo, vs 30% en modelado.
- **Mantenibilidad es clave**: Código limpio, modular y bien documentado es tan importante como la precisión del modelo.
- **Automatización ahorra tiempo**: La inversión inicial en crear un pipeline automatizado se paga rápidamente al evitar trabajo manual repetitivo.
- **Pensamiento end-to-end**: Es fundamental diseñar desde el inicio considerando todo el ciclo de vida del modelo, no solo el entrenamiento.

Este proyecto sienta las bases para un sistema de producción real, con las herramientas y buenas prácticas necesarias para escalar y mantener el sistema a largo plazo.

---

**Proyecto desarrollado para MDS7202 - Laboratorio de Programación Científica para Ciencia de Datos**
**Entrega 2: Pipelines Productivos y Aplicación Web**
**Fecha:** Noviembre 2025
