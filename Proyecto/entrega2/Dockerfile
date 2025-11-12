# Dockerfile for SodAI Drinks Prediction Pipeline
# ================================================
# This Dockerfile creates a complete environment for running the Airflow pipeline
# with MLflow, Optuna, XGBoost, and SHAP integration.

FROM apache/airflow:2.10.3-python3.11

# Set user to root for installations
USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# Set working directory
WORKDIR /opt/airflow

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy DAGs and modules
COPY --chown=airflow:root ./airflow/dags /opt/airflow/dags

# Copy utility scripts
COPY --chown=airflow:root generate_test_data.py /opt/airflow/

# Create necessary directories with correct permissions
RUN mkdir -p \
    /opt/airflow/data/raw \
    /opt/airflow/data/processed \
    /opt/airflow/data/backup \
    /opt/airflow/mlruns \
    /opt/airflow/models \
    /opt/airflow/predictions \
    /opt/airflow/drift_reports \
    /opt/airflow/logs

# Set environment variables
ENV AIRFLOW_HOME=/opt/airflow
ENV PYTHONPATH=/opt/airflow/dags:$PYTHONPATH
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False
ENV AIRFLOW__WEBSERVER__EXPOSE_CONFIG=True
ENV AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD airflow jobs check --job-type SchedulerJob --hostname "$${HOSTNAME}"

# Default command (will be overridden by docker-compose)
CMD ["airflow", "webserver"]
