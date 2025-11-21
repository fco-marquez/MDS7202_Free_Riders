# 🤖 Automated Pipeline Guide - Entrega 3

## Overview
The pipeline has been configured to **automatically process new batch data** as soon as it arrives. No manual intervention needed!

## 📂 Directory Structure

```
entrega3/
├── airflow/
│   └── data/
│       ├── incoming/          ← 📥 DROP NEW BATCH FILES HERE
│       ├── raw/               ← Files automatically moved here
│       ├── processed/         ← Processed data
│       └── static/            ← Reference data (clientes, productos)
├── predictions/               ← Generated predictions
└── data/                      ← Realized batch data (for analysis)
```

## 🚀 How It Works

### Automatic Processing Flow

```
1. Place new batch file → airflow/data/incoming/
                                    ↓
2. FileSensor detects new .parquet file (checks every 30 seconds)
                                    ↓
3. File automatically moved → airflow/data/raw/
                                    ↓
4. Preprocessing pipeline runs
                                    ↓
5. Drift detection performed
                                    ↓
6. Decision: Retrain or Use Existing Model
                                    ↓
7. Predictions generated → predictions/
```

### Step-by-Step Instructions

#### **For Each Batch (Dec 1-4)**

1. **Receive batch data file** (e.g., `batch_1.parquet`, `batch_2.parquet`, etc.)

2. **Copy file to incoming directory:**
   ```bash
   cp batch_1.parquet airflow/data/incoming/
   ```

3. **That's it!** The pipeline will:
   - ✅ Detect the new file within 30 seconds
   - ✅ Move it to raw/
   - ✅ Process and clean the data
   - ✅ Check for drift vs baseline
   - ✅ Retrain model if drift detected (>30% features)
   - ✅ Generate predictions for next week
   - ✅ Save predictions with timestamp

4. **Monitor progress:**
   - Check Airflow UI at `http://localhost:8080`
   - View logs for each task
   - Check MLflow UI at `http://localhost:5000` for metrics

5. **Retrieve predictions:**
   ```bash
   ls predictions/
   # predictions_2025-12-01.parquet
   # predictions_2025-12-02.parquet
   # etc.
   ```

## 📅 Batch Schedule for Entrega 3

| Date | Action | Batch File | Prediction Period |
|------|--------|-----------|-------------------|
| **Dec 1** | Submit predictions #1 | (use entrega1 data) | 01/01/25 - 05/01/25 |
| **Dec 2** | Receive batch_1.parquet | Drop in incoming/ | 06/01/25 - 12/01/25 |
| **Dec 2** | Submit predictions #2 | Process batch 1 | 06/01/25 - 12/01/25 |
| **Dec 3** | Receive batch_2.parquet | Drop in incoming/ | 13/01/25 - 19/01/25 |
| **Dec 3** | Submit predictions #3 | Process batch 2 | 13/01/25 - 19/01/25 |
| **Dec 4** | Receive batch_3.parquet | Drop in incoming/ | 20/01/25 - 26/01/25 |
| **Dec 4** | Submit predictions #4 | Process batch 3 | 20/01/25 - 26/01/25 |
| **Dec 5** | Receive batch_4.parquet | Drop in incoming/ | (for analysis only) |

## 🔧 Configuration

### DAG Settings
- **Schedule:** Runs daily at midnight
- **FileSensor:** Checks every 30 seconds
- **Timeout:** 7 days max wait
- **Mode:** Reschedule (doesn't block scheduler)
- **Max Active Runs:** 1 (prevents overlap)

### Drift Detection
- **Threshold:** 30% of features showing drift
- **Tests Used:**
  - Kolmogorov-Smirnov (numerical features)
  - Chi-Square (categorical features)
- **p-value:** < 0.05 indicates drift

### Model Training
- **Optimizer:** Optuna (50 trials)
- **Model:** XGBoost
- **Split:** 80% train / 20% validation
- **Metric:** F1-score
- **Tracking:** MLflow

## 🎯 What Gets Generated

For each batch processed:

1. **Drift Report** → `airflow/drift_reports/drift_report_YYYY-MM-DD.json`
   - Statistical test results
   - Features with significant drift
   - Recommendation: retrain or skip

2. **Updated Model** (if retrained) → `airflow/models/best_model.pkl`
   - New XGBoost model
   - Hyperparameters logged in MLflow
   - Performance metrics

3. **Predictions** → `predictions/predictions_YYYY-MM-DD.parquet`
   - Columns: `customer_id`, `product_id`, `week`, `prediction`, `probability`
   - Ready for CodaLab submission

4. **MLflow Artifacts**
   - Confusion matrix
   - Feature importance (SHAP)
   - Metrics (Accuracy, Precision, Recall, F1, AUC-PR)

## 🚨 Troubleshooting

### Pipeline not detecting files?
```bash
# Check if DAG is running
docker exec sodai_airflow airflow dags list | grep sodai

# Check FileSensor status
docker logs sodai_airflow | grep "wait_for_new_batch"
```

### Files stuck in incoming/?
```bash
# Manually trigger the DAG
docker exec sodai_airflow airflow dags trigger sodai_prediction_pipeline
```

### Need to reset and reprocess?
```bash
# Clear failed tasks
docker exec sodai_airflow airflow tasks clear sodai_prediction_pipeline -y

# Or restart Airflow
docker restart sodai_airflow
```

## 📊 Monitoring

### Airflow UI
- URL: `http://localhost:8080`
- Login: `admin` / `admin`
- View DAG runs, task logs, and execution times

### MLflow UI
- URL: `http://localhost:5000`
- View experiments, compare runs, download models

### Logs
```bash
# Airflow logs
docker logs -f sodai_airflow

# Specific task logs
docker exec sodai_airflow airflow tasks logs sodai_prediction_pipeline wait_for_new_batch <execution_date>
```

## 🎓 Tips for Informe.ipynb

When analyzing results in the notebook:

1. **Load predictions:**
   ```python
   import pandas as pd
   pred1 = pd.read_parquet('predictions/predictions_2025-12-01.parquet')
   pred2 = pd.read_parquet('predictions/predictions_2025-12-02.parquet')
   # etc.
   ```

2. **Load drift reports:**
   ```python
   import json
   with open('airflow/drift_reports/drift_report_2025-12-02.json') as f:
       drift1 = json.load(f)
   ```

3. **Query MLflow:**
   ```python
   import mlflow
   mlflow.set_tracking_uri('http://localhost:5000')
   runs = mlflow.search_runs(experiment_names=['sodai_drinks_prediction'])
   ```

## 📝 Notes

- **First prediction (Dec 1)** uses the model trained in entrega2 (no new data yet)
- **Subsequent predictions** use either the existing model or retrained model based on drift
- All artifacts are saved and tracked for reproducibility
- The pipeline handles missing data, duplicates, and edge cases automatically

---

**Team:** Free Riders
**Course:** MDS7202 - Laboratorio de Programación Científica
**Project:** SodAI Drinks Prediction Pipeline
