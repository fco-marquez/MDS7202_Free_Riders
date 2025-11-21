# 🚀 Quick Start - Automated Pipeline

## TL;DR - How to Use

### 1. Start the Pipeline
```bash
cd entrega3
docker-compose up -d
```

### 2. Drop New Batch Files Here
```bash
cp your_batch_file.parquet airflow/data/incoming/
```

### 3. Pipeline Auto-Processes
- Detects file within 30 seconds
- Processes data automatically
- Generates predictions
- Saves to `predictions/` folder

### 4. Get Predictions
```bash
ls predictions/
# Find your predictions_YYYY-MM-DD.parquet file
```

## 📍 Key Locations

| What | Where |
|------|-------|
| **Drop new data** | `airflow/data/incoming/` |
| **Get predictions** | `predictions/` |
| **View metrics** | MLflow UI: http://localhost:5000 |
| **Monitor pipeline** | Airflow UI: http://localhost:8080 |

## ⚡ One-Liner Per Batch

```bash
# Batch 1 (Dec 2)
cp batch_1.parquet airflow/data/incoming/ && echo "✓ Processing batch 1..."

# Batch 2 (Dec 3)
cp batch_2.parquet airflow/data/incoming/ && echo "✓ Processing batch 2..."

# Batch 3 (Dec 4)
cp batch_3.parquet airflow/data/incoming/ && echo "✓ Processing batch 3..."

# Batch 4 (Dec 5)
cp batch_4.parquet airflow/data/incoming/ && echo "✓ Processing batch 4..."
```

## 🎯 That's It!
No manual triggers needed. The pipeline watches for files and processes them automatically.

---
For detailed docs, see `AUTOMATED_PIPELINE_GUIDE.md`
