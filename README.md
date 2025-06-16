# 🚀 MLflow + DagsHub | MLOps Practice

This project shows complete ML experiment tracking using **MLflow** with local and remote (DagsHub) setups. It includes classification models on Wine and Breast Cancer datasets, autologging, manual logging, remote logging with DagsHub, and hyperparameter tuning with nested runs.

### 📦 Files Included:
- `wine_autolog.py`: MLflow autologging example (local)
- `wine_manual_logging.py`: Manual logging with metrics, params, plots, and model
- `wine_dagshub.py`: Remote tracking using DagsHub
- `hypertune.py`: GridSearchCV + nested MLflow logging + dataset + best model

---

## ⚙️ Setup

1. Install requirements:
```bash
pip install mlflow dagshub scikit-learn pandas matplotlib seaborn
```

2. For local MLflow tracking:
```bash
mlflow ui
```
Open: http://127.0.0.1:5000

3. For DagsHub remote logging (already configured):
```python
mlflow.set_tracking_uri("https://dagshub.com/AnandVadgama/MLFlow-MLOps-prac.mlflow")
dagshub.init(repo_owner="AnandVadgama", repo_name="MLFlow-MLOps-prac", mlflow=True)
```

---

## 🔁 Flow Overview

- **Wine Dataset – Autologging** (`wine_autolog.py`)  
  Uses `mlflow.autolog()` to log everything automatically.

- **Wine Dataset – Manual Logging** (`wine_manual_logging.py`)  
  Logs accuracy, parameters, confusion matrix plot, model, and tags manually.

- **Wine Dataset – DagsHub Logging** (`wine_dagshub.py`)  
  Logs all artifacts to DagsHub. Fully integrated with GitHub repo.

- **Breast Cancer – Hyperparameter Tuning** (`hypertune.py`)  
  Uses GridSearchCV with MLflow nested runs. Logs each trial, best model, dataset, and final score.

---