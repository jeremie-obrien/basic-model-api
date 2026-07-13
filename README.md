# model-api

A small end-to-end ML demo: train several scikit-learn/XGBoost classifiers on the Iris dataset, track and compare them with MLflow, promote the best one to a registered model, and serve it behind a FastAPI prediction endpoint with a pytest test suite.

## What this demonstrates

- **scikit-learn (+ XGBoost)** — four classifiers (Random Forest, XGBoost, Logistic Regression, KNN) trained on the same data for comparison.
- **Train/val/test split** — a single 60/20/20 split (`experiments/data_split.py`) shared by every experiment, so models are compared fairly and the test set stays held out until final evaluation.
- **MLflow** — each training run logs params, metrics, and the model artifact to a local MLflow tracking store; the best run (by validation F1) is evaluated on the test set and registered as a new model version.
- **FastAPI** — a `/predict` endpoint loads the registered model and serves live predictions, plus a `/health` check and auto-generated Swagger docs.
- **pytest** — a test suite exercises the API with valid and invalid inputs using FastAPI's `TestClient`.

## Project structure

```
experiments/
  data_split.py       # loads Iris, produces the shared train/val/test split
  train_rf.py         # Random Forest experiment        -> logs to MLflow
  train_xgb.py        # XGBoost experiment              -> logs to MLflow
  train_lr.py         # Logistic Regression experiment  -> logs to MLflow
  train_knn.py        # KNN experiment                  -> logs to MLflow
register_best.py     # picks best val F1 run, evaluates on test set, writes model.pkl, registers in MLflow
main.py              # FastAPI app - loads model.pkl at startup, serves /predict and /health
test_main.py         # pytest suite (FastAPI TestClient)
model.pkl            # produced by register_best.py (generated, don't edit by hand)
mlruns/              # MLflow local tracking store (auto-created)
requirements.txt
Dockerfile
```

## Setup

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

All commands below should be run from the repo root.

## Usage

1. **Run the experiments** — each trains a model and logs it to MLflow:

   ```
   python -m experiments.train_rf
   python -m experiments.train_xgb
   python -m experiments.train_lr
   python -m experiments.train_knn
   ```

2. **Register the best model** — picks the highest validation F1 run, evaluates it on the held-out test set, and (if it clears the quality gate) writes `model.pkl` and registers a new MLflow model version:

   ```
   python -m register_best
   ```

3. **Start the API:**

   ```
   uvicorn main:app --reload
   ```

   - `GET /health` → `{"status": "ok"}`
   - `POST /predict` → takes four Iris features (`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, all floats > 0) and returns the predicted class, class name, and confidence
   - `GET /docs` → interactive Swagger UI

4. **Run the tests:**

   ```
   pytest test_main.py -v
   ```

5. **Browse experiment results in MLflow:**

   ```
   mlflow ui
   ```

   Then open http://127.0.0.1:5000 to compare runs, metrics, and registered model versions.

## Docker

```
docker build -t model-api .
docker run -p 8000:8000 model-api
```

Note: the image copies the existing `model.pkl`, so run `register_best.py` locally before building.
