# model-api

## Project Structure

```
experiments/
  train_rf.py       # RandomForest experiment → logs to MLflow
  train_xgb.py      # XGBoost experiment → logs to MLflow
  train_lr.py       # Logistic Regression experiment → logs to MLflow
  train_knn.py      # KNN experiment → logs to MLflow
register_best.py    # picks best val F1 run, evaluates on test set, writes model.pkl, registers in MLflow registry
main.py             # FastAPI app — loads model.pkl at startup, serves /predict and /health
test_main.py        # test suite (pytest + FastAPI TestClient)
model.pkl           # produced by register_best.py (don't write by hand)
mlruns/             # MLflow local tracking store (auto-created, gitignore this)
requirements.txt
Dockerfile
```

## Workflow

1. Run experiment scripts from the repo root:
   ```
   python -m experiments.train_rf
   python -m experiments.train_xgb
   python -m experiments.train_lr
   python -m experiments.train_knn
   ```

2. Register the best model (by val F1, then evaluates on held-out test set):
   ```
   python -m register_best
   ```

3. Start the API:
   ```
   uvicorn main:app --reload
   ```

4. Run the test suite:
   ```
   pytest test_main.py -v
   ```

## MLflow

- Experiment name: `iris-classifiers`
- Registered model name: `iris-best-model`
- Each `register_best.py` run adds a new version to the registry.
- View the tracking UI: `mlflow ui` → http://127.0.0.1:5000
- View experiment results (runs, metrics, params): http://127.0.0.1:5000/#/experiments/1

## API

- `GET  /health`   → `{"status": "ok"}`
- `POST /predict`  → takes four iris features (floats, all > 0), returns predicted class, class name, and confidence
- `GET  /docs`     → auto-generated Swagger UI

## Notes

- All scripts must be run from the repo root so MLflow writes mlruns/ in the right place.
- Experiment scripts use pickle serialization (`SERIALIZATION_FORMAT_PICKLE`) to avoid skops trust errors with XGBoost and KNN internal types.
- model.pkl is overwritten each time register_best.py runs; restart uvicorn to pick up a new model.
