# model-api

See [README.md](README.md) for project structure, setup, and usage.

## Notes for agents

- Run all scripts from the repo root — MLflow writes `mlruns/` relative to cwd.
- Experiment scripts log models with `SERIALIZATION_FORMAT_PICKLE` to avoid skops trust errors with XGBoost/KNN internal types.
- `register_best.py` picks the best run by val F1, then evaluates it on the held-out test set and enforces a quality gate (precision & accuracy ≥ 0.9) before writing `model.pkl` and registering a new version.
- `model.pkl` is overwritten each time `register_best.py` runs; restart uvicorn to pick up a new model.
- MLflow experiment name: `iris-classifiers`; registered model name: `iris-best-model`.
