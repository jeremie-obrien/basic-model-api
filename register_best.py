import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, f1_score, precision_score

from experiments.data_split import X_test, y_test

EXPERIMENT_NAME = "iris-classifiers"
REGISTERED_MODEL_NAME = "iris-best-model"


def run_experiments():
    experiments_dir = Path(__file__).parent / "experiments"
    subprocess.run([sys.executable, "-m", "experiments.data_split"], check=True)
    for script in sorted(experiments_dir.glob("train_*.py")):
        module = f"experiments.{script.stem}"
        print(f"Running {module}...")
        subprocess.run([sys.executable, "-m", module], check=True)


client = MlflowClient()
experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    print(f"Experiment '{EXPERIMENT_NAME}' not found — running train scripts...")
    run_experiments()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.val_f1_score DESC"],
    max_results=1,
)
if not runs:
    print("No runs found — running train scripts...")
    run_experiments()
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.val_f1_score DESC"],
        max_results=1,
    )
if not runs:
    raise RuntimeError("No runs found even after running train scripts.")

best_run = runs[0]
best_run_id = best_run.info.run_id
val_f1 = best_run.data.metrics["val_f1_score"]
run_name = best_run.data.tags.get("mlflow.runName", best_run_id)

print(f"Best run: {run_name}  (val F1={val_f1:.4f})")

model_uri = f"runs:/{best_run_id}/model"

model = mlflow.sklearn.load_model(model_uri)

y_pred = model.predict(X_test)
test_f1 = f1_score(y_pred=y_pred, y_true=y_test, average="weighted")
test_precision = precision_score(y_pred=y_pred, y_true=y_test, average="weighted")
test_accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)

print(f"Test F1 (held-out):        {test_f1:.4f}")
print(f"Test precision (held-out): {test_precision:.4f}")
print(f"Test accuracy (held-out):  {test_accuracy:.4f}")

if test_precision < 0.9 or test_accuracy < 0.9:
    raise RuntimeError(
        f"Model did not meet quality gate: precision={test_precision:.4f}, accuracy={test_accuracy:.4f} (both must be ≥ 0.9)"
    )

joblib.dump(model, "model.pkl")
print("Saved to model.pkl")

mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
print(f"Registered as '{REGISTERED_MODEL_NAME}' version {mv.version}")
