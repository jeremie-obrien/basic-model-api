import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from experiments.data_split import X_train, X_val, y_train, y_val

EXPERIMENT_NAME = "iris-classifiers"

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="logistic_regression"):
    params = {"max_iter": 200, "random_state": 42}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    f1 = f1_score(y_val, model.predict(X_val), average="weighted")

    mlflow.log_params(params)
    mlflow.log_metric("val_f1_score", f1)
    mlflow.sklearn.log_model(model, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

    print(f"LogisticRegr   val F1: {f1:.4f}")
