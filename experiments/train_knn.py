import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

from experiments.data_split import X_train, X_val, y_train, y_val

EXPERIMENT_NAME = "iris-classifiers"

mlflow.set_experiment(EXPERIMENT_NAME)

with mlflow.start_run(run_name="knn"):
    params = {"n_neighbors": 5}
    model = KNeighborsClassifier(**params)
    model.fit(X_train, y_train)

    f1 = f1_score(y_val, model.predict(X_val), average="weighted")

    mlflow.log_params(params)
    mlflow.log_metric("val_f1_score", f1)
    mlflow.sklearn.log_model(model, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

    print(f"KNN            val F1: {f1:.4f}")
