import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


VALID_SAMPLES = [
    (5.1, 3.5, 1.4, 0.2),   # setosa-region
    (6.0, 2.7, 5.1, 1.6),   # versicolor-region
    (6.3, 3.3, 6.0, 2.5),   # virginica-region
]


@pytest.mark.parametrize("sl,sw,pl,pw", VALID_SAMPLES)
def test_predict_valid(client, sl, sw, pl, pw):
    response = client.post("/predict", json={
        "sepal_length": sl,
        "sepal_width": sw,
        "petal_length": pl,
        "petal_width": pw,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_class"] in {0, 1, 2}
    assert body["class_name"] in {"setosa", "versicolor", "virginica"}
    assert 0.0 < body["confidence"] <= 1.0


@pytest.mark.parametrize("payload", [
    {"sepal_length": "bad", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": -1.0, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 0.0,  "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
    {"sepal_length": 5.1,  "sepal_width": 3.5, "petal_length": 1.4},
])
def test_predict_invalid(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
