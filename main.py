from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

CLASS_NAMES = ["setosa", "versicolor", "virginica"]


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)


class IrisPrediction(BaseModel):
    predicted_class: int
    class_name: str
    confidence: float


model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = joblib.load("model.pkl")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=IrisPrediction)
def predict(features: IrisFeatures):
    X = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width,
    ]])
    predicted_class = int(model.predict(X)[0])
    confidence = float(model.predict_proba(X)[0][predicted_class])
    return IrisPrediction(
        predicted_class=predicted_class,
        class_name=CLASS_NAMES[predicted_class],
        confidence=confidence,
    )
