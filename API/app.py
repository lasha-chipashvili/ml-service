from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict
from typing import List
import joblib
import numpy as np
from pathlib import Path

# ── Models ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

loaded_model_v1 = joblib.load(BASE_DIR / "ML" / "iris_model_v1.joblib")
loaded_model_v2 = joblib.load(BASE_DIR / "ML" / "iris_model_v2.joblib")

class_names = ["setosa", "versicolor", "virginica"]

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Iris ML Prediction API",
    description="Practical lecture API for REST design, validation, and versioned inference.",
    version="2.0.0"
)

# ── Schemas ───────────────────────────────────────────────────────────────────
_IRIS_EXAMPLE = {
    "sepal_length_cm": 5.1,
    "sepal_width_cm": 3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm": 0.2
}

class IrisInput(BaseModel):
    sepal_length_cm: float = Field(..., gt=0, lt=10, description="Sepal length in cm")
    sepal_width_cm:  float = Field(..., gt=0, lt=10, description="Sepal width in cm")
    petal_length_cm: float = Field(..., gt=0, lt=10, description="Petal length in cm")
    petal_width_cm:  float = Field(..., gt=0, lt=10, description="Petal width in cm")

    model_config = ConfigDict(json_schema_extra={"example": _IRIS_EXAMPLE})


class IrisPredictionV1(BaseModel):
    model_name: str
    model_version: str
    predicted_class_id: int
    predicted_class_name: str
    probabilities: List[float]


class IrisPredictionV2(BaseModel):
    model_name: str
    model_version: str
    predicted_class_id: int
    predicted_class_name: str
    probabilities: List[float]
    top_feature_importances: dict  # extra field only V2 can provide


# ── Helpers ───────────────────────────────────────────────────────────────────
_FEATURE_NAMES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]

def _input_to_array(data: IrisInput) -> np.ndarray:
    return np.array([[
        data.sepal_length_cm,
        data.sepal_width_cm,
        data.petal_length_cm,
        data.petal_width_cm,
    ]])


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Iris ML Prediction API",
        "available_versions": ["v1", "v2"],
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/predict", response_model=IrisPredictionV1, tags=["v1"])
def predict_v1(data: IrisInput):
    """Logistic Regression — returns class and probabilities."""
    features = _input_to_array(data)
    probabilities = loaded_model_v1.predict_proba(features)[0]
    predicted_id = int(np.argmax(probabilities))

    return {
        "model_name": "iris-logistic-regression",
        "model_version": "1.0.0",
        "predicted_class_id": predicted_id,
        "predicted_class_name": class_names[predicted_id],
        "probabilities": probabilities.round(4).tolist(),
    }


@app.post("/v2/predict", response_model=IrisPredictionV2, tags=["v2"])
def predict_v2(data: IrisInput):
    """Random Forest — returns class, probabilities, and feature importances."""
    features = _input_to_array(data)
    probabilities = loaded_model_v2.predict_proba(features)[0]
    predicted_id = int(np.argmax(probabilities))

    # Feature importances are on the classifier step inside the Pipeline
    importances = loaded_model_v2.named_steps["classifier"].feature_importances_
    top_feature_importances = {
        name: round(float(imp), 4)
        for name, imp in zip(_FEATURE_NAMES, importances)
    }

    return {
        "model_name": "iris-random-forest",
        "model_version": "2.0.0",
        "predicted_class_id": predicted_id,
        "predicted_class_name": class_names[predicted_id],
        "probabilities": probabilities.round(4).tolist(),
        "top_feature_importances": top_feature_importances,
    }


print("FastAPI app created.")