from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os

# ── Data ──────────────────────────────────────────────────────────────────────
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names.tolist()
feature_names = iris.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

MODEL_DIR = os.path.dirname(__file__)

# ── V1 — Logistic Regression ──────────────────────────────────────────────────
print("=" * 50)
print("Model V1 — Logistic Regression")
print("=" * 50)

model_v1 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])
model_v1.fit(X_train, y_train)

y_pred_v1 = model_v1.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_v1))
print(classification_report(y_test, y_pred_v1, target_names=class_names))


joblib.dump(model_v1, os.path.join(MODEL_DIR, "iris_model_v1.joblib"))
metadata_v1 = {
    "model_name": "iris-logistic-regression",
    "model_version": "1.0.0",
    "features": feature_names,
    "classes": class_names,
    "accuracy": round(accuracy_score(y_test, y_pred_v1), 4)
}

with open(os.path.join(MODEL_DIR, "iris_model_v1_metadata.json"), "w") as f:
    json.dump(metadata_v1, f, indent=2)

print("V1 saved → iris_model_v1.joblib + iris_model_v1_metadata.json\n")

# ── V2 — Random Forest ────────────────────────────────────────────────────────
print("=" * 50)
print("Model V2 — Random Forest")
print("=" * 50)

model_v2 = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])
model_v2.fit(X_train, y_train)

y_pred_v2 = model_v2.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred_v2))
print(classification_report(y_test, y_pred_v2, target_names=class_names))

joblib.dump(model_v2, os.path.join(MODEL_DIR, "iris_model_v2.joblib"))

metadata_v2 = {
    "model_name": "iris-random-forest",
    "model_version": "2.0.0",
    "features": feature_names,
    "classes": class_names,
    "accuracy": round(accuracy_score(y_test, y_pred_v2), 4),
    "hyperparameters": {
        "n_estimators": 100,
        "random_state": 42
    }
}

with open(os.path.join(MODEL_DIR, "iris_model_v2_metadata.json"), "w") as f:
    json.dump(metadata_v2, f, indent=2)

print("V2 saved → iris_model_v2.joblib + iris_model_v2_metadata.json\n")

# ── Comparison ────────────────────────────────────────────────────────────────
print("=" * 50)
print("Summary")
print("=" * 50)
print(f"V1 Logistic Regression  accuracy: {metadata_v1['accuracy']:.4f}")
print(f"V2 Random Forest        accuracy: {metadata_v2['accuracy']:.4f}")
winner = "V2 (Random Forest)" if metadata_v2["accuracy"] >= metadata_v1["accuracy"] else "V1 (Logistic Regression)"
print(f"Best model: {winner}")