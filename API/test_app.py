import pytest
from fastapi.testclient import TestClient
from API.app import app

client = TestClient(app)

# ── საერთო fixtures ────────────────────────────────────────────────────────────
VALID_IRIS = {
    "sepal_length_cm": 5.1,
    "sepal_width_cm":  3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm":  0.2,
}

INVALID_IRIS_OUT_OF_RANGE = {
    "sepal_length_cm": 99.0,   # gt=0, lt=10 → ვალიდაცია ვერ გაივლის
    "sepal_width_cm":  3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm":  0.2,
}

INVALID_IRIS_MISSING_FIELD = {
    "sepal_length_cm": 5.1,
    "sepal_width_cm":  3.5,
    # petal_length_cm და petal_width_cm გამოტოვებულია
}

INVALID_IRIS_WRONG_TYPE = {
    "sepal_length_cm": "five",   # string, float-ის მაგივრად
    "sepal_width_cm":  3.5,
    "petal_length_cm": 1.4,
    "petal_width_cm":  0.2,
}


# ══════════════════════════════════════════════════════════════════════════════
# ── ზოგადი endpoint-ები ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def test_root():
    """
    შედეგი: 200 OK
    ახსნა: root endpoint ბრუნებს სერვისის მეტადატას და ხელმისაწვდომ ვერსიებს.
    """
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert "v1" in body["available_versions"]
    assert "v2" in body["available_versions"]


def test_health():
    """
    შედეგი: 200 OK  |  {"status": "ok"}
    ახსნა: health check endpoint გამოიყენება მონიტორინგისთვის (liveness probe).
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ══════════════════════════════════════════════════════════════════════════════
# ── /v1/predict ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def test_v1_valid_request():
    """
    შედეგი: 200 OK
    ახსნა: ვალიდური Iris მონაცემები გადაეცემა Logistic Regression მოდელს.
           პასუხი შეიცავს კლასს, კლასის სახელს და ალბათობებს.
           ამ მნიშვნელობებისთვის (setosa-ს ნიმუში) მოდელი ბრუნებს class_id=0.
    """
    response = client.post("/v1/predict", json=VALID_IRIS)
    assert response.status_code == 200

    body = response.json()
    assert body["model_name"]    == "iris-logistic-regression"
    assert body["model_version"] == "1.0.0"
    assert body["predicted_class_id"]   == 0
    assert body["predicted_class_name"] == "setosa"
    assert len(body["probabilities"])   == 3
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-3  # ალბათობების ჯამი ≈ 1


def test_v1_invalid_out_of_range():
    """
    შედეგი: 422 Unprocessable Entity
    ახსნა: sepal_length_cm=99 არღვევს Field(lt=10) შეზღუდვას.
           FastAPI / Pydantic ავტომატურად ბრუნებს 422-ს ვალიდაციის შეცდომით,
           მოდელამდე საერთოდ არ მიდის მოთხოვნა.
    """
    response = client.post("/v1/predict", json=INVALID_IRIS_OUT_OF_RANGE)
    assert response.status_code == 422


def test_v1_invalid_missing_field():
    """
    შედეგი: 422 Unprocessable Entity
    ახსნა: სავალდებულო ველები (petal_length_cm, petal_width_cm) არ არის.
           Pydantic ვალიდაცია ვერ გაივლის, 422 სტატუსი ბრუნდება.
    """
    response = client.post("/v1/predict", json=INVALID_IRIS_MISSING_FIELD)
    assert response.status_code == 422


def test_v1_invalid_wrong_type():
    """
    შედეგი: 422 Unprocessable Entity
    ახსნა: sepal_length_cm="five" — string გადაეცა float ველს.
           Pydantic ვერ გარდაქმნის მნიშვნელობას, 422 ბრუნდება.
    """
    response = client.post("/v1/predict", json=INVALID_IRIS_WRONG_TYPE)
    assert response.status_code == 422


def test_v1_empty_body():
    """
    შედეგი: 422 Unprocessable Entity
    ახსნა: სრულიად ცარიელი body — ყველა სავალდებულო ველი აკლია.
    """
    response = client.post("/v1/predict", json={})
    assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# ── /v2/predict ───────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def test_v2_valid_request():
    """
    შედეგი: 200 OK
    ახსნა: ვალიდური მოთხოვნა Random Forest მოდელზე.
           V2 პასუხი ემატება top_feature_importances-ით —
           ეს V1-ისგან განმასხვავებელი დამატებითი ინფორმაციაა.
    """
    response = client.post("/v2/predict", json=VALID_IRIS)
    assert response.status_code == 200

    body = response.json()
    assert body["model_name"]    == "iris-random-forest"
    assert body["model_version"] == "2.0.0"
    assert body["predicted_class_id"]   == 0
    assert body["predicted_class_name"] == "setosa"
    assert len(body["probabilities"])   == 3
    assert abs(sum(body["probabilities"]) - 1.0) < 1e-3

    # V2-specific: feature importances
    fi = body["top_feature_importances"]
    assert set(fi.keys()) == {
        "sepal_length_cm", "sepal_width_cm",
        "petal_length_cm", "petal_width_cm"
    }
    assert abs(sum(fi.values()) - 1.0) < 1e-3  # importances ჯამი ≈ 1


def test_v2_invalid_out_of_range():
    """
    შედეგი: 422 Unprocessable Entity
    ახსნა: V1-ის ანალოგიური — იგივე Pydantic schema გამოიყენება,
           ვალიდაცია ორივე ვერსიაში ერთნაირად მუშაობს.
    """
    response = client.post("/v2/predict", json=INVALID_IRIS_OUT_OF_RANGE)
    assert response.status_code == 422


def test_v2_invalid_missing_field():
    """
    შედეგი: 422 Unprocessable Entity
    ახსნა: სავალდებულო ველები აკლია — 422 ბრუნდება V2-შიც.
    """
    response = client.post("/v2/predict", json=INVALID_IRIS_MISSING_FIELD)
    assert response.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# ── ვერსიების შედარება ────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

def test_v1_vs_v2_same_input_different_models():
    """
    შედეგი: ორივე 200 OK, მაგრამ სხვადასხვა model_name/model_version
    ახსნა: ერთი და იგივე input-ზე ორი განსხვავებული მოდელი პასუხობს.
           კლასიფიკაციის შედეგი შეიძლება დაემთხვეს (ორივე setosa-ს ირჩევს),
           მაგრამ ალბათობის განაწილება განსხვავებულია — სხვადასხვა ალგორითმი.
    """
    r1 = client.post("/v1/predict", json=VALID_IRIS)
    r2 = client.post("/v2/predict", json=VALID_IRIS)

    assert r1.status_code == 200
    assert r2.status_code == 200

    b1, b2 = r1.json(), r2.json()

    assert b1["model_name"]    != b2["model_name"]
    assert b1["model_version"] != b2["model_version"]

    # V1-ს არ აქვს feature importances, V2-ს აქვს
    assert "top_feature_importances" not in b1
    assert "top_feature_importances" in b2

    print("\n── V1 probabilities:", b1["probabilities"])
    print("── V2 probabilities:", b2["probabilities"])