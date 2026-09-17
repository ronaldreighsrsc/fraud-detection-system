import pytest
from fastapi.testclient import TestClient
from fastapii import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "2.0.0 Enterprise" in data.get("version", "")


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data.get("status") == "ok"
    assert "modelos_cargados" in data


def test_v2_system_info(client):
    response = client.get("/api/v2/info")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "OPERATIONAL"
    assert data["sla_target"] == "< 30 ms"
    assert "CMF Capítulo 20-10" in data["regulatory_compliance"][0]


def test_evaluate_transaction_normal(client):
    payload = {
        "tx_id": "TX-TEST-NORM-01",
        "origin_account": "ACC_NORMAL_01",
        "destination_account": "ACC_NORMAL_02",
        "destination_country": "CHL",
        "account_age_days": 450.0,
        "transaction_amount": 25000.0,
        "hour_of_day": 14.0,
        "day_of_week": 2.0,
        "merchant_category": 3.0,
        "distance_from_home": 2.5,
        "is_international": 0.0
    }
    response = client.post("/api/v2/evaluate_transaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["tx_id"] == "TX-TEST-NORM-01"
    assert data["action"] in ["APPROVE", "CHALLENGE_STEPUP"]
    assert "top_shap_reasons" in data
    assert len(data["top_shap_reasons"]) <= 3
    assert "latency_ms" in data


def test_evaluate_transaction_blacklisted_origin(client):
    payload = {
        "tx_id": "TX-TEST-BLOCK-01",
        "origin_account": "999999",  # Cuenta en lista negra sembrada en lifespan
        "destination_account": "ACC_NORMAL_02",
        "destination_country": "CHL",
        "transaction_amount": 10000.0,
        "hour_of_day": 12.0
    }
    response = client.post("/api/v2/evaluate_transaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "BLOCK_IMMEDIATE"
    assert data["risk_level"] == "CRITICAL"
    assert data["notify_compliance"] is True
    assert "CMF-01" in data["reason"]


def test_evaluate_transaction_sanctioned_country(client):
    payload = {
        "tx_id": "TX-TEST-SANCT-01",
        "origin_account": "ACC_NORMAL_10",
        "destination_account": "ACC_FOREIGN_99",
        "destination_country": "PRK",  # País sancionado
        "transaction_amount": 50000.0,
        "hour_of_day": 15.0
    }
    response = client.post("/api/v2/evaluate_transaction", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["action"] == "BLOCK_IMMEDIATE"
    assert "PLAFT-03" in data["reason"]


def test_generate_ros_endpoint(client):
    response = client.post("/api/v2/compliance/generate_ros/TX-BCI-ALERT-55")
    assert response.status_code == 200
    data = response.json()
    assert data["tx_id"] == "TX-BCI-ALERT-55"
    assert "ros_formal_report" in data
    report = data["ros_formal_report"]
    assert "CONFIDENCIAL" in report
    assert "UAF" in report
    assert "Banco Bci" in report


def test_v1_predict_endpoint_backward_compatibility(client):
    # Payload v1 exacto con 18 variables
    payload = {
        "transaction_amount": 150.5, "hour_of_day": 14, "day_of_week": 3,
        "merchant_category": 2, "distance_from_home": 5.2, "is_international": 0,
        "amount_deviation": 10.0, "amount_zscore": 0.5, "amount_to_median_ratio": 1.1,
        "tx_frequency_1h": 2, "tx_frequency_24h": 5, "hour_sin": -0.5,
        "hour_cos": -0.866, "day_sin": 0.433, "day_cos": -0.9,
        "amount_x_distance": 782.6, "amount_x_international": 0.0, "zscore_x_frequency": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "Transacción analizada"
    assert "predictions" in data
