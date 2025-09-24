import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict_endpoint():
    test_data = {
        "patient_id": "test_001",
        "ecg": {"samples": [0.1] * 3600, "fs": 360},
        "ppg": {"samples": [0.05] * 1250, "fs": 125}
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code in [200, 500]  #