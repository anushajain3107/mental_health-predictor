import pytest
from fastapi.testclient import TestClient
import joblib
import pandas as pd
import time
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api import app

client = TestClient(app)

@pytest.fixture
def sample_valid_payload():
    return {
        "Age": 21,
        "Gender": "Female",
        "Country": "India",
        "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "LinkedIn",
        "Purpose_Of_Use": "Education",
        "Avg_Daily_Usage_Hours": 2.5,
        "Daily_Unlocks": 90,
        "Study_Hours": 5.5,
        "Physical_Activity_Hours": 2.0,
        "Sleep_Hours_Per_Night": 8.0,
        "Stress_Level": "Low"
    }

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True

def test_single_prediction_success(sample_valid_payload):
    response = client.post("/predict", json=sample_valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_score" in data
    assert 1.0 <= data["predicted_score"] <= 10.0
    assert data["wellbeing_category"] == "Healthy Wellbeing"
    assert "latency_ms" in data

def test_single_prediction_validation_failure():
    invalid_payload = {
        "Age": 12,  # Invalid: below minimum 15
        "Gender": "Female",
        "Country": "India",
        "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "LinkedIn",
        "Purpose_Of_Use": "Education",
        "Avg_Daily_Usage_Hours": -2.0,  # Invalid: negative
        "Daily_Unlocks": 90,
        "Study_Hours": 5.5,
        "Physical_Activity_Hours": 2.0,
        "Sleep_Hours_Per_Night": 8.0,
        "Stress_Level": "UnknownStress"  # Invalid: not in Literal
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422

def test_batch_prediction(sample_valid_payload):
    batch_payload = {
        "profiles": [
            sample_valid_payload,
            {
                "Age": 22,
                "Gender": "Male",
                "Country": "USA",
                "Academic_Level": "Graduate",
                "Most_Used_Platform": "Instagram",
                "Purpose_Of_Use": "Entertainment",
                "Avg_Daily_Usage_Hours": 7.0,
                "Daily_Unlocks": 220,
                "Study_Hours": 1.5,
                "Physical_Activity_Hours": 0.5,
                "Sleep_Hours_Per_Night": 5.0,
                "Stress_Level": "Very High"
            }
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["predicted_score"] > data["predictions"][1]["predicted_score"]

def test_pipeline_determinism(sample_valid_payload):
    res1 = client.post("/predict", json=sample_valid_payload).json()["predicted_score"]
    res2 = client.post("/predict", json=sample_valid_payload).json()["predicted_score"]
    assert res1 == res2

def test_inference_latency_benchmark(sample_valid_payload):
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        client.post("/predict", json=sample_valid_payload)
        latencies.append((time.perf_counter() - t0) * 1000)
    
    avg_latency = sum(latencies) / len(latencies)
    assert avg_latency < 30.0, f"Average latency too high: {avg_latency:.2f}ms"
