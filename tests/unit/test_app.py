import pytest
from fastapi.testclient import TestClient

from src.app import app

client = TestClient(app)


@pytest.fixture
def sample_texts():
    return [
        "cannot login to my account",
        "need help with billing issue",
        "server is down, need immediate support",
    ]


def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_predict_batch(sample_texts):
    payload = {"text": sample_texts}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == len(sample_texts)


def test_predict_one():
    payload = {"text": "unable to access my files"}
    response = client.post("/predict_one", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "input" in data
    assert "prediction" in data
    assert data["input"] == "unable to access my files"
