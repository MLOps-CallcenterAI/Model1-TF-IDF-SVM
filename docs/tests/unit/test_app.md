# 🧪 `test_app.py` — Unit Tests for CallCenterAI FastAPI API

## 📘 Overview

The `test_app.py` module provides **automated tests** for the FastAPI service defined in `app.py`.
It ensures that all API endpoints (`/`, `/predict`, `/predict_one`) work as expected — both functionally and structurally — before deployment.

These tests are written using **pytest** and **FastAPI’s TestClient**, which allows calling the API directly without running a live server.

---

## ⚙️ 1. Imports and Setup

```python
import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)
```

### Description

| Library              | Purpose                                                       |
| -------------------- | ------------------------------------------------------------- |
| `pytest`             | Framework for defining and running tests                      |
| `fastapi.testclient` | Provides a lightweight HTTP client for local endpoint testing |
| `src.app`            | Imports the FastAPI app instance to test                      |
| `client`             | A reusable TestClient object that simulates API requests      |

---

## 🧩 2. Pytest Fixture: `sample_texts`

```python
@pytest.fixture
def sample_texts():
    return [
        "cannot login to my account",
        "need help with billing issue",
        "server is down, need immediate support",
    ]
```

### Purpose

* Provides reusable **sample input texts** for batch predictions.
* Fixtures in pytest allow parameter reuse across multiple tests for clarity and DRY code.

---

## 🔹 3. Test: `test_home_route`

```python
def test_home_route():
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]
```

### ✅ Validates:

| Check                         | Purpose                                               |
| ----------------------------- | ----------------------------------------------------- |
| `response.status_code == 200` | Confirms that the `/` route is reachable              |
| `"Welcome" in message`        | Ensures the API returns the expected greeting message |

### Example Response:

```json
{
  "message": "Welcome to CallCenterAI Model API! Use /predict or /predict_one endpoints."
}
```

---

## 🔹 4. Test: `test_predict_batch`

```python
def test_predict_batch(sample_texts):
    payload = {"text": sample_texts}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert isinstance(data["predictions"], list)
    assert len(data["predictions"]) == len(sample_texts)
```

### ✅ Validates:

| Check           | Description                                        |
| --------------- | -------------------------------------------------- |
| Status Code 200 | Endpoint `/predict` is accessible and functional   |
| Key Presence    | Response contains `"predictions"`                  |
| Data Type       | Predictions are returned as a list                 |
| Length Match    | Number of predictions equals number of input texts |

### Example Input:

```json
{
  "text": [
    "cannot login to my account",
    "need help with billing issue",
    "server is down, need immediate support"
  ]
}
```

### Example Response:

```json
{
  "predictions": ["Account Issue", "Billing", "Technical Support"]
}
```

---

## 🔹 5. Test: `test_predict_one`

```python
def test_predict_one():
    payload = {"text": "unable to access my files"}
    response = client.post("/predict_one", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "input" in data
    assert "prediction" in data
    assert data["input"] == "unable to access my files"
```

### ✅ Validates:

| Check           | Description                                         |
| --------------- | --------------------------------------------------- |
| Status Code 200 | Ensures `/predict_one` endpoint is working          |
| Keys Present    | Checks for `"input"` and `"prediction"` in response |
| Input Match     | Confirms returned input matches the sent text       |

### Example Input:

```json
{ "text": "unable to access my files" }
```

### Example Response:

```json
{
  "input": "unable to access my files",
  "prediction": "File Access Issue"
}
```

---

## 🧱 6. How to Run Tests

You can run all tests using **pytest** from the project root:

```bash
pytest -v tests/test_app.py
```

Or run all tests recursively:

```bash
pytest -v
```

---

## 🧭 7. Test Design Summary

| Test                 | Endpoint       | Goal                                                | Expected Outcome                          |
| -------------------- | -------------- | --------------------------------------------------- | ----------------------------------------- |
| `test_home_route`    | `/`            | Verify API is up and returning the greeting message | `200 OK`, contains "Welcome"              |
| `test_predict_batch` | `/predict`     | Ensure batch prediction works correctly             | List of predictions, length matches input |
| `test_predict_one`   | `/predict_one` | Ensure single prediction works correctly            | Returns `input` and `prediction` fields   |

---

## 🧰 8. Notes

* **Mocking the model:**
  If the real MLflow model is unavailable during testing, you can mock it to isolate the API logic:

  ```python
  from unittest.mock import MagicMock
  from src import app
  app.model = MagicMock()
  app.model.predict.return_value = ["MockPrediction"]
  ```

* This allows CI/CD to run tests **without requiring Databricks or MLflow connectivity.**

* Ensure `.env` is properly configured when running locally so the real model can load successfully.

---
