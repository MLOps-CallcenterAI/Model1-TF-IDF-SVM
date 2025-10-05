# ⚡ `app.py` — FastAPI Inference API for CallCenterAI

## 📘 Overview

The `app.py` module provides a **FastAPI-based REST API** for real-time text classification using the **TF-IDF + SVM model** trained and registered via MLflow on **Databricks**.

It handles:

* Secure model loading from MLflow/Databricks
* Prediction endpoints (`/predict` and `/predict_one`)
* Input validation using **Pydantic**
* Structured error handling

---

## ⚙️ 1. Imports and Dependencies

```python
import os
import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
```

### Description

| Library    | Purpose                                      |
| ---------- | -------------------------------------------- |
| `os`       | Environment variable management              |
| `mlflow`   | Loading the registered model from Databricks |
| `pandas`   | DataFrame structure for handling text inputs |
| `dotenv`   | Load credentials and config from `.env`      |
| `fastapi`  | Create and manage REST API endpoints         |
| `pydantic` | Validate input payloads                      |

---

## 🔐 2. Environment Setup

Before the API starts, environment variables are loaded from `.env`:

```python
load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
MODEL_URI = os.getenv("MODEL_URI")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise ValueError("Please set your Databricks HOST and TOKEN in the .env file")
```

If credentials are missing, the app fails fast to ensure secure operation.

Then, the variables are exported for MLflow:

```python
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN
```

---

## 🌐 3. MLflow Model Setup

The app connects to the Databricks MLflow tracking server:

```python
mlflow.set_tracking_uri("databricks")
print(f"🔹 Loading model from URI: {MODEL_URI} ...")
model = mlflow.sklearn.load_model(MODEL_URI)
```

### Expected `MODEL_URI` format:

```
models:/<model_name>/<version>
```

Example:

```
models:/tfidf_svm_classifier/3
```

The model is a **Scikit-learn pipeline**, containing:

* A **TF-IDF vectorizer** for text feature extraction
* A **Calibrated Linear SVM classifier** for category prediction

---

## 🚀 4. FastAPI Initialization

The API metadata is defined when creating the FastAPI instance:

```python
app = FastAPI(
    title="CallCenterAI Model API",
    description="API for text classification using TF-IDF + SVM model deployed via MLflow on Databricks.",
    version="1.1",
)
```

This metadata appears in the auto-generated documentation at:

* **Swagger UI** → `http://localhost:8000/docs`
* **ReDoc UI** → `http://localhost:8000/redoc`

---

## 📦 5. Request Models

Pydantic models ensure that input payloads are validated automatically.

```python
class TextBatchRequest(BaseModel):
    text: list[str]

class SingleTextRequest(BaseModel):
    text: str
```

| Model               | Field  | Type      | Description           |
| ------------------- | ------ | --------- | --------------------- |
| `TextBatchRequest`  | `text` | list[str] | Multiple ticket texts |
| `SingleTextRequest` | `text` | str       | Single ticket text    |

---

## 🔹 6. Endpoints

### **1️⃣ Root Endpoint**

```python
@app.get("/")
def home():
    return {
        "message": "Welcome to CallCenterAI Model API! Use /predict or /predict_one endpoints."
    }
```

**Purpose:** Basic health check — confirms that the API is running and provides usage guidance.

---

### **2️⃣ Batch Prediction**

```python
@app.post("/predict")
def predict_batch(request: TextBatchRequest):
    try:
        df = pd.DataFrame({"text": request.text})
        predictions = model.predict(df["text"])
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Request Body Example:**

```json
{
  "text": [
    "Unable to log into my account",
    "How do I change my billing information?"
  ]
}
```

**Response Example:**

```json
{
  "predictions": ["Account Access", "Billing Issue"]
}
```

**Key Points**

* Handles multiple inputs efficiently
* Converts text list → DataFrame for vectorization
* Returns predictions as a list of strings
* Handles all runtime errors with `HTTPException`

---

### **3️⃣ Single Prediction**

```python
@app.post("/predict_one")
def predict_single(request: SingleTextRequest):
    try:
        df = pd.DataFrame({"text": [request.text]})
        prediction = model.predict(df["text"])[0]
        return {"input": request.text, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Request Example:**

```json
{ "text": "Server is down, need urgent assistance" }
```

**Response Example:**

```json
{
  "input": "Server is down, need urgent assistance",
  "prediction": "Technical Support"
}
```

---

## 🧱 7. Error Handling

The API uses `try/except` blocks to catch all internal exceptions.
Any error during preprocessing or model inference is returned as a structured response:

```json
{
  "detail": "Model not found or invalid input format"
}
```

Status code: **500 Internal Server Error**

---

## 🧪 8. Testing the API

You can test locally using **curl** or **HTTPie**:

```bash
curl -X POST http://localhost:8000/predict_one \
     -H "Content-Type: application/json" \
     -d '{"text": "I need to reset my password"}'
```

Or with **Python**:

```python
import requests
r = requests.post(
    "http://localhost:8000/predict_one",
    json={"text": "I cannot access my files"}
)
print(r.json())
```

---

## 🧾 9. Summary

| Component              | Description                                           |
| ---------------------- | ----------------------------------------------------- |
| **Purpose**            | Serve predictions from a trained MLflow model         |
| **Framework**          | FastAPI                                               |
| **Model Source**       | MLflow Databricks registry                            |
| **Endpoints**          | `/`, `/predict`, `/predict_one`                       |
| **Request Validation** | Pydantic (`TextBatchRequest`, `SingleTextRequest`)    |
| **Error Handling**     | Structured via `HTTPException`                        |
| **Auto Docs**          | `/docs` (Swagger) and `/redoc`                        |
| **Integration**        | Connected with `model.py` and Databricks-hosted model |

---

## 🧭 10. Deployment Notes

* The service can run inside Docker using the provided `Dockerfile`
* Requires a valid `.env` file in project root:

  ```bash
  DATABRICKS_HOST=https://adb-xxxx.azuredatabricks.net
  DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXXXXXX
  MODEL_URI=models:/tfidf_svm_classifier/3
  ```
* Expose port `8000` in the container for API access
* Example run command:

  ```bash
  uvicorn src.app:app --host 0.0.0.0 --port 8000
  ```

---
