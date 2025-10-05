# 🧠 `model.py` — Model Loading & Prediction Pipeline

## 📄 Overview

The `model.py` script is responsible for **loading a trained machine learning model from MLflow**, connecting securely to **Databricks**, and **performing predictions** on sample input text data.

It acts as the **inference layer** of the system — loading the model artifact registered by the training notebook (`clean&train.ipynb`) and exposing it for use in the FastAPI app (`app.py`).

---

## ⚙️ 1. Imports and Dependencies

```python
import os
import mlflow
import pandas as pd
from dotenv import load_dotenv
```

### Description

| Library  | Purpose                                        |
| -------- | ---------------------------------------------- |
| `os`     | Environment variable management                |
| `mlflow` | Model loading and experiment tracking          |
| `pandas` | DataFrame creation for testing predictions     |
| `dotenv` | Loads environment variables from a `.env` file |

---

## 🔐 2. Environment Configuration

The script loads Databricks credentials and model information securely from a `.env` file:

```python
load_dotenv()  # expects .env file with URL and token

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise ValueError("Please set your Databricks HOST and TOKEN in the .env file")
```

If either variable is missing, the script raises a clear exception to ensure that secure access is always enforced.

---

## 🌐 3. Databricks & MLflow Setup

The environment variables are then passed to the system and used to connect to the Databricks workspace:

```python
os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

mlflow.set_tracking_uri("databricks")
experiment_name = os.getenv("EXPERIMENT_NAME")
mlflow.set_experiment(experiment_name)
```

This ensures that MLflow operations (such as model retrieval) are tracked and authenticated correctly in Databricks.

---

## 📦 4. Model Loading

The model is retrieved from the Databricks model registry using its URI:

```python
model_uri = os.getenv("MODEL_URI")
print(f"Loading model from URI: {model_uri} ...")

model = mlflow.sklearn.load_model(model_uri)
```

### Notes

* The `MODEL_URI` typically looks like:

  ```
  models:/<model_name>/<version_number>
  ```

  Example:

  ```
  models:/tfidf_svm_classifier/1
  ```
* The loaded object is a **Scikit-learn pipeline**, which includes:

  * A **TF-IDF vectorizer** (feature extraction)
  * A **Calibrated LinearSVC classifier** (prediction)

---

## 🧩 5. Test Input Data

A quick example DataFrame is created to validate the loaded model:

```python
test_input = pd.DataFrame({
    "text": [
        "cannot login to my account",
        "server is down, need immediate support",
        "how to reset my password?",
        "request size increase for my database",
        "cannot purchase the premium plan",
        "need help with billing issue",
        "unable to access my files",
    ]
})
```

This represents a small batch of support tickets or user requests that the model will classify.

---

## 🧠 6. Prediction Phase

Predictions are performed directly on the text column:

```python
predictions = model.predict(test_input["text"])
print(predictions)
```

### Expected Output

An array of predicted categories, such as:

```
['Account Access', 'Technical Support', 'Account Access',
 'Resource Management', 'Payment Issue', 'Billing Support', 'File Access']
```

The exact labels depend on the trained model’s category mapping.

---

## 🔄 7. Integration with API

Although the script can run independently, it is primarily designed to be **imported and used by `app.py`** — the FastAPI backend.
In the production flow:

* `app.py` loads this model once at startup.
* Incoming API requests with ticket text are passed to `model.predict()`.
* Predictions are returned as JSON responses.

This structure ensures **modularity**, **fast inference**, and **clean separation between ML logic and API serving**.

---

## 🧾 8. Summary

| Component              | Description                                                    |
| ---------------------- | -------------------------------------------------------------- |
| **Purpose**            | Load and run predictions with the trained MLflow model         |
| **Model Source**       | Databricks Model Registry (via MLflow)                         |
| **Environment Config** | `.env` file (contains host, token, experiment name, model URI) |
| **Dependencies**       | `mlflow`, `pandas`, `dotenv`, `os`                             |
| **Outputs**            | Predicted labels for input text samples                        |
| **Integration**        | Used by FastAPI app (`app.py`) for serving                     |

---

## 🚀 9. Usage Example

To run the script locally for validation:

```bash
python src/model.py
```

Ensure that a `.env` file is present in the project root with:

```bash
DATABRICKS_HOST=https://adb-xxxxxxxxxx.azuredatabricks.net
DATABRICKS_TOKEN=dapiXXXXXXXXXXXXXXXXXXXXXXXX
EXPERIMENT_NAME=/Users/your.email@databricks.com/CallCenterAI-Model1-TF-IDF-SVM
MODEL_URI=models:/m-49e62ea546064bd1b56cff76a8aded2f
```

---
