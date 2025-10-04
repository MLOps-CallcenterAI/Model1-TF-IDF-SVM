import os

import mlflow
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load environment variables
load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
MODEL_URI = os.getenv("MODEL_URI")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise ValueError("Please set your Databricks HOST and TOKEN in the .env file")

os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# MLflow setup
mlflow.set_tracking_uri("databricks")

# Load the model from Databricks
print(f"🔹 Loading model from URI: {MODEL_URI} ...")
model = mlflow.sklearn.load_model(MODEL_URI)

# Initialize FastAPI
app = FastAPI(
    title="CallCenterAI Model API",
    description="API for text classification using TF-IDF + SVM model deployed via MLflow on Databricks.",
    version="1.1",
)


# Request models
class TextBatchRequest(BaseModel):
    text: list[str]


class SingleTextRequest(BaseModel):
    text: str


@app.get("/")
def home():
    return {
        "message": "Welcome to CallCenterAI Model API! Use /predict or /predict_one endpoints."
    }


@app.post("/predict")
def predict_batch(request: TextBatchRequest):
    """Predict multiple tickets at once"""
    try:
        df = pd.DataFrame({"text": request.text})
        predictions = model.predict(df["text"])
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_one")
def predict_single(request: SingleTextRequest):
    """Predict a single ticket"""
    try:
        df = pd.DataFrame({"text": [request.text]})
        prediction = model.predict(df["text"])[0]
        return {"input": request.text, "prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
