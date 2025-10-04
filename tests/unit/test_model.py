import os

import mlflow
import numpy as np
import pandas as pd
import pytest
from dotenv import load_dotenv


@pytest.fixture(scope="session")
def setup_env():
    """Load environment variables and verify Databricks credentials"""
    load_dotenv()
    host = os.getenv("url")
    token = os.getenv("access_token")

    if not host or not token:
        pytest.skip("Databricks credentials are missing in .env file")

    os.environ["DATABRICKS_HOST"] = host
    os.environ["DATABRICKS_TOKEN"] = token

    return {"host": host, "token": token}


@pytest.fixture(scope="session")
def load_model(setup_env):
    """Load MLflow model once for all tests"""
    mlflow.set_tracking_uri("databricks")
    model_uri = "models:/m-49e62ea546064bd1b56cff76a8aded2f"
    model = mlflow.sklearn.load_model(model_uri)
    return model


@pytest.fixture
def sample_data():
    """Provide test input DataFrame"""
    return pd.DataFrame(
        {
            "text": [
                "cannot login to my account",
                "server is down, need immediate support",
                "how to reset my password?",
                "request size increase for my database",
                "cannot purchase the premium plan",
                "need help with billing issue",
                "unable to access my files",
            ]
        }
    )


def test_model_loading(load_model):
    """Verify model loads correctly from MLflow"""
    assert load_model is not None, "Model failed to load"
    assert hasattr(load_model, "predict"), "Loaded object is not a valid model"


def test_model_prediction(load_model, sample_data):
    """Test model predictions"""
    predictions = load_model.predict(sample_data["text"])
    assert isinstance(
        predictions, (list, tuple, pd.Series, np.ndarray)
    ), "Predictions are not iterable"
    assert len(predictions) == len(sample_data), "Prediction count mismatch"
    assert all(
        isinstance(p, str) or isinstance(p, int) for p in predictions
    ), "Invalid prediction type"
