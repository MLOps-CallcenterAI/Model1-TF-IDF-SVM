# %%
import os

import mlflow
import pandas as pd
from dotenv import load_dotenv

# %%
load_dotenv()  # expects .env file with URL and token

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")  # e.g., https://dbc-xxxx.cloud.databricks.com
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise ValueError("Please set your Databricks HOST and TOKEN in the .env file")

os.environ["DATABRICKS_HOST"] = DATABRICKS_HOST
os.environ["DATABRICKS_TOKEN"] = DATABRICKS_TOKEN

# %%
mlflow.set_tracking_uri("databricks")
experiment_name = os.getenv("EXPERIMENT_NAME")
mlflow.set_experiment(experiment_name)

model_uri = os.getenv("MODEL_URI")
print(f"Loading model from URI: {model_uri} ...")

model = mlflow.sklearn.load_model(model_uri)  # Directly load scikit-learn pipeline

# %%
test_input = pd.DataFrame(
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

# %%
# Pass the text column directly
predictions = model.predict(test_input["text"])
print(predictions)
