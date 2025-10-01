import re

import mlflow
import mlflow.sklearn
import nltk
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from nltk.stem import WordNetLemmatizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import LinearSVC

# Download required NLTK data (run once)
nltk.download("wordnet")
nltk.download("punkt")
nltk.download("stopwords")

# Set MLflow tracking URI to match your observed port
mlflow.set_tracking_uri("http://localhost:8080")


# Load and preprocess dataset
def load_data(dataset_path="dataset/all_tickets_processed_improved_v3.csv"):
    df = pd.read_csv(dataset_path)
    if "Document" not in df.columns or "Topic_group" not in df.columns:
        raise ValueError("Dataset must contain 'Document' and 'Topic_group' columns")

    # Custom stopwords and lemmatization
    stop_words = set(nltk.corpus.stopwords.words("english")) | {
        "please",
        "ticket",
        "help",
    }
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        text = re.sub(r"[^\w\s]", "", text.lower())  # Remove punctuation
        tokens = re.findall(r"\w+", text)
        tokens = [
            lemmatizer.lemmatize(word) for word in tokens if word not in stop_words
        ]
        return " ".join(tokens)

    df["Document"] = df["Document"].apply(preprocess_text)
    X = df["Document"].values
    y = df["Topic_group"].values
    return X, y


# Train TF-IDF + SVM model with hyperparameter tuning
def train_tfidf_svm(X, y):
    # Vectorizer with enhanced settings
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3), max_features=10000, min_df=5, sublinear_tf=True
    )
    X_tfidf = vectorizer.fit_transform(X)

    # Define parameter grid for GridSearchCV
    param_grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l2"]}
    svm = LinearSVC(class_weight="balanced", max_iter=2000)
    grid_search = GridSearchCV(svm, param_grid, cv=3, scoring="f1_weighted")
    grid_search.fit(X_tfidf, y)

    # Train with calibrated classifier using best model
    best_svm = grid_search.best_estimator_
    calibrated_svm = CalibratedClassifierCV(best_svm, cv=3)
    calibrated_svm.fit(X_tfidf, y)

    # Stratified k-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_svm, X_tfidf, y, cv=cv, scoring="accuracy")

    # Predict on full dataset for initial evaluation
    y_pred = calibrated_svm.predict(X_tfidf)
    probabilities = calibrated_svm.predict_proba(X_tfidf)

    return calibrated_svm, vectorizer, y, y_pred, probabilities, cv_scores.mean()


# Main training function with MLflow logging
def main():
    # Load data
    X, y = load_data()

    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model, vectorizer, y_test, y_pred, probabilities, cv_accuracy = train_tfidf_svm(
            X, y
        )

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log parameters and metrics
        mlflow.log_param("model", "TF-IDF + SVM")
        mlflow.log_param("ngram_range", "(1, 3)")
        mlflow.log_param("max_features", 10000)
        mlflow.log_param("min_df", 5)
        mlflow.log_param("sublinear_tf", True)
        mlflow.log_param("best_C", model.calibrated_classifiers_[0].estimator.C)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("cv_accuracy", cv_accuracy)

        # Prepare input example for signature
        input_example = np.array(["new hardware issue"])
        X_example_tfidf = vectorizer.transform(input_example)
        signature = infer_signature(X_example_tfidf, model.predict(X_example_tfidf))

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="tfidf_svm_model",
            signature=signature,
            input_example=input_example,
        )

        print(
            f"Training complete. Accuracy: {accuracy:.2f}, F1: {f1:.2f}, CV Accuracy: {cv_accuracy:.2f}"
        )


if __name__ == "__main__":
    main()
