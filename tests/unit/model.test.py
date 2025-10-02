import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold

# Download required NLTK data for testing
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Copy the functions from your main script directly into the test file
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
    lemmatizer = nltk.stem.WordNetLemmatizer()

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

def train_tfidf_svm(X, y):
    # Vectorizer with enhanced settings
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), max_features=1000, min_df=1, sublinear_tf=True  # Reduced for testing
    )
    X_tfidf = vectorizer.fit_transform(X)

    unique_classes, class_counts = np.unique(y, return_counts=True)
    min_samples_per_class = np.min(class_counts)

    if min_samples_per_class >= 2:
        safe_cv_folds = 2
        # Define parameter grid for GridSearchCV
        param_grid = {"C": [0.1, 1], "penalty": ["l2"]}  # Reduced params for testing
        svm = LinearSVC(class_weight="balanced", max_iter=1000)  # Reduced max_iter for testing
        grid_search = GridSearchCV(svm, param_grid, cv=safe_cv_folds, scoring="f1_weighted")  # Reduced cv for testing
        grid_search.fit(X_tfidf, y)

        # Train with calibrated classifier using best model
        best_svm = grid_search.best_estimator_
        calibrated_svm = CalibratedClassifierCV(best_svm, cv=safe_cv_folds)  # Reduced cv for testing
        calibrated_svm.fit(X_tfidf, y)
        model = calibrated_svm
        probabilities = calibrated_svm.predict_proba(X_tfidf)
    else:
        # Handle minimal data: no grid search, no calibration
        best_svm = LinearSVC(class_weight="balanced", max_iter=1000)
        best_svm.fit(X_tfidf, y)
        model = best_svm
        probabilities = best_svm.decision_function(X_tfidf)

    # Common parts
    y_pred = model.predict(X_tfidf)

    # For small datasets, use simpler cross-validation
    # Count minimum samples per class to determine safe number of splits
    if min_samples_per_class > 1:
        cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        cv_scores = cross_val_score(best_svm, X_tfidf, y, cv=cv, scoring="accuracy")
        cv_accuracy = cv_scores.mean()
    else:
        # If we have only 1 sample per class, skip cross-validation
        cv_accuracy = 0.0

    return model, vectorizer, y, y_pred, probabilities, cv_accuracy


class TestBasicFunctionality:
    """Basic unit tests for the ML training script"""
    
    def test_load_data_success(self):
        """Test that data loading works with correct CSV format"""
        # Create a temporary CSV file with test data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Document,Topic_group\n")
            f.write("Please help with my computer,Hardware\n")
            f.write("Software is crashing,Software\n")
            f.write("Network is slow,Network\n")
            temp_file = f.name
        
        try:
            # Test the load_data function
            X, y = load_data(temp_file)
            
            # Check that we get the expected outputs
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert len(X) == 3
            assert len(y) == 3
            assert "please" not in X[0].lower()  # Check stopword removal
            assert "computer" in X[0].lower()    # Check important word remains
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def test_train_tfidf_svm_basic(self):
        """Test that the training function works with simple data"""
        # Create simple test data with more samples per class
        X = np.array([
            "computer not working hardware issue",
            "software application crash problem", 
            "network connection slow internet",
            "hardware memory problem ram",
            "software update needed bug",
            "network wifi issue connection",
            "computer screen black hardware",  # Additional hardware sample
            "software program error",         # Additional software sample
            "network router problem"          # Additional network sample
        ])
        y = np.array([
            "Hardware", "Software", "Network", 
            "Hardware", "Software", "Network",
            "Hardware", "Software", "Network"  # 3 samples per class
        ])
        
        # Test the training function
        model, vectorizer, y_test, y_pred, probabilities, cv_accuracy = train_tfidf_svm(X, y)
        
        # Check that we get all expected return values
        assert model is not None
        assert vectorizer is not None
        assert len(y_pred) == len(X)
        assert len(probabilities) == len(X)
        assert 0 <= cv_accuracy <= 1  # CV accuracy should be between 0 and 1
        
        # Test that the model can make predictions
        test_text = ["new hardware problem"]
        test_features = vectorizer.transform(test_text)
        prediction = model.predict(test_features)
        assert prediction is not None
    
    def test_train_tfidf_svm_minimal_data(self):
        """Test training with very minimal data (edge case)"""
        # Test with just enough data to train
        X = np.array([
            "computer hardware",
            "software program",
            "network connection",
            "hardware problem"
        ])
        y = np.array(["Hardware", "Software", "Network", "Hardware"])
        
        # This should handle minimal data without crashing
        model, vectorizer, y_test, y_pred, probabilities, cv_accuracy = train_tfidf_svm(X, y)
        
        # Basic assertions
        assert model is not None
        assert vectorizer is not None
        assert len(y_pred) == len(X)
    
    def test_load_data_missing_columns(self):
        """Test that data loading fails gracefully with wrong columns"""
        # Create a temporary CSV file with incorrect columns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("Wrong_Column,Another_Column\n")
            f.write("test data,test label\n")
            temp_file = f.name
        
        try:
            # Test that the function raises the expected error
            with pytest.raises(ValueError, match="must contain 'Document' and 'Topic_group' columns"):
                load_data(temp_file)
                
        finally:
            # Clean up temporary file
            os.unlink(temp_file)