# Unit Tests Documentation: ML Pipeline (TF-IDF + SVM)

## Overview

This document provides comprehensive documentation for the unit tests implemented in `tests/unit/test_model.py`. These tests validate the core functionality of the machine learning pipeline for text classification using TF-IDF vectorization and a Linear SVM classifier. The tests focus on data loading, preprocessing, model training, hyperparameter tuning via GridSearchCV, calibration, and cross-validation, while handling edge cases like minimal datasets.

The tests use **Pytest** as the testing framework and rely on libraries such as **Pandas**, **NumPy**, **Scikit-learn**, **NLTK**, and **re** for execution. NLTK data (WordNet, Punkt, Stopwords) is downloaded automatically during test runs.

Key functions tested:
- **`load_data(dataset_path)`**: Loads a CSV dataset, preprocesses text (lowercasing, punctuation removal, tokenization, stopword removal, lemmatization), and returns features (`X`) and labels (`y`).
- **`train_tfidf_svm(X, y)`**: Trains a TF-IDF vectorizer, performs GridSearchCV for SVM hyperparameters, calibrates the classifier, computes predictions and probabilities, and evaluates with cross-validation.

The test suite ensures:
- Robustness against invalid inputs (e.g., missing columns).
- Correct preprocessing and model outputs.
- Graceful handling of small datasets (e.g., fewer than 2 samples per class, where CV is skipped or simplified).

All tests are self-contained, using temporary files for CSV data where needed, and clean up resources post-execution.

## Test Suite: `TestBasicFunctionality`

This class contains four unit tests covering success paths, basic training, minimal data scenarios, and error handling.

### 1. `test_load_data_success`

**Description**:  
Verifies that the `load_data` function successfully loads and preprocesses a valid CSV dataset with the required columns (`Document` and `Topic_group`). It tests text preprocessing, including stopword removal and lemmatization.

**Test Data**:  
- A temporary CSV file is created with 3 sample rows:  
  | Document                  | Topic_group |
  |---------------------------|-------------|
  | Please help with my computer | Hardware   |
  | Software is crashing      | Software   |
  | Network is slow           | Network    |

**Execution Steps**:  
1. Create and write to a temporary CSV file.  
2. Call `load_data(temp_file)` to load and preprocess.  
3. Clean up the temporary file.

**Assertions**:  
- `X` and `y` are NumPy arrays.  
- Length of `X` and `y` is 3.  
- Preprocessing works: "please" (stopword) is removed from the first document; "computer" (key term) remains.

**Expected Outcome**:  
PASS – Confirms data loading and preprocessing integrity.

**Edge Cases Covered**:  
None (focuses on happy path).

### 2. `test_train_tfidf_svm_basic`

**Description**:  
Tests the full `train_tfidf_svm` function with a balanced dataset (3 samples per class). Validates TF-IDF vectorization, GridSearchCV (with reduced params for testing), calibration, predictions, probabilities, and cross-validation. Ensures the model can make new predictions.

**Test Data**:  
- `X` (9 text samples): Balanced across classes with hardware/software/network themes.  
  Example: `["computer not working hardware issue", "software application crash problem", ...]`  
- `y` (9 labels): `["Hardware", "Software", "Network", ...]` (3 per class).

**Execution Steps**:  
1. Call `train_tfidf_svm(X, y)` with reduced settings (e.g., `ngram_range=(1,2)`, `max_features=1000`, `cv=2`).  
2. Use the returned model to predict on a test sample: `["new hardware problem"]`.

**Assertions**:  
- Model and vectorizer are not `None`.  
- `y_pred` length equals input length (9).  
- `probabilities` length equals input length (9).  
- CV accuracy is between 0 and 1.  
- New predictions are generated successfully.

**Expected Outcome**:  
PASS – Ensures end-to-end training works with sufficient data.

**Edge Cases Covered**:  
- Balanced classes for safe CV (2-fold).

### 3. `test_train_tfidf_svm_minimal_data`

**Description**:  
Tests `train_tfidf_svm` with an imbalanced, minimal dataset (1 sample for "Software" and "Network", 2 for "Hardware"). Validates fallback behavior: skips GridSearchCV and calibration if `<2` samples per class, uses basic SVM fit, sets CV accuracy to 0.0, and still produces predictions/probabilities.

**Test Data**:  
- `X` (4 text samples):  
  `["computer hardware", "software program", "network connection", "hardware problem"]`  
- `y` (4 labels): `["Hardware", "Software", "Network", "Hardware"]`.

**Execution Steps**:  
1. Call `train_tfidf_svm(X, y)`.  
   - Detects `min_samples_per_class=1` → Uses basic `LinearSVC` fit (no GridSearchCV/calibration).  
   - Probabilities from `decision_function` (not calibrated `predict_proba`).  
   - Skips CV (sets `cv_accuracy=0.0`).  
2. Vectorization uses reduced settings.

**Assertions**:  
- Model and vectorizer are not `None`.  
- `y_pred` length equals input length (4).

**Expected Outcome**:  
PASS – Demonstrates robustness for production edge cases with sparse data.

**Edge Cases Covered**:  
- Imbalanced classes (`min_samples_per_class < 2`).  
- No CV or calibration to avoid errors.

### 4. `test_load_data_missing_columns`

**Description**:  
Ensures `load_data` raises a `ValueError` when the CSV lacks required columns (`Document` or `Topic_group`).

**Test Data**:  
- A temporary CSV file with invalid columns:  
  | Wrong_Column | Another_Column |
  |--------------|----------------|
  | test data    | test label     |

**Execution Steps**:  
1. Create and write to a temporary CSV file.  
2. Attempt `load_data(temp_file)` and catch the exception.  
3. Clean up the temporary file.

**Assertions**:  
- Raises `ValueError` with message matching: `"must contain 'Document' and 'Topic_group' columns"`.

**Expected Outcome**:  
PASS – Validates input validation and error handling.

**Edge Cases Covered**:  
- Malformed dataset structure.

## Running the Tests

Execute with:  
```bash
PYTHONPATH=. pytest tests/unit/ -v
```

- **Platform**: Linux (tested with Python 3.12.3, Pytest 8.4.2).  
- **Dependencies**: Handled via virtual environment (e.g., `.venv`).  
- **Output**: Verbose mode (`-v`) shows progress; all tests should PASS without warnings.  
- **Warnings**: May include scikit-learn deprecation notes (e.g., `cv='prefit'`), but do not affect results.

## Coverage and Limitations

- **Coverage**: Focuses on unit-level (functions); integration tests (e.g., full MLflow run) are out of scope.  
- **Limitations**: Tests use synthetic data; real dataset (`all_tickets_processed_improved_v3.csv`) is mocked. Reduced hyperparameters speed up runs but may differ from production.  
- **Future Enhancements**: Add tests for MLflow logging, signature inference, and larger datasets.

For issues, refer to the test file or scikit-learn docs on CV constraints.