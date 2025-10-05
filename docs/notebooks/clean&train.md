# 🧹 Clean & Train Notebook Documentation

## 📁 Overview

This notebook handles **data exploration**, **text preprocessing**, **visual analysis**, and **training of a TF-IDF + SVM classifier** for ticket categorization.
It also integrates **MLflow** for model tracking and Databricks experiment management.

---

## ⚙️ 1. Setup & Imports

Main libraries used:

* **pandas**, **matplotlib**, **seaborn** → data handling & visualization
* **nltk** → text cleaning, stopword removal, and lemmatization
* **sklearn** → feature extraction, model training, and evaluation
* **mlflow** → experiment tracking
* **dotenv** → environment configuration for Databricks connection

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
import string
```

The notebook loads:

```python
df = pd.read_csv("../dataset/all_tickets_processed_improved_v3.csv")
```

---

## 🧾 2. Dataset Overview

Displays dataset information:

```python
df.shape, df.columns
df.head()
df.info()
df.isnull().sum()
```

These outputs allow checking:

* Dataset dimensions
* Missing values
* Data type distribution

---

## 📊 3. Ticket Category Distribution

Bar chart of ticket counts by category:

```python
topic_counts = df["Topic_group"].value_counts()
sns.barplot(x=topic_counts.values, y=topic_counts.index, palette="viridis")
```

Pie chart of percentage distribution:

```python
plt.pie(
    topic_counts.values,
    labels=topic_counts.index,
    autopct="%1.1f%%",
    startangle=90,
)
```

---

## 📈 4. Descriptive Statistics

Provides numeric and categorical summaries:

* Total tickets
* Number of unique categories
* Most and least frequent categories
* Percentage distribution per category

---

## 🧮 5. Document Length Analysis

New column added:

```python
df["doc_length"] = df["Document"].str.len()
```

Two plots are produced:

* **Boxplot** of document length by category
* **Histogram** of overall document length distribution

Followed by statistics on mean, median, max, and min lengths.

---

## 🗣️ 6. Frequent Words by Category

Function `plot_top_words_category(category_name, top_n=20)`
displays the top N most common non-stopwords for each category using NLTK stopword filtering.

```python
plot_top_words_category("Technical Support", top_n=20)
```

This helps identify topic-specific vocabulary.

---

## 📜 7. Summary & Recommendations

The notebook prints a summary highlighting:

* Dataset size & class balance
* Average text length
* Recommended preprocessing & model approaches

**Recommendations include:**

* Handling class imbalance if detected
* Cleaning and normalizing text data
* Using TF-IDF + SVM or Transformer models
* Stratified cross-validation for fair evaluation

---

## 🧠 8. Preprocessing & Model Training

### Load & Preprocess Function

```python
def load_data(dataset_path):
    # text cleaning + lemmatization
```

Steps include:

* Removing punctuation
* Tokenizing and lemmatizing words
* Removing stopwords (`english` + domain-specific ones like “please”, “ticket”, “help”)

Output:

```python
X, y = load_data()
```

---

## ⚙️ 9. TF-IDF + SVM Training

Function `train_tfidf_svm(X, y)` performs:

1. **TF-IDF vectorization**
2. **GridSearchCV** for best SVM regularization parameter C
3. **CalibratedClassifierCV** for probability calibration
4. **5-fold Stratified cross-validation**

Returns:

```python
model, vectorizer, y_true, y_pred, probs, cv_accuracy
```

---

## 🧩 10. Model Evaluation

Metrics computed:

```python
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
```

Prints:

```
✅ Accuracy: 0.XXXX
✅ F1 Score: 0.XXXX
✅ CV Accuracy: 0.XXXX
```

---

## 🚀 11. MLflow Tracking (Databricks)

Environment variables loaded from `.env`:

```bash
url=...
access_token=...
EXPERIMENT_NAME=...
```

Configuration:

```python
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment(EXPERIMENT_NAME)
```

Model version incremented automatically, and the run is logged with:

* **Parameters:** model type, hyperparameters
* **Metrics:** accuracy, F1, cross-validation accuracy
* **Artifacts:** serialized model pipeline

```python
mlflow.sklearn.log_model(
    pipeline,
    artifact_path="tfidf_svm_model",
    signature=signature,
    input_example=input_example,
    registered_model_name="workspace.default.tfidf_svm_classifier",
)
```

---

## 🧾 12. Key Outcomes

| Component               | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| **Dataset**             | All preprocessed tickets from v3                          |
| **Model**               | TF-IDF + LinearSVC (Calibrated)                           |
| **Vectorizer**          | n-gram (1,3), max_features=10000                          |
| **Metrics**             | Accuracy, F1, CV Accuracy                                 |
| **Experiment Tracking** | MLflow on Databricks                                      |
| **Output**              | Registered model `workspace.default.tfidf_svm_classifier` |

---
