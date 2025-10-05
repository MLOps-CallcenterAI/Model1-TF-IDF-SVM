# 📞 CallCenterAI — Ticket Classification API - TF-IDF + SVM

## Overview

CallCenterAI is a **FastAPI-based machine learning service** that classifies customer support tickets using **TF-IDF** vectorization and an **SVM model**.
It is designed for deployment in production using **Docker**, **DVC**, and **GitHub Actions CI/CD pipelines**.

---

## 📂 Project Structure

```
.
├── .github/workflows/        # CI/CD automation workflows
│   ├── Stage-CI.yaml
│   ├── build_and_push.yaml
│   ├── deploy.yaml
│   ├── get_latest_version.yaml
│   ├── lint.yaml
│   └── test.yaml
├── dataset/                  # Dataset versioned with DVC
│   └── all_tickets_processed_improved_v3.csv.dvc
├── docs/                     # Documentation
│   ├── github_actions/
│   │   └── Stage-CI.md
│   ├── notebooks/
│   │   └── clean&train.md
│   ├── src/
│   │   ├── app.md
│   │   └── model.md
│   └── tests/unit/
│       ├── test_app.md
│       └── test_model.md
├── notebooks/                # Jupyter notebooks for data cleaning and training
│   └── clean&train.ipynb
├── src/                      # Application source code
│   ├── __init__.py
│   ├── app.py
│   └── model.py
├── tests/                    # Unit tests
│   └── unit/
│       ├── test_app.py
│       └── test_model.py
├── Dockerfile                # Docker configuration
├── requirements              # Production dependencies
├── requirements.dev          # Development dependencies
├── README.md
└── .dvc, .gitignore, .dvcignore, .dockerignore
```

---

## ⚙️ Features

* **Machine Learning Inference API** with FastAPI

  * `/predict`: Batch prediction for multiple texts
  * `/predict_one`: Single text prediction
  * `/`: Health check endpoint

* **Model Management**

  * Trained using TF-IDF + SVM
  * Versioned with **MLflow** and integrated with **Databricks**

* **CI/CD Integration**

  * Automated testing, linting, and deployment via **GitHub Actions**
  * Docker image build & push with **version tagging**

* **Data Versioning**

  * Dataset managed with **DVC** for reproducibility

* **Documentation**

  * Clean and organized markdown docs for every project module

---

## 📘 Documentation Index

| Section            | Description                          | Link                                                                                                                       |
| ------------------ | ------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| **Application**    | FastAPI routes and app logic         | [docs/src/app.md](docs/src/app.md)                                                                                         |
| **Model**          | Model training and inference details | [docs/src/model.md](docs/src/model.md)                                                                                     |
| **Tests**          | Unit test documentation              | [docs/tests/unit/test_app.md](docs/tests/unit/test_app.md), [docs/tests/unit/test_model.md](docs/tests/unit/test_model.md) |
| **GitHub Actions** | CI/CD pipeline details               | [docs/github_actions/Stage-CI.md](docs/github_actions/Stage-CI.md)                                                         |
| **Notebook**       | Data cleaning & model training steps | [docs/notebooks/clean&train.md](docs/notebooks/clean&train.md)                                                             |

---

## 🧪 Running Tests

```bash
pytest tests/unit -v
```

---

## 🐳 Docker Usage

### Build the image

```bash
docker build -t callcenterai:latest .
```

### Run the container

```bash
docker run -p 8000:8000 callcenterai:latest
```

---

## 🚀 CI/CD Overview

* **Stage-CI.yaml**: Main CI workflow that triggers on `main` branch updates
* **build_and_push.yaml**: Builds and pushes Docker images
* **deploy.yaml**: Deploys to Kubernetes cluster
* **lint.yaml**: Runs code quality checks
* **test.yaml**: Executes automated unit tests
* **get_latest_version.yaml**: Handles semantic version tagging

---

## 🧠 Tech Stack

* **FastAPI**
* **Scikit-learn**
* **MLflow / Databricks**
* **Docker**
* **DVC**
* **GitHub Actions**

---