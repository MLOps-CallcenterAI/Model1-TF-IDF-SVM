# Stage CI Pipeline Documentation

This document explains the purpose, structure, and behind-the-scenes details of the **Stage CI Pipeline** implemented in [`.github/workflows/Stage-CI.yaml`](https://github.com/MLOps-CallcenterAI/Model1-TF-IDF-SVM/blob/main/.github/workflows/Stage-CI.yaml).

---

## Overview

The **Stage CI Pipeline** automates the build, test, lint, Docker image creation, version management, and deployment for the Model1-TF-IDF-SVM project. It coordinates several modular workflow files to ensure code quality, reproducibility, and seamless updates to infrastructure.

### Trigger Conditions

The pipeline runs on:

- Pushes to the `main` branch affecting:
  - files in `src/` and `test/` directories
  - `Dockerfile`
  - `requirements` and `requirements.dev`
- Manual invocation via **workflow_dispatch**

---

## Pipeline Structure

The workflow orchestrates the following jobs:

1. **Lint**
2. **Test**
3. **Build_and_Push**
4. **Get_Latest_Version**
5. **Deploy**

Each job leverages modular workflow files for maintainability and reuse.

---

### 1. Lint

**Workflow Used:** [`.github/workflows/lint.yaml`](https://github.com/MLOps-CallcenterAI/Model1-TF-IDF-SVM/blob/main/.github/workflows/lint.yaml)

**Steps:**
- Checks out code
- Sets up Python 3.11 environment
- Installs code quality tools (`black`, `flake8`, `isort`)
- Runs:
  - `black` for code style
  - `flake8` for linting (ignores E203, max line length 110)
  - `isort` for import order

**Purpose:**  
Ensures code adheres to style and linting guidelines before further steps.

---

### 2. Test

**Workflow Used:** [`.github/workflows/test.yaml`](https://github.com/MLOps-CallcenterAI/Model1-TF-IDF-SVM/blob/main/.github/workflows/test.yaml)

**Steps:**
- Checks out code
- Sets up Python 3.11
- Installs dependencies (from `requirements.dev` if available)
- Creates a `.env` file using environment variables for Databricks and MLflow
- Masks secrets in logs
- Runs unit tests with `pytest` in the `tests/unit/` directory

**Purpose:**  
Validates code correctness using unit tests in a controlled environment.

---

### 3. Build_and_Push

**Workflow Used:** [`.github/workflows/build_and_push.yaml`](https://github.com/MLOps-CallcenterAI/Model1-TF-IDF-SVM/blob/main/.github/workflows/build_and_push.yaml)

**Secrets Required:**  
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

**Steps:**
- Checks out code
- Sets up Docker Buildx
- Logs into Docker Hub
- Determines the next Docker image version:
  - Fetches tags from Docker Hub for `medhedimaaroufi/callcenterai-svm`
  - Identifies the latest tag matching `v1.0.*` pattern
  - Calculates the next minor version (e.g., `v1.0.3` → `v1.0.4`)
- Builds and pushes the image to Docker Hub with the new tag
- Runs a Trivy security scan on the newly built image (fails if critical vulnerabilities are found)

**Purpose:**  
Automates Docker image versioning, building, security scanning, and publishing.

---

### 4. Get_Latest_Version

**Workflow Used:** [`.github/workflows/get_latest_version.yaml`](https://github.com/MLOps-CallcenterAI/Model1-TF-IDF-SVM/blob/main/.github/workflows/get_latest_version.yaml)

**Secrets Required:**  
- `DOCKER_USERNAME`
- `DOCKER_PASSWORD`

**Steps:**
- Checks out code
- Fetches the latest Docker image tag from Docker Hub matching `v1.0.*`

**Purpose:**  
Makes the latest image version available for deployment to infrastructure.

---

### 5. Deploy

**Workflow Used:** [`.github/workflows/deploy.yaml`](https://github.com/MLOps-CallcenterAI/Model1-TF-IDF-SVM/blob/main/.github/workflows/deploy.yaml)

**Secrets Required:**  
- `TOKEN` (GitHub token for infrastructure repo access)

**Inputs:**
- `filename`: Manifest file to update (e.g., `manifests/default/svm.yaml`)
- `image_version`: Latest Docker image tag (from previous job)

**Steps:**
- Checks out the infrastructure configuration repo (`MLOps-CallcenterAI/Infrastructure-Configuration`)
- Sets up git
- Runs a script (`scripts/update_image.sh`) to update the manifest file with the new Docker image version
- Commits and pushes the change to the infrastructure repo

**Purpose:**  
Updates infrastructure manifests to deploy the latest model image automatically.

---

## Behind the Scenes: How It All Fits Together

- **Modular Design:**  
  Each major task (linting, testing, building, version management, deployment) is encapsulated in its own workflow, called by the Stage CI pipeline for clarity and reusability.

- **Version Automation:**  
  The pipeline automatically bumps the Docker image version based on existing tags in Docker Hub, ensuring each build is uniquely tagged.

- **Security:**  
  Every Docker build is scanned with Trivy for vulnerabilities before being published.

- **Infrastructure Sync:**  
  Upon successful build and push, the infrastructure configuration is updated to use the latest image, ensuring deployments always reflect the current codebase.

- **Secrets Management:**  
  Sensitive credentials (Docker Hub, GitHub tokens, Databricks info) are handled securely via GitHub Secrets and masked in logs.

---

## Diagram

```
Trigger (push/dispatch)
        |
        v
+-------------------+
|   Stage CI        |
+-------------------+
        |
        v
+-------+-------+-------+-------+
| Lint | Test | Build | Version|
+-------+-------+-------+-------+
        |
        v
    Deploy (Infra repo)
```

---

## Summary

The Stage CI Pipeline provides robust, automated quality checks, versioning, security, and deployment for the Model1-TF-IDF-SVM project, streamlining model development and release into production infrastructure.
