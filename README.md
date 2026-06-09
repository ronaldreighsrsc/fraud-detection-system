# Fraud Detection System

This repository implements an end-to-end **Fraud Detection System** using Anomaly Detection techniques from the NVIDIA Applications of AI for Anomaly Detection certification. It compares Deep Autoencoders, XGBoost Classifiers, and Isolation Forest models to identify anomalous transactions in financial/retail data.

The project demonstrates the full ML lifecycle: synthetic data generation, feature engineering, model training with anti-leakage cross-validation, comparative evaluation, and interactive visualization.

---

## Project Architecture

The code is structured following modularity and single-responsibility principles (SOLID), dividing the workflow into three sequential blocks:

```text
fraud-detection-system/
 |-- data/
 |   |-- raw/                    # Synthetic transactions (generated or external datasets)
 |   |-- processed/              # Engineered features ready for modeling
 |-- src/
 |   |-- preprocessing/          # Block 1: Data Synthesis, ETL, Feature Engineering
 |   |   |-- data_synthesizer.py # Realistic synthetic transaction generator
 |   |   |-- data_loader.py      # Data loading and validation
 |   |   |-- feature_engineer.py # Transaction-level feature engineering
 |   |-- models/                 # Block 2: Anomaly Detection Engines
 |   |   |-- autoencoder_deep.py # Deep Autoencoder (reconstruction error based)
 |   |   |-- autoencoder_lstm.py # LSTM Autoencoder for sequential patterns
 |   |   |-- gan_detector.py     # Generative Adversarial Network for anomaly scoring
 |   |   |-- xgb_detector.py     # XGBoost Classifier with Purged CV
 |   |   |-- isolation_forest.py # Baseline: Isolation Forest (unsupervised)
 |   |-- evaluation/             # Block 3: Model Tournament & Financial Analysis
 |   |   |-- metrics_engine.py   # Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR
 |   |   |-- threshold_analyzer.py # Optimal threshold via Precision-Recall curves
 |   |   |-- results/            # Output artifacts (.npy, .csv)
 |   |-- dashboard/              # Interactive Streamlit Dashboard
 |   |   |-- app.py              # Multi-page application
 |   |   |-- pages_utils.py      # Plotly visualization utilities
 |   |-- main_preprocessing.py   # Block 1 Orchestrator
 |   |-- main_training.py        # Block 2 Orchestrator (Model Tournament)
 |   |-- main_evaluation.py      # Block 3 Orchestrator (Comparative Results)
 |-- fastapii.py                 # Block 4 RESTful API (FastAPI)
 |-- .github/
 |   |-- workflows/
 |   |   |-- ci.yml              # Continuous Integration (lint + smoke test)
 |-- requirements.txt
 |-- README.md
```

---

## Installation and Setup

1. **Clone the repository** and open the terminal in the project root.
2. **Create and activate a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution Workflow (Pipeline)

To reproduce the experiments, execute the following scripts in order:

### Block 1: Data Preprocessing
Generates synthetic transaction data (100K transactions, ~2% fraud rate) with realistic fraud patterns (velocity attacks, geo-anomalies, amount spikes). Then applies feature engineering: customer statistics, velocity features, cyclical time encoding, and interaction features.
```bash
python src/main_preprocessing.py
```
*(Generates: `data/processed/transactions_engineered.csv`)*

### Block 2: Model Training
Trains three anomaly detection models with temporal train/test split (80/20). The Autoencoder is trained exclusively on normal transactions. XGBoost uses Purged K-Fold CV (López de Prado) with `scale_pos_weight` for class imbalance.
```bash
python src/main_training.py
```
*(Generates: `.npy` prediction files and `feature_importances.csv` in `src/evaluation/results/`)*

### Block 3: Comparative Evaluation
Runs the model tournament with accuracy, precision, recall, F1-score, AUC-ROC, AUC-PR, and estimated financial cost analysis (cost of false negatives vs false positives).
```bash
python src/main_evaluation.py
```
*(Generates: Tournament table in console and `fraud_tournament.csv`)*

### Interactive Dashboard (Visualization)
Launches a **Streamlit** web application with four interactive pages: Dataset Overview, Model Tournament, Anomaly Analysis (with interactive threshold slider), and Data Explorer.
```bash
streamlit run src/dashboard/app.py
```
*(Opens a local web server at `http://localhost:8501` with interactive Plotly charts).*

### Block 4: RESTful API Deployment (Real-Time Inference)
Exposes the 5 trained ML models (XGBoost, Isolation Forest, Deep Autoencoder, LSTM, GAN) through a high-performance **FastAPI** server. It utilizes `lifespan` for in-memory model caching and validates incoming JSON payloads representing transactions using **Pydantic**.
```bash
uvicorn fastapii:app --reload
```
*(Opens an interactive Swagger UI documentation at `http://127.0.0.1:8000/docs` to test the predictive endpoints).*

---

## Implemented Models

1. **Deep Autoencoder:** Symmetric architecture (Input→64→32→16→32→64→Input) trained only on normal transactions. Anomalies detected via reconstruction error exceeding a dynamic P95 threshold.
2. **LSTM Autoencoder:** Sequential deep learning model capturing the temporal velocity of transactions. Achieves state-of-the-art anomaly detection on time-series fraud.
3. **GAN Anomaly Detector:** Generative Adversarial Network where the Discriminator learns the genuine transaction distribution to flag out-of-distribution fraudulent attacks.
4. **XGBoost Classifier:** Gradient Boosting with `scale_pos_weight` for class imbalance, Purged & Embargoed K-Fold CV, and F1-Score optimization.
5. **Isolation Forest (Baseline):** Unsupervised ensemble method using random partitions to isolate anomalies.

---

## Synthetic Data Strategy

The `data_synthesizer.py` module generates realistic financial transactions profiling customers in **Santiago, Chile** with three fraud vectors:
- **Amount Spikes (40%):** Amounts 5-20x the customer's average.
- **Geo-Anomalies (35%):** Transactions 50-500 km from the customer's home.
- **Velocity Attacks (25%):** Unusual hours (midnight-5am) with atypical merchants.

> **Real-World Compatibility:** The pipeline is designed to be compatible with the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) dataset from Kaggle. To use real data, simply replace the CSV in `data/raw/` and update the column mappings in `data_loader.py`.

---

## Certifications & Methodology

- **NVIDIA:** Applications of AI for Anomaly Detection
- **Anti-Leakage:** Purged K-Fold Cross-Validation with Embargo (López de Prado, 2018)
- **Evaluation:** Multi-metric tournament with financial cost analysis

---

## 🚀 V2.0: Big Data & GPU Scalability (NVIDIA RAPIDS)

While the current main branch is designed for CPU execution (Pandas/Scikit-Learn) to ensure maximum compatibility for anyone cloning the repo, this architecture is fully prepared to scale to massive financial datasets (e.g., millions of transactions per day).

To achieve extreme performance without CPU-to-GPU memory bottlenecks, the ETL and training pipelines can be migrated to **NVIDIA RAPIDS**:

1. **cuDF (GPU Pandas):** Replaces traditional Pandas for Feature Engineering (rolling statistics, customer behavior aggregation). Data is loaded directly into GPU VRAM.
2. **Apache Arrow:** Eliminates serialization overhead. Data remains in the GPU memory while moving from the cuDF preprocessing step directly into the model.
3. **XGBoost (GPU Accelerated):** Uses 	ree_method='gpu_hist' to train on the GPU directly from the cuDF data structure, dropping training times from hours to seconds even with millions of rows.

*Note: An experimental implementation of this GPU pipeline is available in the eature/rapids-gpu-scaling branch.*
