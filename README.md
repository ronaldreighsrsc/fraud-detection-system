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
 |   |   |-- autoencoder.py      # Deep Autoencoder (reconstruction error based)
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

---

## Implemented Models

1. **Deep Autoencoder:** Symmetric architecture (Input→64→32→16→32→64→Input) trained only on normal transactions. Anomalies detected via reconstruction error exceeding a dynamic P95 threshold.
2. **XGBoost Classifier:** Gradient Boosting with `scale_pos_weight` for class imbalance, Purged & Embargoed K-Fold CV, and F1-Score optimization.
3. **Isolation Forest (Baseline):** Unsupervised ensemble method using random partitions to isolate anomalies.

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
