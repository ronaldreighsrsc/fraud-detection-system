import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.feature_engineer import TransactionFeatureEngineer
from models.autoencoder_deep import FraudDeepAutoencoder
from models.autoencoder_lstm import FraudLSTMAutoencoder
from models.gan_detector import FraudGANDetector
from models.xgb_detector import FraudXGBoostDetector
from models.isolation_forest import AnomalyIsolationForest
import warnings

warnings.filterwarnings("ignore")


def run_training_pipeline():
    """
    Bloque 2: Pipeline de Entrenamiento de Modelos.
    Carga el dataset procesado, divide en train/test respetando
    la separación temporal, y entrena los tres modelos de detección.
    """
    print("🚀 Iniciando Pipeline de Entrenamiento (Bloque 2)...")

    # --- PASO 1: Carga del Dataset Procesado ---
    print("\n--- PASO 1: Cargando Dataset Procesado ---")
    processed_path = "./data/processed/transactions_engineered.csv"
    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"❌ Dataset procesado no encontrado en {processed_path}\n"
            "   Ejecuta primero: python src/main_preprocessing.py"
        )

    df = pd.read_csv(processed_path)
    print(f"  📊 Dataset cargado: {len(df):,} registros")

    # --- PASO 2: Preparación de Features ---
    print("\n--- PASO 2: Preparación de Features ---")
    engineer = TransactionFeatureEngineer()
    feature_cols = engineer.get_feature_columns()

    # Validar que las columnas existen
    available_features = [c for c in feature_cols if c in df.columns]
    print(f"  📐 Features seleccionadas: {len(available_features)}")

    X = df[available_features].replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = df['is_fraud'].values

    # División temporal (80/20) - respetamos el orden temporal
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"  📦 Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  📦 Fraudes en Train: {y_train.sum():,} | Fraudes en Test: {y_test.sum():,}")

    os.makedirs("./src/evaluation/results", exist_ok=True)
    os.makedirs("./models/saved_models", exist_ok=True)

    # --- MODELO 1A: Deep Autoencoder ---
    print("\n--- MODELO 1A: Deep Autoencoder ---")
    autoencoder_deep = FraudDeepAutoencoder(encoding_dim=16, epochs=100, batch_size=256)

    # Entrenar SOLO con datos normales
    X_train_normal = X_train[y_train == 0]
    ae_info = autoencoder_deep.fit(X_train_normal)
    autoencoder_deep.save("./models/saved_models/ae_deep.pkl")

    ae_preds, ae_scores = autoencoder_deep.predict(X_test)
    np.save("./src/evaluation/results/ae_deep_predictions.npy", ae_preds)
    np.save("./src/evaluation/results/ae_deep_scores.npy", ae_scores)
    print(f"  💾 Predicciones del Deep Autoencoder guardadas")

    # --- MODELO 1B: LSTM Autoencoder ---
    print("\n--- MODELO 1B: LSTM Autoencoder ---")
    autoencoder_lstm = FraudLSTMAutoencoder(encoding_dim=16, epochs=50, batch_size=256)

    ae_lstm_info = autoencoder_lstm.fit(X_train_normal)
    autoencoder_lstm.save("./models/saved_models/ae_lstm.pkl")

    ae_lstm_preds, ae_lstm_scores = autoencoder_lstm.predict(X_test)
    np.save("./src/evaluation/results/ae_lstm_predictions.npy", ae_lstm_preds)
    np.save("./src/evaluation/results/ae_lstm_scores.npy", ae_lstm_scores)
    print(f"  💾 Predicciones del LSTM Autoencoder guardadas")

    # --- MODELO 1C: GAN Anomaly Detector ---
    print("\n--- MODELO 1C: GAN Anomaly Detector ---")
    gan_detector = FraudGANDetector(latent_dim=16, epochs=60, batch_size=256)

    gan_info = gan_detector.fit(X_train_normal)
    gan_detector.save("./models/saved_models/gan.pkl")

    gan_preds, gan_scores = gan_detector.predict(X_test)
    np.save("./src/evaluation/results/gan_predictions.npy", gan_preds)
    np.save("./src/evaluation/results/gan_scores.npy", gan_scores)
    print(f"  💾 Predicciones de la GAN guardadas")

    # --- MODELO 2: XGBoost ---
    print("\n--- MODELO 2: XGBoost Classifier ---")
    xgb_detector = FraudXGBoostDetector(n_splits=5, purge_size=50, embargo_size=10)

    best_params = xgb_detector.find_best_params(X_train, y_train)
    xgb_detector.train(X_train, y_train, best_params)
    xgb_detector.save("./models/saved_models/xgb.pkl")

    xgb_preds, xgb_probs = xgb_detector.predict(X_test)
    np.save("./src/evaluation/results/xgb_predictions.npy", xgb_preds)
    np.save("./src/evaluation/results/xgb_probs.npy", xgb_probs)

    # Guardar importancias
    importances = xgb_detector.get_feature_importances(available_features)
    importances.to_csv("./src/evaluation/results/feature_importances.csv", index=False)
    print(f"  💾 Predicciones e importancias del XGBoost guardadas")

    # --- MODELO 3: Isolation Forest ---
    print("\n--- MODELO 3: Isolation Forest (Baseline) ---")
    iso_forest = AnomalyIsolationForest(contamination=0.02, n_estimators=200)
    iso_forest.fit(X_train)
    iso_forest.save("./models/saved_models/iso.pkl")

    iso_preds, iso_scores = iso_forest.predict(X_test)
    np.save("./src/evaluation/results/iso_predictions.npy", iso_preds)
    np.save("./src/evaluation/results/iso_scores.npy", iso_scores)
    print(f"  💾 Predicciones del Isolation Forest guardadas")

    # Guardar labels de test
    np.save("./src/evaluation/results/y_test.npy", y_test)

    print(f"\n✅ ¡Pipeline de entrenamiento completado!")
    print(f"📁 Artefactos guardados en: ./src/evaluation/results/")


if __name__ == "__main__": #??
    run_training_pipeline()
