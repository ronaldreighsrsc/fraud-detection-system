import os
import warnings
from preprocessing.data_synthesizer import TransactionSynthesizer
from preprocessing.data_loader import DataLoader
from preprocessing.feature_engineer import TransactionFeatureEngineer

warnings.filterwarnings("ignore")


def run_preprocessing_pipeline():
    """
    Bloque 1: Pipeline de Preprocesamiento de Datos.
    Genera datos sintéticos (si no existen), los carga, valida,
    y aplica Feature Engineering transaccional.
    """
    print("🚀 Iniciando Pipeline de Preprocesamiento (Bloque 1)...")

    # --- PASO 1: Generación de Datos Sintéticos ---
    raw_path = "./data/raw/transactions.csv"
    if not os.path.exists(raw_path):
        print("\n--- PASO 1: Generación de Datos Sintéticos ---")
        os.makedirs("./data/raw", exist_ok=True)
        synthesizer = TransactionSynthesizer(
            n_customers=1000,
            n_transactions=100_000,
            fraud_ratio=0.02,
        )
        synthesizer.generate(output_path=raw_path)
    else:
        print(f"\n--- PASO 1: Datos ya existen en {raw_path} (skip generación) ---")

    # --- PASO 2: Carga y Validación ---
    print("\n--- PASO 2: Carga y Validación de Datos ---")
    loader = DataLoader(raw_data_path="./data/raw/")
    df = loader.load_transactions()

    # --- PASO 3: Feature Engineering ---
    print("\n--- PASO 3: Feature Engineering Transaccional ---")
    engineer = TransactionFeatureEngineer()
    df = engineer.engineer_features(df)

    # --- PASO 4: Guardado ---
    print("\n--- PASO 4: Guardando Dataset Procesado ---")
    os.makedirs("./data/processed", exist_ok=True)
    output_path = "./data/processed/transactions_engineered.csv"
    df.to_csv(output_path, index=False)

    print(f"\n✅ ¡Pipeline de preprocesamiento completado!")
    print(f"📁 Dataset enriquecido guardado en: {output_path}")


if __name__ == "__main__":
    run_preprocessing_pipeline()
