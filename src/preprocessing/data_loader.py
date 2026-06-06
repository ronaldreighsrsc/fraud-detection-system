import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")


class DataLoader:
    """
    Módulo de carga y validación de datos transaccionales.
    Soporta tanto datos sintéticos generados por DataSynthesizer como
    datasets externos (e.g., IEEE-CIS Fraud Detection de Kaggle).
    """

    def __init__(self, raw_data_path: str = "./data/raw/"):
        self.raw_data_path = raw_data_path

    def load_transactions(self, filename: str = "transactions.csv") -> pd.DataFrame:
        """
        Carga el dataset de transacciones desde un archivo CSV.
        Aplica validaciones básicas de integridad.
        """
        filepath = os.path.join(self.raw_data_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"❌ No se encontró el archivo: {filepath}\n"
                f"   Ejecuta primero 'main_preprocessing.py' para generar los datos sintéticos."
            )

        print(f"📂 Cargando datos desde: {filepath}")
        df = pd.read_csv(filepath)

        # Validación de columnas requeridas
        required_cols = [
            'customer_id', 'transaction_amount', 'hour_of_day',
            'day_of_week', 'merchant_category', 'distance_from_home',
            'is_international', 'is_fraud'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"❌ Columnas faltantes en el dataset: {missing}")

        # Parsear timestamp si existe
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        self._print_summary(df)
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Imprime un resumen ejecutivo del dataset cargado."""
        n_total = len(df)
        n_fraud = df['is_fraud'].sum()
        n_legit = n_total - n_fraud
        ratio = n_fraud / n_total * 100

        print(f"  📊 Resumen del dataset:")
        print(f"     Total de transacciones: {n_total:,}")
        print(f"     Legítimas: {n_legit:,} ({100 - ratio:.1f}%)")
        print(f"     Fraudulentas: {n_fraud:,} ({ratio:.1f}%)")
        print(f"     Clientes únicos: {df['customer_id'].nunique():,}")

        if 'timestamp' in df.columns:
            print(f"     Rango temporal: {df['timestamp'].min()} → {df['timestamp'].max()}")

        # Check de NaN
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"  ⚠️  Valores nulos detectados: {nan_count}")
        else:
            print(f"     ✅ Sin valores nulos")
