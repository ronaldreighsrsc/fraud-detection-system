import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings

warnings.filterwarnings("ignore")


class TransactionFeatureEngineer:
    """
    Ingeniería de features para detección de fraude.
    Calcula variables derivadas que capturan el comportamiento transaccional
    del cliente y facilitan la identificación de patrones anómalos.

    Features generadas:
        - amount_deviation: desviación del monto respecto al promedio del cliente.
        - amount_zscore: z-score del monto por cliente.
        - tx_frequency_1h: nro. de transacciones del cliente en la última hora.
        - tx_frequency_24h: nro. de transacciones del cliente en las últimas 24h.
        - amount_to_median_ratio: ratio del monto vs mediana del cliente.
        - hour_sin / hour_cos: encoding cíclico de la hora del día.
        - day_sin / day_cos: encoding cíclico del día de la semana.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._customer_stats = None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pipeline principal de feature engineering.
        Recibe el DataFrame crudo y retorna el DataFrame enriquecido.
        """
        print("🔧 Iniciando Feature Engineering transaccional...")
        df = df.copy()

        # Paso 1: Estadísticas por cliente
        df = self._add_customer_stats(df)
        print("  ✅ Estadísticas por cliente calculadas")

        # Paso 2: Features de velocidad (frequency)
        df = self._add_velocity_features(df)
        print("  ✅ Features de velocidad calculadas")

        # Paso 3: Encoding cíclico temporal
        df = self._add_cyclical_encoding(df)
        print("  ✅ Encoding cíclico temporal aplicado")

        # Paso 4: Features de interacción
        df = self._add_interaction_features(df)
        print("  ✅ Features de interacción creadas")

        print(f"  📐 Dimensiones finales: {df.shape[0]:,} filas × {df.shape[1]} columnas")
        return df

    def _add_customer_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula estadísticas de gasto por cliente y las cruza con cada transacción."""
        stats = df.groupby('customer_id')['transaction_amount'].agg(
            ['mean', 'std', 'median']
        ).rename(columns={'mean': 'cust_mean', 'std': 'cust_std', 'median': 'cust_median'})
        stats['cust_std'] = stats['cust_std'].fillna(1.0)  # Evitar división por 0

        df = df.merge(stats, on='customer_id', how='left')

        # Desviación absoluta y z-score respecto al comportamiento del cliente
        df['amount_deviation'] = df['transaction_amount'] - df['cust_mean']
        df['amount_zscore'] = df['amount_deviation'] / df['cust_std']
        df['amount_to_median_ratio'] = df['transaction_amount'] / df['cust_median'].replace(0, 1)

        # Limpiar columnas auxiliares
        df = df.drop(columns=['cust_mean', 'cust_std', 'cust_median'])

        self._customer_stats = stats
        return df

    def _add_velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula frecuencia de transacciones por cliente en ventanas temporales.
        Si no hay timestamp, usa una aproximación basada en posición.
        """
        if 'timestamp' in df.columns:
            df = df.sort_values(['customer_id', 'timestamp'])
            # Ventana de 1 hora
            df['tx_frequency_1h'] = df.groupby('customer_id')['timestamp'].transform(
                lambda x: x.diff().dt.total_seconds().fillna(9999).apply(
                    lambda s: 1 if s < 3600 else 0
                ).rolling(5, min_periods=1).sum()
            )
            # Ventana de 24 horas
            df['tx_frequency_24h'] = df.groupby('customer_id')['timestamp'].transform(
                lambda x: x.diff().dt.total_seconds().fillna(9999).apply(
                    lambda s: 1 if s < 86400 else 0
                ).rolling(20, min_periods=1).sum()
            )
        else:
            # Fallback: frecuencia basada en conteo simple
            freq = df.groupby('customer_id').cumcount()
            df['tx_frequency_1h'] = freq % 5
            df['tx_frequency_24h'] = freq % 20

        return df

    def _add_cyclical_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encoding cíclico para capturar la naturaleza circular de horas y días."""
        # Hora del día → seno/coseno (período = 24h)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)

        # Día de la semana → seno/coseno (período = 7 días)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de interacción que combinan variables para capturar patrones complejos."""
        # Monto * distancia: fraude suele ser monto alto + distancia alta
        df['amount_x_distance'] = df['transaction_amount'] * df['distance_from_home']

        # Monto * flag internacional
        df['amount_x_international'] = df['transaction_amount'] * df['is_international']

        # Z-score * frecuencia: anomalía de monto combinada con velocidad
        df['zscore_x_frequency'] = df['amount_zscore'] * df['tx_frequency_1h']

        return df

    def get_feature_columns(self) -> list:
        """Retorna la lista de features seleccionadas para el modelado."""
        return [
            'transaction_amount', 'hour_of_day', 'day_of_week',
            'merchant_category', 'distance_from_home', 'is_international',
            'amount_deviation', 'amount_zscore', 'amount_to_median_ratio',
            'tx_frequency_1h', 'tx_frequency_24h',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'amount_x_distance', 'amount_x_international', 'zscore_x_frequency',
        ]

    def scale_features(self, df: pd.DataFrame, feature_cols: list,
                       fit: bool = True) -> np.ndarray:
        """Escala las features con StandardScaler, limpiando Infs y NaNs."""
        X = df[feature_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)
