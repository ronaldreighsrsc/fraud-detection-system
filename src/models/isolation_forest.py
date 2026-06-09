import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
import joblib

warnings.filterwarnings("ignore")


class AnomalyIsolationForest:
    """
    Modelo baseline de Isolation Forest para detección de anomalías.
    Método no supervisado que aísla observaciones construyendo
    particiones aleatorias en el espacio de features.

    Sirve como benchmark contra los modelos más complejos
    (Autoencoder y XGBoost) para validar que la complejidad
    adicional aporta valor predictivo.
    """

    def __init__(self, contamination: float = 0.02, n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X_train: np.ndarray) -> None:
        """
        Entrena el Isolation Forest con los datos completos.
        El parámetro `contamination` controla el porcentaje esperado de anomalías.
        """
        print("  🌲 Entrenando Isolation Forest (baseline no supervisado)...")

        X_scaled = self.scaler.fit_transform(X_train)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)

        print(f"  ✅ Isolation Forest entrenado ({self.n_estimators} árboles, "
              f"contamination={self.contamination:.2%})")

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predice anomalías.
        Isolation Forest retorna -1 para anomalías y 1 para normales.
        Convertimos a: 1 = fraude, 0 = normal (consistente con los otros modelos).
        Retorna: (predicciones binarias, anomaly scores).
        """
        X_scaled = self.scaler.transform(X)

        raw_predictions = self.model.predict(X_scaled)
        predictions = np.where(raw_predictions == -1, 1, 0)

        # Score de anomalía (más negativo = más anómalo)
        anomaly_scores = -self.model.decision_function(X_scaled)

        return predictions, anomaly_scores

    def save(self, filepath: str) -> None:
        """Guarda el modelo y scaler en un archivo."""
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        
        state = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(state, filepath)
        print(f"  💾 Modelo Isolation Forest guardado en {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Carga un modelo guardado previamente."""
        state = joblib.load(filepath)
        
        instance = cls()
        instance.model = state['model']
        instance.scaler = state['scaler']
        
        return instance
