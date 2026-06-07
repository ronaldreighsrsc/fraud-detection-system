import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


class FraudDeepAutoencoder:
    """
    Deep Autoencoder para Detección de Anomalías Financieras.ado en error de reconstrucción.

    Arquitectura: Input → 64 → 32 → 16 (espacio latente) → 32 → 64 → Input
    Entrenado EXCLUSIVAMENTE con transacciones legítimas (clase 0).
    Las transacciones fraudulentas generan un error de reconstrucción alto,
    lo que permite detectarlas como anomalías.

    Inspirado en: NVIDIA Applications of AI for Anomaly Detection.
    """

    def __init__(self, encoding_dim: int = 16, epochs: int = 100,
                 batch_size: int = 256, learning_rate: float = 1e-3,
                 random_state: int = 42):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.encoder = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.history = None

    def _build_model(self, input_dim: int) -> keras.Model:
        """
        Construye la arquitectura del Autoencoder simétrico.
        Usa BatchNormalization y Dropout para regularización.
        """
        tf.random.set_seed(self.random_state)

        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        x = keras.layers.Dense(64, activation='relu')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu', name='latent_space')(x)

        # Decoder (simétrico al encoder)
        x = keras.layers.Dense(32, activation='relu')(encoded)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        decoded = keras.layers.Dense(input_dim, activation='linear')(x)

        autoencoder = keras.Model(inputs, decoded, name='fraud_deep_autoencoder')
        encoder = keras.Model(inputs, encoded, name='deep_encoder')

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )

        return autoencoder, encoder

    def fit(self, X_train_normal: np.ndarray) -> dict:
        """
        Entrena el Autoencoder SOLO con datos normales (sin fraude).
        Luego calcula el threshold óptimo basado en el percentil 95
        del error de reconstrucción sobre los datos de entrenamiento.
        """
        print("  🧠 Entrenando Deep Autoencoder (solo transacciones legítimas)...")

        # Escalamiento
        X_scaled = self.scaler.fit_transform(X_train_normal)

        # Construir modelo
        self.model, self.encoder = self._build_model(X_scaled.shape[1])

        # Entrenamiento con Early Stopping
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )

        self.history = self.model.fit(
            X_scaled, X_scaled,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )

        # Calcular threshold dinámico
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
        self.threshold = np.percentile(mse, 95)

        print(f"  ✅ Autoencoder entrenado ({len(self.history.history['loss'])} épocas)")
        print(f"  📏 Threshold de anomalía (P95): {self.threshold:.6f}")

        return {
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'threshold': self.threshold,
        }

    def predict(self, X: np.ndarray) -> tuple:
        """
        Predice anomalías calculando el error de reconstrucción.
        Retorna: (predicciones binarias, scores continuos de anomalía).
        """
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

        predictions = (mse > self.threshold).astype(int)
        return predictions, mse

    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """Retorna los errores de reconstrucción crudos (para análisis de threshold)."""
        X_scaled = self.scaler.transform(X)
        reconstructed = self.model.predict(X_scaled, verbose=0)
        return np.mean(np.power(X_scaled - reconstructed, 2), axis=1)

    def get_latent_representation(self, X: np.ndarray) -> np.ndarray:
        """Extrae las representaciones del espacio latente (para visualización)."""
        X_scaled = self.scaler.transform(X)
        return self.encoder.predict(X_scaled, verbose=0)
