import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import warnings
import joblib

warnings.filterwarnings("ignore")

class FraudLSTMAutoencoder:
    """
    LSTM Autoencoder para Detección de Fraude en Secuencias Transaccionales.

    Arquitectura: Input -> Reshape(3D) -> LSTM Encoder -> RepeatVector -> LSTM Decoder -> Dense
    Entrenado EXCLUSIVAMENTE con datos de operación NORMAL.
    """

    def __init__(self, encoding_dim: int = 16, epochs: int = 50,
                 batch_size: int = 256, learning_rate: float = 1e-3,
                 random_state: int = 42):
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.history = None
        self.threshold = None

    def _build_model(self, input_dim: int) -> keras.Model:
        """Construye la arquitectura LSTM Autoencoder."""
        tf.random.set_seed(self.random_state)

        inputs = keras.Input(shape=(1, input_dim))
        
        # Encoder
        encoded = keras.layers.LSTM(32, activation='relu', return_sequences=False)(inputs)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(encoded)

        # Decoder
        decoded = keras.layers.RepeatVector(1)(encoded)
        decoded = keras.layers.LSTM(32, activation='relu', return_sequences=True)(decoded)
        decoded = keras.layers.TimeDistributed(keras.layers.Dense(input_dim, activation='linear'))(decoded)

        autoencoder = keras.Model(inputs, decoded, name='fraud_lstm_autoencoder')

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse'
        )
        return autoencoder

    def fit(self, X_train_normal: np.ndarray) -> dict:
        """
        Entrena el LSTM Autoencoder.
        """
        print("  🧠 Entrenando LSTM Autoencoder (solo transacciones legítimas)...")

        X_scaled = self.scaler.fit_transform(X_train_normal)
        X_3d = np.expand_dims(X_scaled, axis=1)

        self.model = self._build_model(X_scaled.shape[1])

        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        self.history = self.model.fit(
            X_3d, X_3d,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )

        # Calcular umbral P99
        reconstructed = self.model.predict(X_3d, verbose=0)
        mse = np.mean(np.power(X_3d - reconstructed, 2), axis=(1, 2))
        self.threshold = np.percentile(mse, 99)

        print(f"  ✅ LSTM Autoencoder entrenado")
        print(f"  📏 MSE base (P99): {self.threshold:.6f}")

        return {
            'epochs_trained': len(self.history.history['loss']),
            'final_loss': self.history.history['loss'][-1]
        }

    def predict(self, X: np.ndarray) -> tuple:
        """Predice anomalías basadas en el Error de Reconstrucción."""
        X_scaled = self.scaler.transform(X)
        X_3d = np.expand_dims(X_scaled, axis=1)
        reconstructed = self.model.predict(X_3d, verbose=0)
        mse = np.mean(np.power(X_3d - reconstructed, 2), axis=(1, 2))
        
        predictions = (mse > self.threshold).astype(int)
        return predictions, mse

    def save(self, filepath: str) -> None:
        """Guarda el modelo, scaler y threshold en disco."""
        if self.model is None:
            raise ValueError("No hay modelo entrenado para guardar.")
        
        keras_path = filepath.replace(".pkl", ".keras")
        self.model.save(keras_path)
        
        state = {
            'scaler': self.scaler,
            'threshold': self.threshold,
            'encoding_dim': self.encoding_dim
        }
        joblib.dump(state, filepath)
        print(f"  💾 LSTM Autoencoder guardado en {filepath} y {keras_path}")

    @classmethod
    def load(cls, filepath: str):
        """Carga un modelo guardado previamente."""
        state = joblib.load(filepath)
        keras_path = filepath.replace(".pkl", ".keras")
        
        instance = cls(encoding_dim=state['encoding_dim'])
        instance.scaler = state['scaler']
        instance.threshold = state['threshold']
        
        instance.model = keras.models.load_model(keras_path)
        
        return instance
