from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import numpy as np

# Importar modelos
from src.models.xgb_detector import FraudXGBoostDetector
from src.models.isolation_forest import AnomalyIsolationForest
from src.models.autoencoder_deep import FraudDeepAutoencoder
from src.models.autoencoder_lstm import FraudLSTMAutoencoder
from src.models.gan_detector import FraudGANDetector

# Diccionario global para guardar los modelos en RAM
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Esto se ejecuta ANTES de que el servidor empiece a recibir peticiones
    print("⏳ Cargando modelos en memoria...")
    try:
        ml_models['xgb'] = FraudXGBoostDetector.load("./models/saved_models/xgb.pkl")
        ml_models['iso'] = AnomalyIsolationForest.load("./models/saved_models/iso.pkl")
        ml_models['ae_deep'] = FraudDeepAutoencoder.load("./models/saved_models/ae_deep.pkl")
        ml_models['ae_lstm'] = FraudLSTMAutoencoder.load("./models/saved_models/ae_lstm.pkl")
        ml_models['gan'] = FraudGANDetector.load("./models/saved_models/gan.pkl")
        print("✅ ¡Todos los modelos cargados exitosamente!")
    except Exception as e:
        print(f"⚠️ Error cargando modelos (¿Ya corriste main_training.py?): {e}")
    
    yield  # Aquí el servidor se queda encendido funcionando
    
    # Esto se ejecuta cuando apagas el servidor (Limpieza)
    ml_models.clear()

# 1. Crear la instancia de la aplicación
app = FastAPI(
    title="Fraud Detection API",
    description="API con los 5 modelos de IA integrados",
    version="2.0.0",
    lifespan=lifespan
)

# 2. Esquema de Datos (18 features)
class Transaction(BaseModel):
    transaction_amount: float
    hour_of_day: float
    day_of_week: float
    merchant_category: float
    distance_from_home: float
    is_international: float
    amount_deviation: float
    amount_zscore: float
    amount_to_median_ratio: float
    tx_frequency_1h: float
    tx_frequency_24h: float
    hour_sin: float
    hour_cos: float
    day_sin: float
    day_cos: float
    amount_x_distance: float
    amount_x_international: float
    zscore_x_frequency: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_amount": 150.5, "hour_of_day": 14, "day_of_week": 3,
                "merchant_category": 2, "distance_from_home": 5.2, "is_international": 0,
                "amount_deviation": 10.0, "amount_zscore": 0.5, "amount_to_median_ratio": 1.1,
                "tx_frequency_1h": 2, "tx_frequency_24h": 5, "hour_sin": -0.5,
                "hour_cos": -0.866, "day_sin": 0.433, "day_cos": -0.9,
                "amount_x_distance": 782.6, "amount_x_international": 0.0, "zscore_x_frequency": 1.0
            }
        }

@app.get("/")
def read_root():
    return {"mensaje": "¡API de Detección de Fraudes Operativa! 🚀"}

# 3. Definir una ruta de prueba de salud (health check)
@app.get("/health")
def health_check():
    loaded = list(ml_models.keys())
    return {"status": "ok", "modelos_cargados": loaded}

# --- NUEVO: Ruta de Inferencia (POST) ---
# Usamos POST porque el usuario nos está "enviando" una Transacción para analizar
@app.post("/predict")
def predict_fraud(transaction: Transaction):
    # Convertir el JSON recibido a un array de Numpy de 1 fila y 18 columnas
    # El orden debe coincidir exactamente con el orden del entrenamiento
    features = np.array([[
        transaction.transaction_amount, transaction.hour_of_day, transaction.day_of_week,
        transaction.merchant_category, transaction.distance_from_home, transaction.is_international,
        transaction.amount_deviation, transaction.amount_zscore, transaction.amount_to_median_ratio,
        transaction.tx_frequency_1h, transaction.tx_frequency_24h,
        transaction.hour_sin, transaction.hour_cos, transaction.day_sin, transaction.day_cos,
        transaction.amount_x_distance, transaction.amount_x_international, transaction.zscore_x_frequency
    ]])

    resultados = {}
    
    # Inferencia XGBoost
    if 'xgb' in ml_models:
        pred, prob = ml_models['xgb'].predict(features)
        resultados['xgboost'] = {"is_fraud": bool(pred[0]), "probability": float(prob[0])}
        
    # Inferencia Isolation Forest
    if 'iso' in ml_models:
        pred, score = ml_models['iso'].predict(features)
        resultados['isolation_forest'] = {"is_fraud": bool(pred[0]), "anomaly_score": float(score[0])}
        
    # Inferencia Deep Autoencoder
    if 'ae_deep' in ml_models:
        pred, score = ml_models['ae_deep'].predict(features)
        resultados['deep_autoencoder'] = {"is_fraud": bool(pred[0]), "reconstruction_mse": float(score[0])}

    # Inferencia LSTM Autoencoder
    if 'ae_lstm' in ml_models:
        pred, score = ml_models['ae_lstm'].predict(features)
        resultados['lstm_autoencoder'] = {"is_fraud": bool(pred[0]), "reconstruction_mse": float(score[0])}
        
    # Inferencia GAN
    if 'gan' in ml_models:
        pred, score = ml_models['gan'].predict(features)
        resultados['gan'] = {"is_fraud": bool(pred[0]), "anomaly_score": float(score[0])}

    return {
        "status": "Transacción analizada",
        "predictions": resultados
    }
