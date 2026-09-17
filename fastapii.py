import sys
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
import numpy as np
import uuid
import time

# Importar modelos v1
from src.models.xgb_detector import FraudXGBoostDetector
from src.models.isolation_forest import AnomalyIsolationForest
from src.models.autoencoder_deep import FraudDeepAutoencoder
from src.models.autoencoder_lstm import FraudLSTMAutoencoder
from src.models.gan_detector import FraudGANDetector

# Importar subsistemas v2 Enterprise
from src.cache.redis_client import FastGraphCache
from src.rules.rules_engine import HybridRulesEngine
from src.explainability.shap_explainer import ModelExplainer
from src.compliance.uaf_ros_agent import ComplianceROSAgent

# Diccionario global para guardar los componentes en RAM
ml_models = {}
v2_components = {}
incident_store: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Inicialización de modelos y componentes de alta disponibilidad en RAM
    antes de aceptar tráfico transaccional.
    """
    print("⏳ [Serving Init] Cargando modelos de IA y motores v2 en memoria...")
    
    # 1. Cargar modelos predictivos
    try:
        ml_models['xgb'] = FraudXGBoostDetector.load("./models/saved_models/xgb.pkl")
        ml_models['iso'] = AnomalyIsolationForest.load("./models/saved_models/iso.pkl")
        ml_models['ae_deep'] = FraudDeepAutoencoder.load("./models/saved_models/ae_deep.pkl")
        ml_models['ae_lstm'] = FraudLSTMAutoencoder.load("./models/saved_models/ae_lstm.pkl")
        ml_models['gan'] = FraudGANDetector.load("./models/saved_models/gan.pkl")
        print("  ✅ [ML Models] 5 Modelos de IA cargados exitosamente.")
    except Exception as e:
        print(f"  ⚠️ [ML Models] Advertencia al cargar modelos: {e}")

    # 2. Inicializar componentes v2 Enterprise
    try:
        # Caché Redis / Fakeredis
        v2_components['cache'] = FastGraphCache()
        
        # Motor de Reglas CMF
        rules = HybridRulesEngine()
        # Sembrar cuentas conocidas de prueba
        rules.add_blacklisted_account("999999")
        rules.add_blacklisted_account("ACC_BLOCKED_01")
        v2_components['rules'] = rules

        # Explicador SHAP
        if 'xgb' in ml_models:
            v2_components['explainer'] = ModelExplainer(ml_models['xgb'])
        else:
            v2_components['explainer'] = ModelExplainer(None)

        # Agente ROS UAF
        v2_components['ros_agent'] = ComplianceROSAgent(bank_name="Banco Bci")

        print("  ✅ [v2 Enterprise] Redis Cache, Rules Engine, TreeSHAP y ROS Agent listos.")
    except Exception as e:
        print(f"  ⚠️ [v2 Enterprise] Error inicializando componentes v2: {e}")

    yield

    # Limpieza al apagar
    ml_models.clear()
    v2_components.clear()
    incident_store.clear()
    print("🛑 Servidor apagado y memoria liberada.")


# Instancia principal de FastAPI
app = FastAPI(
    title="Bci Autonomous Risk, AML Graph Network & Hybrid Fraud Defense System",
    description="Plataforma de Prevención de Fraudes, Lavado de Activos (PLAFT) y Gobernanza CMF v2.0",
    version="2.0.0",
    lifespan=lifespan
)


# ==============================================================================
# SCHEMAS DE ENTRADA / SALIDA
# ==============================================================================

class Transaction(BaseModel):
    """Esquema v1 para compatibilidad histórica."""
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


class TransactionV2Payload(BaseModel):
    """
    Payload transaccional de grado bancario para la v2.0.
    Permite tanto el envío de variables directas como de variables derivadas automáticas.
    """
    tx_id: Optional[str] = Field(default=None, description="Identificador único transaccional")
    origin_account: str = Field(..., description="Cuenta origen / RUT cliente")
    destination_account: str = Field(..., description="Cuenta receptora de fondos")
    destination_country: Optional[str] = Field(default="CHL", description="Código ISO país destino")
    account_age_days: Optional[float] = Field(default=365.0, description="Antigüedad de la cuenta en días")
    transaction_amount: float = Field(..., description="Monto en CLP de la operación")
    hour_of_day: float = Field(..., description="Hora de la transacción (0-23)")
    day_of_week: float = Field(default=2.0, description="Día de la semana (0-6)")
    merchant_category: float = Field(default=1.0, description="Categoría de comercio (MCC)")
    distance_from_home: float = Field(default=5.0, description="Distancia geolocalizada en km")
    is_international: float = Field(default=0.0, description="Indicador de transacción internacional (0 o 1)")
    
    # Variables analíticas opcionales (se auto-calculan si no se proveen)
    amount_deviation: Optional[float] = None
    amount_zscore: Optional[float] = None
    amount_to_median_ratio: Optional[float] = None
    tx_frequency_1h: Optional[float] = 1.0
    tx_frequency_24h: Optional[float] = 3.0

    def to_feature_vector(self) -> np.ndarray:
        """Construye el vector canónico de 18 features exigido por el modelo XGBoost."""
        amt = float(self.transaction_amount)
        h = float(self.hour_of_day)
        d = float(self.day_of_week)
        mcc = float(self.merchant_category)
        dist = float(self.distance_from_home)
        intl = float(self.is_international)

        dev = self.amount_deviation if self.amount_deviation is not None else (amt - 50_000.0)
        zscore = self.amount_zscore if self.amount_zscore is not None else (dev / 30_000.0)
        median_ratio = self.amount_to_median_ratio if self.amount_to_median_ratio is not None else (amt / 45_000.0)
        freq_1h = float(self.tx_frequency_1h or 1.0)
        freq_24h = float(self.tx_frequency_24h or 3.0)

        # Encodings cíclicos
        h_sin = float(np.sin(2 * np.pi * h / 24.0))
        h_cos = float(np.cos(2 * np.pi * h / 24.0))
        d_sin = float(np.sin(2 * np.pi * d / 7.0))
        d_cos = float(np.cos(2 * np.pi * d / 7.0))

        # Interacciones
        amt_x_dist = amt * dist
        amt_x_intl = amt * intl
        zscore_x_freq = zscore * freq_1h

        return np.array([[
            amt, h, d, mcc, dist, intl,
            dev, zscore, median_ratio,
            freq_1h, freq_24h,
            h_sin, h_cos, d_sin, d_cos,
            amt_x_dist, amt_x_intl, zscore_x_freq
        ]], dtype=float)


# ==============================================================================
# FUNCIONES AUXILIARES ASÍNCRONAS
# ==============================================================================

def log_fraud_alert_to_database(tx_id: str, payload_dict: Dict[str, Any], decision: Dict[str, Any],
                                top_shap: List[Dict[str, Any]], graph_features: Dict[str, Any]):
    """
    Registra el incidente en el almacén de auditoría para consumo posterior por Oficiales UAF.
    Se ejecuta como BackgroundTask fuera del SLA síncrono del cliente.
    """
    incident_store[tx_id] = {
        "timestamp": time.time(),
        "tx_details": payload_dict,
        "decision": decision,
        "top_shap": top_shap,
        "graph_insights": graph_features
    }
    print(f"📦 [Audit Trail] Incidente {tx_id} indexado exitosamente para análisis forense.")


# ==============================================================================
# ENDPOINTS v2.0 (ENTERPRISE DUAL SLA)
# ==============================================================================

@app.post("/api/v2/evaluate_transaction")
async def evaluate_transaction(tx: TransactionV2Payload, background_tasks: BackgroundTasks):
    """
    RUTA CRÍTICA TRANSACCIONAL (SLA < 30 ms).
    Ejecuta el pipeline multicapa de decisión bancaria:
    1. Reglas Duras CMF (0.5 ms)
    2. Point Lookup Redis para Topología de Grafos (< 2 ms)
    3. Scoring XGBoost en memoria (< 5 ms)
    4. Matriz de Decisión Híbrida Multicriterio
    5. Explicabilidad Local TreeSHAP (< 15 ms)
    """
    start_time = time.perf_counter()
    tx_id = tx.tx_id or f"TX-{uuid.uuid4().hex[:10].upper()}"

    # Instancias de componentes
    rules_engine: HybridRulesEngine = v2_components.get('rules') or HybridRulesEngine()
    cache: FastGraphCache = v2_components.get('cache') or FastGraphCache()
    explainer: ModelExplainer = v2_components.get('explainer')
    xgb_model = ml_models.get('xgb')

    tx_dict = tx.model_dump()
    tx_dict['tx_id'] = tx_id

    # Paso 1: Reglas Duras CMF
    hard_triggered, hard_reason = rules_engine.evaluate_hard_rules(tx_dict)

    # Paso 2: Point Lookup Redis (< 2 ms)
    graph_features = cache.get_graph_metrics(tx.origin_account)
    # Si la cuenta destino es una mula conocida, también extraemos
    dest_graph = cache.get_graph_metrics(tx.destination_account)
    if dest_graph.get("is_mule_candidate", 0.0) == 1.0:
        graph_features["is_mule_candidate"] = 1.0

    # Paso 3: Scoring Predictivo XGBoost en Memoria
    feature_vec = tx.to_feature_vector()
    if xgb_model is not None and not hard_triggered:
        _, probs = xgb_model.predict(feature_vec)
        ml_score = float(probs[0])
    else:
        ml_score = 1.0 if hard_triggered else 0.05

    # Paso 4: Matriz de Decisión Híbrida
    decision = rules_engine.combine_verdict(
        ml_score=ml_score,
        graph_features=graph_features,
        hard_rule_triggered=hard_triggered,
        rule_reason=hard_reason
    )

    # Paso 5: Explicabilidad SHAP auditable (Ley 21.234)
    if explainer is not None:
        top_shap_reasons = explainer.explain_transaction(feature_vec, top_k=3)
    else:
        top_shap_reasons = []

    # Desacoplamiento de SLA: Si se bloquea, encolar tarea asíncrona de compliance
    if decision["action"] in ["BLOCK_PREVENTIVE", "BLOCK_IMMEDIATE"]:
        background_tasks.add_task(
            log_fraud_alert_to_database,
            tx_id=tx_id,
            payload_dict=tx_dict,
            decision=decision,
            top_shap=top_shap_reasons,
            graph_features=graph_features
        )

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "tx_id": tx_id,
        "action": decision["action"],
        "risk_level": decision["risk_level"],
        "final_risk_score": decision["final_risk_score"],
        "requires_mfa": decision["requires_mfa"],
        "notify_compliance": decision["notify_compliance"],
        "reason": decision["reason"],
        "graph_topology": {
            "origin_in_degree": graph_features.get("in_degree", 0.0),
            "origin_out_degree": graph_features.get("out_degree", 0.0),
            "is_mule_detected": bool(graph_features.get("is_mule_candidate", 0.0))
        },
        "top_shap_reasons": top_shap_reasons,
        "latency_ms": f"{latency_ms} ms"
    }


@app.post("/api/v2/compliance/generate_ros/{tx_id}")
async def generate_uaf_ros_report(tx_id: str):
    """
    RUTA ASÍNCRONA / BACK-OFFICE (Bajo demanda por Analista de Cumplimiento UAF).
    Ejecuta el RAG sobre Tipologías Oficiales UAF sin afectar el SLA transaccional vivo.
    """
    ros_agent: ComplianceROSAgent = v2_components.get('ros_agent') or ComplianceROSAgent()

    # Buscar el incidente en el registro
    alert_data = incident_store.get(tx_id)

    if not alert_data:
        # Si no está en el almacén de incidentes, generamos una plantilla para el tx_id provisto
        tx_details = {
            "tx_id": tx_id,
            "transaction_amount": 7_500_000,
            "origin_account": "ACC_SUSPECT_01",
            "destination_account": "ACC_MULE_101",
            "risk_score": 0.92
        }
        top_shap = [
            {"feature": "transaction_amount", "feature_label": "Monto de la transacción", "shap_impact": 0.42, "actual_value": 7500000.0},
            {"feature": "tx_frequency_1h", "feature_label": "Frecuencia de transferencias (última 1 hora)", "shap_impact": 0.31, "actual_value": 6.0},
            {"feature": "distance_from_home", "feature_label": "Distancia al domicilio habitual", "shap_impact": 0.22, "actual_value": 350.0}
        ]
        graph_insights = {
            "in_degree": 8.0,
            "out_degree": 1.0,
            "pagerank": 0.0142,
            "is_mule_candidate": 1.0
        }
    else:
        tx_details = alert_data["tx_details"]
        top_shap = alert_data["top_shap"]
        graph_insights = alert_data["graph_insights"]

    # Generar informe formal
    ros_document = ros_agent.generate_ros_narrative(
        tx_details=tx_details,
        top_risk_factors=top_shap,
        graph_insights=graph_insights
    )

    return {
        "tx_id": tx_id,
        "status": "GENERADO_EXITOSAMENTE",
        "normativa": "Ley N° 19.913 (UAF Chile) & CMF Capítulo 20-10",
        "ros_formal_report": ros_document
    }


@app.get("/api/v2/info")
def get_system_v2_info():
    """Retorna los metadatos y estado del sistema v2.0 Enterprise."""
    cache = v2_components.get('cache')
    rules = v2_components.get('rules')
    
    return {
        "system_name": "Bci Enterprise Autonomous Risk & Fraud Prevention System",
        "version": "2.0.0",
        "status": "OPERATIONAL",
        "sla_target": "< 30 ms",
        "components": {
            "machine_learning": list(ml_models.keys()),
            "redis_cache": "CONNECTED (FAKEREDIS/RAM)" if (cache and cache.is_fake) else "CONNECTED (REDIS CLUSTER)",
            "rules_engine": "ACTIVE (CMF CAP 20-10)",
            "shap_explainer": "ACTIVE (LEY 21.234)",
            "ros_agent": "ACTIVE (UAF RAG / DETERMINISTIC FALLBACK)"
        },
        "regulatory_compliance": [
            "CMF Capítulo 20-10 (Gestión de Riesgo Operacional)",
            "Ley 21.234 (Fraude en Medios de Pago y Devolución 35 UF)",
            "Ley 19.913 (Lavado de Activos y Reportes ROS UAF)"
        ]
    }


# ==============================================================================
# ENDPOINTS v1 (HISTÓRICOS RETROCOMPATIBLES)
# ==============================================================================

@app.get("/")
def read_root():
    return {
        "mensaje": "¡API de Detección de Fraudes y Riesgo Financiero Operativa! 🚀",
        "version": "2.0.0 Enterprise",
        "docs_url": "/docs"
    }


@app.get("/health")
def health_check():
    loaded = list(ml_models.keys())
    return {
        "status": "ok",
        "modelos_cargados": loaded,
        "v2_enterprise_modules": list(v2_components.keys())
    }


@app.post("/predict")
def predict_fraud(transaction: Transaction):
    """
    Endpoint v1 preservado para retrocompatibilidad total con clientes anteriores.
    """
    features = np.array([[
        transaction.transaction_amount, transaction.hour_of_day, transaction.day_of_week,
        transaction.merchant_category, transaction.distance_from_home, transaction.is_international,
        transaction.amount_deviation, transaction.amount_zscore, transaction.amount_to_median_ratio,
        transaction.tx_frequency_1h, transaction.tx_frequency_24h,
        transaction.hour_sin, transaction.hour_cos, transaction.day_sin, transaction.day_cos,
        transaction.amount_x_distance, transaction.amount_x_international, transaction.zscore_x_frequency
    ]])

    resultados = {}

    if 'xgb' in ml_models:
        pred, prob = ml_models['xgb'].predict(features)
        resultados['xgboost'] = {"is_fraud": bool(pred[0]), "probability": float(prob[0])}

    if 'iso' in ml_models:
        pred, score = ml_models['iso'].predict(features)
        resultados['isolation_forest'] = {"is_fraud": bool(pred[0]), "anomaly_score": float(score[0])}

    if 'ae_deep' in ml_models:
        pred, score = ml_models['ae_deep'].predict(features)
        resultados['deep_autoencoder'] = {"is_fraud": bool(pred[0]), "reconstruction_mse": float(score[0])}

    if 'ae_lstm' in ml_models:
        pred, score = ml_models['ae_lstm'].predict(features)
        resultados['lstm_autoencoder'] = {"is_fraud": bool(pred[0]), "reconstruction_mse": float(score[0])}

    if 'gan' in ml_models:
        pred, score = ml_models['gan'].predict(features)
        resultados['gan'] = {"is_fraud": bool(pred[0]), "anomaly_score": float(score[0])}

    return {
        "status": "Transacción analizada",
        "predictions": resultados
    }
