import pytest
import numpy as np
from src.explainability.shap_explainer import ModelExplainer
from src.models.xgb_detector import FraudXGBoostDetector


def test_shap_explainer_with_saved_model():
    # Intentar cargar el modelo XGBoost preentrenado si existe
    try:
        detector = FraudXGBoostDetector.load("./models/saved_models/xgb.pkl")
    except Exception:
        detector = None

    explainer = ModelExplainer(detector)

    # Vector de prueba de 18 features (Transacción anómala)
    tx_vector = np.array([
        7_500_000.0,  # transaction_amount muy alto
        3.0,          # 3 AM
        2.0,          # Martes
        1.0,          # MCC
        150.0,        # Distancia alta
        1.0,          # Internacional
        7_450_000.0,  # Desviación
        12.5,         # Z-score alto
        15.0,         # Median ratio
        6.0,          # 6 transacciones en 1h
        15.0,         # 24h
        -0.7, 0.7, 0.4, -0.9,
        1_125_000_000.0, 7_500_000.0, 75.0
    ])

    reasons = explainer.explain_transaction(tx_vector, top_k=3)
    assert len(reasons) == 3

    for item in reasons:
        assert 'feature' in item
        assert 'feature_label' in item
        assert 'shap_impact' in item
        assert 'actual_value' in item
        assert 'direction' in item


def test_shap_heuristic_fallback():
    # Test sin modelo (debe activar fallback sin lanzar excepciones)
    explainer = ModelExplainer(None)
    tx_vector = np.zeros(18)
    tx_vector[0] = 5_000_000.0  # Monto alto
    tx_vector[4] = 80.0         # Distancia alta

    reasons = explainer.explain_transaction(tx_vector, top_k=3)
    assert len(reasons) == 3
    assert reasons[0]['shap_impact'] > 0
