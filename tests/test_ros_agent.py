import pytest
from src.compliance.uaf_ros_agent import ComplianceROSAgent


def test_ros_agent_deterministic_generation():
    agent = ComplianceROSAgent(bank_name="Banco Bci")

    tx_details = {
        "tx_id": "TX-BCI-2026-9901",
        "transaction_amount": 12_500_000.0,
        "origin_account": "ACC_SUSPECT_77",
        "destination_account": "ACC_MULE_101",
        "risk_score": 0.94
    }

    top_risk_factors = [
        {"feature": "transaction_amount", "feature_label": "Monto de la transacción", "shap_impact": 0.45, "actual_value": 12500000.0},
        {"feature": "tx_frequency_1h", "feature_label": "Frecuencia de transferencias", "shap_impact": 0.32, "actual_value": 7.0}
    ]

    graph_insights = {
        "in_degree": 9.0,
        "out_degree": 1.0,
        "pagerank": 0.0315,
        "is_mule_candidate": 1.0
    }

    report = agent.generate_ros_narrative(tx_details, top_risk_factors, graph_insights)

    # Validar formato y elementos jurídicos requeridos
    assert "CONFIDENCIAL - PRE-INFORME DE OPERACIÓN SOSPECHOSA (ROS)" in report
    assert "Unidad de Análisis Financiero (UAF)" in report
    assert "Banco Bci" in report
    assert "TX-BCI-2026-9901" in report
    assert "Tipología UAF N° 3" in report
    assert "POSITIVO (ALTO RIESGO CONCENTRADOR)" in report
    assert "Capítulo 20-10" in report
    assert "Ley N° 19.913" in report
