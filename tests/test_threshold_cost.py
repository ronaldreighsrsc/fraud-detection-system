import pytest
import numpy as np
from src.evaluation.threshold_analyzer import ThresholdAnalyzer


def test_optimal_cost_threshold_asymmetry():
    analyzer = ThresholdAnalyzer()

    # Generar probabilidades sintéticas y etiquetas desbalanceadas
    np.random.seed(42)
    n_samples = 1000
    y_true = np.zeros(n_samples, dtype=int)
    # 2% de fraude
    fraud_indices = np.random.choice(n_samples, size=20, replace=False)
    y_true[fraud_indices] = 1

    # Probabilidades predichas correlacionadas con fraude
    y_probs = np.random.beta(a=0.5, b=5.0, size=n_samples)
    y_probs[fraud_indices] = np.random.uniform(0.30, 0.95, size=20)

    # Calibrar bajo matriz asimétrica 40:1 Ley 21.234 ($1M FN vs $25K FP)
    res = analyzer.find_optimal_cost_threshold(y_true, y_probs, c_fn=1_000_000, c_fp=25_000)

    assert 'optimal_threshold' in res
    assert 'min_expected_loss' in res
    assert 'net_savings' in res
    assert res['cost_ratio_fn_fp'] == 40.0

    # Dado el alto costo de dejar pasar un fraude (FN), el umbral óptimo debe ser más conservador que 0.50
    assert res['optimal_threshold'] < 0.50
    # El ahorro económico frente al corte default de 0.50 debe ser positivo
    assert res['net_savings'] >= 0.0
