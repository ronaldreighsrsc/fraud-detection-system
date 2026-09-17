import pytest
import pandas as pd
import numpy as np
from src.graphs.graph_analyzer import TransactionGraphAnalyzer


@pytest.fixture
def sample_transactions_df():
    """Genera un DataFrame transaccional controlado para pruebas de grafos."""
    return pd.DataFrame([
        {'origin_account': 'ACC_01', 'destination_account': 'ACC_MULE_101', 'transaction_amount': 50000.0},
        {'origin_account': 'ACC_02', 'destination_account': 'ACC_MULE_101', 'transaction_amount': 75000.0},
        {'origin_account': 'ACC_03', 'destination_account': 'ACC_MULE_101', 'transaction_amount': 120000.0},
        {'origin_account': 'ACC_04', 'destination_account': 'ACC_MULE_101', 'transaction_amount': 45000.0},
        {'origin_account': 'ACC_05', 'destination_account': 'ACC_MULE_101', 'transaction_amount': 90000.0},
        {'origin_account': 'ACC_06', 'destination_account': 'ACC_MULE_101', 'transaction_amount': 60000.0},
        # La cuenta mula dispersa solo a 1 destino
        {'origin_account': 'ACC_MULE_101', 'destination_account': 'ACC_BOSS', 'transaction_amount': 400000.0},
        # Carrusel circular: A -> B -> C -> A
        {'origin_account': 'RING_A', 'destination_account': 'RING_B', 'transaction_amount': 10000.0},
        {'origin_account': 'RING_B', 'destination_account': 'RING_C', 'transaction_amount': 9500.0},
        {'origin_account': 'RING_C', 'destination_account': 'RING_A', 'transaction_amount': 9000.0},
    ])


def test_graph_builder_and_mule_detection(sample_transactions_df):
    analyzer = TransactionGraphAnalyzer()
    analyzer.build_graph_from_dataframe(sample_transactions_df)

    # 1. Verificar métricas de cuenta mula concentradora
    mule_metrics = analyzer.extract_graph_features('ACC_MULE_101')
    assert mule_metrics['in_degree'] == 6.0
    assert mule_metrics['out_degree'] == 1.0
    assert mule_metrics['is_mule_candidate'] == 1.0
    assert mule_metrics['pagerank'] > 0.0
    assert mule_metrics['total_in_amount'] == 440000.0
    assert mule_metrics['total_out_amount'] == 400000.0

    # 2. Verificar cuenta legítima estándar (solo envía)
    sender_metrics = analyzer.extract_graph_features('ACC_01')
    assert sender_metrics['in_degree'] == 0.0
    assert sender_metrics['out_degree'] == 1.0
    assert sender_metrics['is_mule_candidate'] == 0.0

    # 3. Verificar cuenta inexistente (valores por defecto seguros)
    unknown_metrics = analyzer.extract_graph_features('NON_EXISTENT_ACC')
    assert unknown_metrics['in_degree'] == 0.0
    assert unknown_metrics['out_degree'] == 0.0
    assert unknown_metrics['is_mule_candidate'] == 0.0


def test_fraud_ring_detection(sample_transactions_df):
    analyzer = TransactionGraphAnalyzer()
    analyzer.build_graph_from_dataframe(sample_transactions_df)

    rings = analyzer.detect_fraud_rings(min_cycle_length=3, max_cycle_length=5)
    assert len(rings) >= 1
    # Debe contener el ciclo RING_A -> RING_B -> RING_C -> RING_A
    ring_nodes = set(rings[0])
    assert 'RING_A' in ring_nodes
    assert 'RING_B' in ring_nodes
    assert 'RING_C' in ring_nodes


def test_top_mule_candidates_ranking(sample_transactions_df):
    analyzer = TransactionGraphAnalyzer()
    analyzer.build_graph_from_dataframe(sample_transactions_df)

    top_mules = analyzer.get_top_mule_candidates(top_n=5)
    assert len(top_mules) >= 1
    assert top_mules[0]['account_id'] == 'ACC_MULE_101'
    assert top_mules[0]['is_mule'] == 1
