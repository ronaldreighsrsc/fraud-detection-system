import pytest
from src.cache.redis_client import FastGraphCache


def test_fast_graph_cache_in_memory():
    # Forzar uso de fakeredis para asegurar test hermético y reproducible
    cache = FastGraphCache(use_fake_explicit=True)
    assert cache.is_fake is True
    assert cache.ping() is True

    # 1. Recuperar cuenta inexistente -> valores neutros por defecto
    defaults = cache.get_graph_metrics("ACC_UNKNOWN")
    assert defaults["in_degree"] == 0.0
    assert defaults["out_degree"] == 0.0
    assert defaults["is_mule_candidate"] == 0.0

    # 2. Guardar métricas y consultar
    metrics_to_set = {
        "in_degree": 7.0,
        "out_degree": 1.0,
        "pagerank": 0.0245,
        "is_mule_candidate": 1.0,
        "total_in_amount": 1500000.0,
        "total_out_amount": 200000.0
    }
    cache.set_graph_metrics("ACC_TEST_01", metrics_to_set)

    retrieved = cache.get_graph_metrics("ACC_TEST_01")
    assert retrieved["in_degree"] == 7.0
    assert retrieved["out_degree"] == 1.0
    assert retrieved["pagerank"] == 0.0245
    assert retrieved["is_mule_candidate"] == 1.0
    assert retrieved["total_in_amount"] == 1500000.0

    # 3. Limpieza
    cache.flush_all()
    assert cache.get_graph_metrics("ACC_TEST_01")["in_degree"] == 0.0
