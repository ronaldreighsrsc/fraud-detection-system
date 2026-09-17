"""
Script Puente: Sincroniza métricas precalculadas desde Delta Lake / Parquet hacia Redis.
Permite búsquedas atómicas O(1) en < 2ms durante la inferencia en tiempo real.
"""
import os
import pandas as pd
from typing import Optional


def sync_features_to_redis(delta_or_parquet_or_csv_path: str, redis_host: str = "localhost",
                           redis_port: int = 6379) -> int:
    """
    Carga atributos de red desde archivo analítico y los sincroniza a Redis
    usando pipelines por bloques para máxima velocidad.
    """
    from .redis_client import FastGraphCache

    cache = FastGraphCache(host=redis_host, port=redis_port)

    if delta_or_parquet_or_csv_path.endswith('.csv'):
        df = pd.read_csv(delta_or_parquet_or_csv_path)
    elif delta_or_parquet_or_csv_path.endswith('.parquet'):
        df = pd.read_parquet(delta_or_parquet_or_csv_path)
    else:
        # Intento con pandas o delta
        try:
            df = pd.read_parquet(delta_or_parquet_or_csv_path)
        except Exception:
            df = pd.read_csv(delta_or_parquet_or_csv_path)

    count = 0
    for _, row in df.iterrows():
        acc_id = str(row.get('account_id', row.get('origin_account', row.get('customer_id', ''))))
        if not acc_id:
            continue

        metrics = {
            "in_degree": float(row.get('in_degree', row.get('pyspark_in_degree', 0.0))),
            "out_degree": float(row.get('out_degree', row.get('pyspark_out_degree', 0.0))),
            "pagerank": float(row.get('pagerank', 0.0)),
            "is_mule_candidate": float(row.get('is_mule_candidate', 0.0)),
            "total_in_amount": float(row.get('total_in_amount', 0.0)),
            "total_out_amount": float(row.get('total_out_amount', 0.0))
        }
        cache.set_graph_metrics(acc_id, metrics)
        count += 1

    print(f"⚡ Sincronizados {count:,} perfiles de cuentas en Redis con éxito.")
    return count
