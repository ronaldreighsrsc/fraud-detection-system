"""
Módulo de Análisis de Redes Complejas y Teoría de Grafos para PLAFT.
Identificación de Cuentas Mula y Carruseles de Fraude (Bci Enterprise Edition v2.0).
"""
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class TransactionGraphAnalyzer:
    """
    Analizador de topología y redes de transferencias electrónicas.
    Modela transacciones como un dígrafo ponderado G = (V, E),
    permitiendo la detección de cuentas concentradoras (mulas) y anillos de fraude.
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self._pagerank_cache: Optional[Dict[str, float]] = None

    def build_graph_from_dataframe(self, df: pd.DataFrame) -> None:
        """
        Construye el grafo dirigido a partir de un DataFrame de transacciones.
        Requiere columnas: 'origin_account' (o 'customer_id'), 'destination_account', 'transaction_amount'.
        """
        self.graph.clear()
        self._pagerank_cache = None

        orig_col = 'origin_account' if 'origin_account' in df.columns else 'customer_id'
        dest_col = 'destination_account' if 'destination_account' in df.columns else 'merchant_category'

        for _, row in df.iterrows():
            u = str(row[orig_col])
            v = str(row[dest_col])
            amount = float(row['transaction_amount'])

            if self.graph.has_edge(u, v):
                self.graph[u][v]['weight'] += amount
                self.graph[u][v]['tx_count'] += 1
            else:
                self.graph.add_edge(u, v, weight=amount, tx_count=1)

    def compute_pagerank(self, alpha: float = 0.85, max_iter: int = 100) -> Dict[str, float]:
        """Calcula o recupera el PageRank de todos los nodos en el grafo."""
        if self._pagerank_cache is None:
            if len(self.graph) == 0:
                self._pagerank_cache = {}
            else:
                try:
                    self._pagerank_cache = nx.pagerank(self.graph, alpha=alpha, max_iter=max_iter)
                except Exception:
                    # En caso de no convergencia o grafos singulares
                    self._pagerank_cache = {node: 1.0 / len(self.graph) for node in self.graph.nodes()}
        return self._pagerank_cache

    def extract_graph_features(self, account_id: Any) -> Dict[str, float]:
        """
        Extrae métricas de centralidad y topología para una cuenta específica.
        Si la cuenta no tiene historial en el grafo, retorna valores neutros por defecto.
        """
        acc = str(account_id)
        if not self.graph.has_node(acc):
            return {
                "in_degree": 0.0,
                "out_degree": 0.0,
                "pagerank": 0.0,
                "is_mule_candidate": 0.0,
                "total_in_amount": 0.0,
                "total_out_amount": 0.0
            }

        in_deg = float(self.graph.in_degree(acc))
        out_deg = float(self.graph.out_degree(acc))

        pr_dict = self.compute_pagerank()
        pagerank = float(pr_dict.get(acc, 0.0))

        # Suma de montos entrantes y salientes
        in_amount = sum(data.get('weight', 0.0) for _, _, data in self.graph.in_edges(acc, data=True))
        out_amount = sum(data.get('weight', 0.0) for _, _, data in self.graph.out_edges(acc, data=True))

        # Criterio de Cuenta Mula (PLAFT):
        # Concentra fondos de múltiples orígenes (in_degree >= 5) con baja dispersión inicial (out_degree <= 2)
        is_mule = 1.0 if (in_deg >= 5 and out_deg <= 2) else 0.0

        return {
            "in_degree": in_deg,
            "out_degree": out_deg,
            "pagerank": round(pagerank, 6),
            "is_mule_candidate": is_mule,
            "total_in_amount": round(in_amount, 2),
            "total_out_amount": round(out_amount, 2)
        }

    def detect_fraud_rings(self, min_cycle_length: int = 3, max_cycle_length: int = 5) -> List[List[str]]:
        """
        Detecta carruseles de transferencias (ciclos dirigidos cerrados donde los fondos recirculan).
        Patrón característico de lavado de activos para simular liquidez legítima.
        """
        if len(self.graph) == 0:
            return []

        cycles = []
        try:
            for cycle in nx.simple_cycles(self.graph):
                if min_cycle_length <= len(cycle) <= max_cycle_length:
                    cycles.append(cycle)
                if len(cycles) >= 50:  # Limitar para evitar explosión combinatoria
                    break
        except Exception:
            pass

        return cycles

    def get_top_mule_candidates(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Retorna las cuentas con mayor probabilidad de operar como cuentas mula."""
        candidates = []
        for node in self.graph.nodes():
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            if in_deg >= 3:
                candidates.append({
                    "account_id": node,
                    "in_degree": in_deg,
                    "out_degree": out_deg,
                    "ratio_in_out": in_deg / max(1, out_deg),
                    "is_mule": 1 if (in_deg >= 5 and out_deg <= 2) else 0
                })

        candidates.sort(key=lambda x: (x["is_mule"], x["in_degree"]), reverse=True)
        return candidates[:top_n]
