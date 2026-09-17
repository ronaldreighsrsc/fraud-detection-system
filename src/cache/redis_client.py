"""
Cliente Redis con Fallback en Memoria (fakeredis) y Tolerancia a Fallos.
Proporciona búsquedas atómicas O(1) en < 2ms para métricas topológicas de grafos.
"""
import os
from typing import Dict, Any, Optional


class FastGraphCache:
    """
    Capa de Acceso a Métricas de Red en Memoria.
    Garantiza latencia ultra-baja en la ruta crítica transaccional.
    Si un clúster de Redis no está configurado o falla la conexión, conmuta a fakeredis.
    """

    def __init__(self, host: Optional[str] = None, port: int = 6379,
                 use_fake_explicit: Optional[bool] = None):
        self.host = host or os.getenv("REDIS_HOST", "localhost")
        self.port = int(os.getenv("REDIS_PORT", port))
        force_fake = os.getenv("USE_FAKE_REDIS", "false").lower() == "true"
        if use_fake_explicit is not None:
            force_fake = use_fake_explicit

        self.is_fake = False

        if force_fake:
            self._init_fakeredis()
        else:
            try:
                import redis
                self.r = redis.Redis(
                    host=self.host,
                    port=self.port,
                    decode_responses=True,
                    socket_timeout=0.05,
                    socket_connect_timeout=0.05
                )
                self.r.ping()
                self.is_fake = False
            except Exception:
                self._init_fakeredis()

    def _init_fakeredis(self):
        """Inicializa almacenamiento en memoria usando fakeredis."""
        try:
            import fakeredis
            self.r = fakeredis.FakeStrictRedis(decode_responses=True)
            self.is_fake = True
        except ImportError:
            # Fallback trivial a diccionario si fakeredis no estuviera
            class SimpleDictStore:
                def __init__(self):
                    self.store = {}
                def hgetall(self, k):
                    return self.store.get(k, {})
                def hset(self, k, mapping=None, **kwargs):
                    m = mapping or kwargs
                    if k not in self.store:
                        self.store[k] = {}
                    self.store[k].update(m)
                def expire(self, k, t):
                    pass
                def ping(self):
                    return True
                def flushall(self):
                    self.store.clear()
            self.r = SimpleDictStore()
            self.is_fake = True

    def set_graph_metrics(self, account_id: Any, metrics: Dict[str, Any], ttl: int = 86400) -> None:
        """Almacena o actualiza el vector de métricas de red para una cuenta."""
        key = f"account:{str(account_id)}:graph"
        mapping = {
            "in_degree": str(metrics.get("in_degree", 0.0)),
            "out_degree": str(metrics.get("out_degree", 0.0)),
            "pagerank": str(metrics.get("pagerank", 0.0)),
            "is_mule_candidate": str(metrics.get("is_mule_candidate", 0.0)),
            "total_in_amount": str(metrics.get("total_in_amount", 0.0)),
            "total_out_amount": str(metrics.get("total_out_amount", 0.0))
        }
        self.r.hset(key, mapping=mapping)
        self.r.expire(key, ttl)

    def get_graph_metrics(self, account_id: Any) -> Dict[str, float]:
        """
        Point lookup atómico < 2ms con valores por defecto ante Cache Miss.
        """
        key = f"account:{str(account_id)}:graph"
        data = self.r.hgetall(key)
        if not data:
            return {
                "in_degree": 0.0,
                "out_degree": 0.0,
                "pagerank": 0.0,
                "is_mule_candidate": 0.0,
                "total_in_amount": 0.0,
                "total_out_amount": 0.0
            }

        return {
            "in_degree": float(data.get("in_degree", 0.0)),
            "out_degree": float(data.get("out_degree", 0.0)),
            "pagerank": float(data.get("pagerank", 0.0)),
            "is_mule_candidate": float(data.get("is_mule_candidate", 0.0)),
            "total_in_amount": float(data.get("total_in_amount", 0.0)),
            "total_out_amount": float(data.get("total_out_amount", 0.0))
        }

    def ping(self) -> bool:
        """Verifica disponibilidad del caché."""
        try:
            return bool(self.r.ping())
        except Exception:
            return False

    def flush_all(self) -> None:
        """Limpia el almacenamiento (utilizado en suites de pruebas)."""
        try:
            self.r.flushall()
        except Exception:
            pass
