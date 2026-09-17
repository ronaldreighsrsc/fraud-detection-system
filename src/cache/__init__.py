"""
Módulo de Caché en Memoria (Redis In-Memory Layer con Fallback Fakeredis).
"""
from .redis_client import FastGraphCache

__all__ = ["FastGraphCache"]
