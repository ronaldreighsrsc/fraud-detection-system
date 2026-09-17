# Multi-stage ultra-lean para Producción FastAPI (< 250 MB)
# Bci Autonomous Risk & Fraud Prevention API v2.0
FROM python:3.12-slim AS runtime

WORKDIR /app

# Crear usuario de sistema no-root para seguridad bancaria
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Instalar dependencias estrictas de serving (sin TensorFlow pesado en el contenedor de inferencia rápida)
COPY requirements/base.txt requirements/serving.txt ./requirements/
RUN pip install --no-cache-dir -r requirements/serving.txt

# Copiar artefactos ligeros de inferencia y código fuente
COPY src/ ./src/
COPY models/saved_models/ ./models/saved_models/
COPY fastapii.py .

USER appuser
EXPOSE 8000

# Workers optimizados para serving de baja latencia (< 30 ms)
CMD ["uvicorn", "fastapii:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
