# Usa una imagen oficial de Python ligera
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala las dependencias del sistema necesarias para compilar algunas librerías de ML
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia los requerimientos primero (Aprovecha el caché de capas de Docker)
COPY requirements.txt .

# Instala las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código fuente y los modelos pre-entrenados
# OJO: Gracias al .dockerignore, no copiaremos venv ni los datasets pesados
COPY src/ ./src/
COPY models/saved_models/ ./models/saved_models/
COPY fastapii.py .

# Expone el puerto 8000 para que podamos conectarnos desde fuera
EXPOSE 8000

# Comando para iniciar la API cuando el contenedor arranque
CMD ["uvicorn", "fastapii:app", "--host", "0.0.0.0", "--port", "8000"]
