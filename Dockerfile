FROM python:3.12-slim

# Dependencias de compilación
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY titanic_ml_pipeline/ ./titanic_ml_pipeline/
COPY pyproject.toml setup.py README.md ./

# Dependencias
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.3 \
    scipy==1.16.1 \
    scikit-learn==1.7.1 \
    seaborn==0.13.2 \
    joblib==1.5.1

# Directorio de trabajo y archivos de salida
ENV PYTHONUNBUFFERED=1

# Ejecutar el módulo principal
CMD ["python", "-m", "titanic_ml_pipeline.main"]