FROM python:3.11-slim AS app

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app:/app/app

WORKDIR /app

# System dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (camada cacheada — só reinstala se requirements.txt mudar)
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/vdb \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["uvicorn",\
    "app.app:app",\
    "--host", "0.0.0.0",\
    "--port", "8000",\
    "--proxy-headers",\
    "--log-level", "info"]

# ------------------------------------------------------------------
# Build target: mlflow-server
# Uso no docker-compose: build: context: . target: mlflow-server
# ------------------------------------------------------------------
FROM python:3.11-slim AS mlflow-server

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /mlflow

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir mlflow==3.10.1 sqlalchemy

VOLUME /mlflow
EXPOSE 5000
