FROM python:3.11-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app:/app/app

WORKDIR /app

# Install system build deps required for some Python packages and curl for healthchecks
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer when requirements.txt unchanged)
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user and directory for vectorstore, set permissions
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/vdb \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Run with uvicorn; proxy headers and reduced log verbosity for production
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--log-level", "info"]