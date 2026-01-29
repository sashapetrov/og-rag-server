FROM python:3.11-slim

# OCI Labels (https://github.com/opencontainers/image-spec/blob/main/annotations.md)
LABEL org.opencontainers.image.title="OG RAG Server"
LABEL org.opencontainers.image.description="Autonomous RAG server with ChromaDB, auto-indexing and configurable embeddings"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Cummunder <cummunder@users.noreply.github.com>"
LABEL org.opencontainers.image.url="https://github.com/cummunder/og-rag-server"
LABEL org.opencontainers.image.source="https://github.com/cummunder/og-rag-server"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.base.name="python:3.11-slim"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Update package list and install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    cron \
    inotify-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache transformer model for autonomous operation
RUN mkdir -p /app/data/models

# Set cache directories - HF_HOME is primary for transformers v5+
ENV HF_HOME=/app/data/models
ENV SENTENCE_TRANSFORMERS_HOME=/app/data/models
ENV HUGGINGFACE_HUB_CACHE=/app/data/models

# Keep online mode for model downloads
ENV HF_HUB_DISABLE_TELEMETRY=1

# Create directory structure
RUN mkdir -p /app/data/documents /app/data/vector_db /app/scripts /app/config /app/backups /app/logs

# Copy application files
COPY app/ /app/
COPY config/ /app/config/

# Create volume for persistent data
VOLUME /app/data
VOLUME /app/backups

# Default backup configuration
ENV BACKUP_ENABLED=true
ENV BACKUP_RETENTION_DAYS=7
ENV BACKUP_INTERVAL_HOURS=6

# Copy backup script
COPY scripts/backup_data.sh /app/scripts/backup_data.sh
RUN chmod +x /app/scripts/backup_data.sh

# Create enhanced health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Copy entrypoint script
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 5000

ENTRYPOINT ["/app/entrypoint.sh"]
