#!/bin/bash
set -e

echo "Starting Enhanced RAG System Container..."

# Initialize data directories
mkdir -p /app/data/documents /app/data/vector_db /app/data/models /app/backups /app/logs

# Set autonomous operation environment
export TRANSFORMERS_CACHE=/app/data/models
export SENTENCE_TRANSFORMERS_HOME=/app/data/models
export HF_HOME=/app/data/models
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0

echo "Autonomous mode: ENABLED"
echo "Models cached at: /app/data/models"

# Start cron service
echo "Starting cron service..."
cron

# Setup automatic backups based on configuration
if [ "${BACKUP_ENABLED}" = "true" ]; then
    INTERVAL="${BACKUP_INTERVAL_HOURS:-6}"
    echo "Backup enabled: interval=${INTERVAL}h, retention=${BACKUP_RETENTION_DAYS:-7} days"
    echo "0 */${INTERVAL} * * * /app/scripts/backup_data.sh > /dev/null 2>&1" | crontab -
else
    echo "Backup disabled (BACKUP_ENABLED=false)"
    crontab -r 2>/dev/null || true
fi

# Documents will be processed by Python server on startup
echo "Documents will be processed by Python server on startup..."

# Start RAG HTTP server
echo "Starting RAG HTTP server on port 5000..."
exec python server.py
