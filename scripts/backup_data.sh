#!/bin/bash
set -e

# Check if backup is enabled
if [ "${BACKUP_ENABLED}" = "false" ]; then
    echo "Backup is disabled (BACKUP_ENABLED=false)"
    exit 0
fi

# Use environment variable or default to 7 days
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-7}"

BACKUP_DIR="/app/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r /app/data/* "$BACKUP_DIR/" 2>/dev/null || true
echo "Backup created: $BACKUP_DIR"

# Remove backups older than RETENTION_DAYS
find /app/backups -maxdepth 1 -type d -mtime +${RETENTION_DAYS} -exec rm -rf {} + 2>/dev/null || true
echo "Cleaned backups older than ${RETENTION_DAYS} days"
