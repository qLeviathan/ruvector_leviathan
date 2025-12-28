#!/bin/bash
# =============================================================================
# MEMORY BACKUP SERVICE ENTRYPOINT
# =============================================================================

set -e

echo "=============================================="
echo "RUVECTOR MEMORY BACKUP SERVICE"
echo "=============================================="
echo "Backup Interval: ${BACKUP_INTERVAL}s"
echo "Retention: ${BACKUP_RETENTION_DAYS} days"
echo "Compression: ${BACKUP_COMPRESSION}"
echo "Redis: ${REDIS_HOST}:${REDIS_PORT}"
echo "=============================================="

# Wait for Redis to be available
echo "[INIT] Waiting for Redis..."
until redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping > /dev/null 2>&1; do
    sleep 1
done
echo "[INIT] Redis is available"

# Start cron daemon in background
crond -b -l 8

# Start health check server in background
./scripts/health-server.sh &

# Run initial backup
./scripts/backup.sh

# Main backup loop
while true; do
    sleep "${BACKUP_INTERVAL}"
    ./scripts/backup.sh
done
