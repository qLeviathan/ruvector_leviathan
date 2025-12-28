#!/bin/bash
# =============================================================================
# BACKUP SCRIPT - Redis + Shared Memory
# =============================================================================

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/hourly"
REDIS_BACKUP_FILE="${BACKUP_DIR}/redis_${TIMESTAMP}.rdb"
MEMORY_BACKUP_FILE="${BACKUP_DIR}/memory_${TIMESTAMP}.tar"

echo "[BACKUP] Starting backup at ${TIMESTAMP}"

# =============================================================================
# REDIS BACKUP
# =============================================================================
echo "[BACKUP] Backing up Redis..."

# Trigger Redis BGSAVE
redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" BGSAVE

# Wait for BGSAVE to complete
while [ "$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" LASTSAVE)" == "$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" LASTSAVE)" ]; do
    sleep 1
done

# Copy the RDB file
if [ -f /data/redis/dump.rdb ]; then
    cp /data/redis/dump.rdb "${REDIS_BACKUP_FILE}"

    # Compress if enabled
    if [ "${BACKUP_COMPRESSION}" == "gzip" ]; then
        gzip "${REDIS_BACKUP_FILE}"
        REDIS_BACKUP_FILE="${REDIS_BACKUP_FILE}.gz"
    fi

    echo "[BACKUP] Redis backup: ${REDIS_BACKUP_FILE}"
fi

# =============================================================================
# SHARED MEMORY BACKUP
# =============================================================================
echo "[BACKUP] Backing up shared memory..."

if [ -d /app/memory ] && [ "$(ls -A /app/memory)" ]; then
    tar -cf "${MEMORY_BACKUP_FILE}" -C /app/memory .

    if [ "${BACKUP_COMPRESSION}" == "gzip" ]; then
        gzip "${MEMORY_BACKUP_FILE}"
        MEMORY_BACKUP_FILE="${MEMORY_BACKUP_FILE}.gz"
    fi

    echo "[BACKUP] Memory backup: ${MEMORY_BACKUP_FILE}"
fi

# =============================================================================
# ROTATION & CLEANUP
# =============================================================================
echo "[BACKUP] Cleaning up old backups..."

# Keep only last 24 hourly backups
find /backups/hourly -type f -mmin +$((60 * 24)) -delete

# Promote to daily (keep 7 days)
HOUR=$(date +%H)
if [ "${HOUR}" == "00" ]; then
    cp "${REDIS_BACKUP_FILE}" /backups/daily/ 2>/dev/null || true
    cp "${MEMORY_BACKUP_FILE}" /backups/daily/ 2>/dev/null || true
    find /backups/daily -type f -mtime +7 -delete
fi

# Promote to weekly (keep 4 weeks)
DAY=$(date +%u)
if [ "${DAY}" == "7" ] && [ "${HOUR}" == "00" ]; then
    cp "${REDIS_BACKUP_FILE}" /backups/weekly/ 2>/dev/null || true
    cp "${MEMORY_BACKUP_FILE}" /backups/weekly/ 2>/dev/null || true
    find /backups/weekly -type f -mtime +28 -delete
fi

# =============================================================================
# S3 SYNC (if enabled)
# =============================================================================
if [ "${BACKUP_S3_ENABLED}" == "true" ] && [ -n "${BACKUP_S3_BUCKET}" ]; then
    echo "[BACKUP] Syncing to S3..."
    aws s3 sync /backups "s3://${BACKUP_S3_BUCKET}/ruvector-swarm/" \
        --storage-class STANDARD_IA \
        --delete
    echo "[BACKUP] S3 sync complete"
fi

# =============================================================================
# RECORD BACKUP METADATA
# =============================================================================
cat > /backups/latest.json << EOF
{
    "timestamp": "${TIMESTAMP}",
    "redis_file": "${REDIS_BACKUP_FILE}",
    "memory_file": "${MEMORY_BACKUP_FILE}",
    "redis_size": $(stat -f%z "${REDIS_BACKUP_FILE}" 2>/dev/null || stat -c%s "${REDIS_BACKUP_FILE}" 2>/dev/null || echo 0),
    "memory_size": $(stat -f%z "${MEMORY_BACKUP_FILE}" 2>/dev/null || stat -c%s "${MEMORY_BACKUP_FILE}" 2>/dev/null || echo 0)
}
EOF

echo "[BACKUP] Backup complete at $(date)"
