#!/bin/bash
# =============================================================================
# RESTORE SCRIPT - Redis + Shared Memory
# =============================================================================

set -e

BACKUP_TYPE="${1:-latest}"  # latest, hourly, daily, weekly
BACKUP_FILE="${2:-}"        # specific file or empty for most recent

echo "=============================================="
echo "RUVECTOR MEMORY RESTORE"
echo "=============================================="
echo "Type: ${BACKUP_TYPE}"
echo "File: ${BACKUP_FILE:-auto-detect}"
echo "=============================================="

# =============================================================================
# FIND BACKUP FILES
# =============================================================================
if [ "${BACKUP_TYPE}" == "latest" ]; then
    if [ -f /backups/latest.json ]; then
        REDIS_FILE=$(jq -r '.redis_file' /backups/latest.json)
        MEMORY_FILE=$(jq -r '.memory_file' /backups/latest.json)
    else
        echo "[ERROR] No latest.json found"
        exit 1
    fi
elif [ -n "${BACKUP_FILE}" ]; then
    REDIS_FILE="${BACKUP_FILE}"
    MEMORY_FILE=$(echo "${BACKUP_FILE}" | sed 's/redis_/memory_/')
else
    BACKUP_DIR="/backups/${BACKUP_TYPE}"
    REDIS_FILE=$(ls -t "${BACKUP_DIR}"/redis_*.rdb* 2>/dev/null | head -1)
    MEMORY_FILE=$(ls -t "${BACKUP_DIR}"/memory_*.tar* 2>/dev/null | head -1)
fi

if [ -z "${REDIS_FILE}" ] && [ -z "${MEMORY_FILE}" ]; then
    echo "[ERROR] No backup files found"
    exit 1
fi

# =============================================================================
# CONFIRMATION
# =============================================================================
echo ""
echo "Will restore from:"
echo "  Redis:  ${REDIS_FILE}"
echo "  Memory: ${MEMORY_FILE}"
echo ""
echo "WARNING: This will overwrite current data!"
read -p "Continue? (yes/no): " CONFIRM

if [ "${CONFIRM}" != "yes" ]; then
    echo "[ABORT] Restore cancelled"
    exit 0
fi

# =============================================================================
# RESTORE REDIS
# =============================================================================
if [ -f "${REDIS_FILE}" ]; then
    echo "[RESTORE] Stopping Redis writes..."
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" CONFIG SET appendonly no
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" BGSAVE
    sleep 2

    echo "[RESTORE] Restoring Redis data..."

    # Decompress if needed
    if [[ "${REDIS_FILE}" == *.gz ]]; then
        gunzip -c "${REDIS_FILE}" > /tmp/restore.rdb
        REDIS_FILE="/tmp/restore.rdb"
    fi

    # Copy to Redis data directory
    cp "${REDIS_FILE}" /data/redis/dump.rdb

    # Restart Redis to load new data
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" DEBUG RELOAD

    # Re-enable AOF
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" CONFIG SET appendonly yes
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" BGREWRITEAOF

    echo "[RESTORE] Redis restored successfully"
fi

# =============================================================================
# RESTORE SHARED MEMORY
# =============================================================================
if [ -f "${MEMORY_FILE}" ]; then
    echo "[RESTORE] Restoring shared memory..."

    # Clear existing memory
    rm -rf /app/memory/*

    # Decompress if needed
    if [[ "${MEMORY_FILE}" == *.gz ]]; then
        gunzip -c "${MEMORY_FILE}" | tar -xf - -C /app/memory
    else
        tar -xf "${MEMORY_FILE}" -C /app/memory
    fi

    echo "[RESTORE] Shared memory restored successfully"
fi

# =============================================================================
# VERIFICATION
# =============================================================================
echo ""
echo "[VERIFY] Checking restored data..."

REDIS_KEYS=$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" DBSIZE | awk '{print $2}')
MEMORY_FILES=$(find /app/memory -type f | wc -l)

echo "  Redis keys: ${REDIS_KEYS}"
echo "  Memory files: ${MEMORY_FILES}"

echo ""
echo "[RESTORE] Complete!"
