#!/bin/bash
# =============================================================================
# TRIGGER IMMEDIATE BACKUP
# =============================================================================

set -e

echo "Triggering immediate backup..."

# Method 1: Via orchestrator API
if curl -sf http://localhost:9000/health > /dev/null 2>&1; then
    curl -X POST http://localhost:9000/backup
    echo "Backup triggered via orchestrator API"
else
    # Method 2: Direct execution in backup container
    CONTAINER_ID=$(docker ps -q --filter "name=memory-backup")
    if [ -n "${CONTAINER_ID}" ]; then
        docker exec "${CONTAINER_ID}" /app/scripts/backup.sh
        echo "Backup triggered directly in container"
    else
        echo "ERROR: No backup service found"
        exit 1
    fi
fi

echo ""
echo "Latest backup:"
if [ -f /var/lib/ruvector-swarm/backups/latest.json ]; then
    cat /var/lib/ruvector-swarm/backups/latest.json | jq .
else
    echo "No backup info available yet"
fi
