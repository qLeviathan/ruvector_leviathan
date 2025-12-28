#!/bin/bash
# =============================================================================
# HEALTH CHECK SERVER
# Simple HTTP server for backup service health checks
# =============================================================================

while true; do
    # Check Redis connectivity
    if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping > /dev/null 2>&1; then
        REDIS_STATUS="healthy"
    else
        REDIS_STATUS="unhealthy"
    fi

    # Get backup info
    if [ -f /backups/latest.json ]; then
        LAST_BACKUP=$(jq -r '.timestamp' /backups/latest.json)
    else
        LAST_BACKUP="never"
    fi

    # Count backups
    HOURLY_COUNT=$(ls /backups/hourly/*.rdb* 2>/dev/null | wc -l)
    DAILY_COUNT=$(ls /backups/daily/*.rdb* 2>/dev/null | wc -l)
    WEEKLY_COUNT=$(ls /backups/weekly/*.rdb* 2>/dev/null | wc -l)

    # Generate health response
    HEALTH_JSON=$(cat << EOF
{
    "status": "${REDIS_STATUS}",
    "last_backup": "${LAST_BACKUP}",
    "backups": {
        "hourly": ${HOURLY_COUNT},
        "daily": ${DAILY_COUNT},
        "weekly": ${WEEKLY_COUNT}
    },
    "retention_days": ${BACKUP_RETENTION_DAYS}
}
EOF
)

    # Respond to HTTP requests on port 8080
    echo -e "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${#HEALTH_JSON}\r\n\r\n${HEALTH_JSON}" | nc -l -p 8080 -q 1 || sleep 1
done
