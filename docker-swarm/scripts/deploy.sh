#!/bin/bash
# =============================================================================
# RUVECTOR SWARM DEPLOYMENT SCRIPT
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"

echo "=============================================="
echo "RUVECTOR SWARM DEPLOYMENT"
echo "=============================================="

# =============================================================================
# PREREQUISITES CHECK
# =============================================================================
echo "[1/7] Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed"
    exit 1
fi

# Check Docker Swarm mode
if ! docker info --format '{{.Swarm.LocalNodeState}}' | grep -q "active"; then
    echo "WARNING: Docker Swarm not initialized"
    read -p "Initialize Docker Swarm? (yes/no): " INIT_SWARM
    if [ "${INIT_SWARM}" == "yes" ]; then
        docker swarm init
    else
        echo "ERROR: Docker Swarm is required"
        exit 1
    fi
fi

echo "  Docker: OK"
echo "  Swarm: OK"

# =============================================================================
# CREATE STORAGE DIRECTORIES
# =============================================================================
echo "[2/7] Creating storage directories..."

STORAGE_BASE="/var/lib/ruvector-swarm"
sudo mkdir -p \
    "${STORAGE_BASE}/memory" \
    "${STORAGE_BASE}/redis" \
    "${STORAGE_BASE}/backups" \
    "${STORAGE_BASE}/orchestrator"

sudo chown -R 1001:1001 "${STORAGE_BASE}"
echo "  Storage: ${STORAGE_BASE}"

# =============================================================================
# BUILD IMAGES
# =============================================================================
echo "[3/7] Building Docker images..."

cd "${PROJECT_DIR}"

# Build orchestrator
echo "  Building orchestrator..."
docker build -t ruvector-swarm/orchestrator:latest ./orchestrator

# Build workers
for worker in researcher coder tester reviewer coordinator; do
    echo "  Building worker-${worker}..."
    docker build -t "ruvector-swarm/worker-${worker}:latest" \
        -f "./workers/Dockerfile.${worker}" \
        ./workers
done

# Build memory backup
echo "  Building memory-backup..."
docker build -t ruvector-swarm/memory-backup:latest \
    -f ./memory/Dockerfile.backup \
    ./memory

echo "  All images built"

# =============================================================================
# CREATE NETWORKS
# =============================================================================
echo "[4/7] Creating overlay networks..."

networks=(
    "control-plane"
    "agent-mesh"
    "monitoring"
    "research-isolated"
    "coder-isolated"
    "tester-isolated"
    "reviewer-isolated"
)

for network in "${networks[@]}"; do
    if ! docker network ls --format '{{.Name}}' | grep -q "^${network}$"; then
        docker network create --driver overlay --attachable "${network}"
        echo "  Created: ${network}"
    else
        echo "  Exists: ${network}"
    fi
done

# =============================================================================
# DEPLOY STACK
# =============================================================================
echo "[5/7] Deploying stack..."

docker stack deploy -c docker-compose.yml ruvector-swarm

echo "  Stack deployed"

# =============================================================================
# WAIT FOR SERVICES
# =============================================================================
echo "[6/7] Waiting for services to start..."

services=(
    "ruvector-swarm_orchestrator"
    "ruvector-swarm_memory-store"
    "ruvector-swarm_memory-backup"
)

for service in "${services[@]}"; do
    echo -n "  Waiting for ${service}..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if docker service ls --format '{{.Name}} {{.Replicas}}' | grep "${service}" | grep -q "1/1"; then
            echo " OK"
            break
        fi
        sleep 2
        ((timeout-=2))
    done
    if [ $timeout -le 0 ]; then
        echo " TIMEOUT"
    fi
done

# =============================================================================
# VERIFY DEPLOYMENT
# =============================================================================
echo "[7/7] Verifying deployment..."

echo ""
echo "Services:"
docker service ls --filter "name=ruvector-swarm"

echo ""
echo "Networks:"
docker network ls --filter "name=ruvector"

echo ""
echo "=============================================="
echo "DEPLOYMENT COMPLETE"
echo "=============================================="
echo ""
echo "Endpoints:"
echo "  Orchestrator API: http://localhost:9000"
echo "  Metrics:          http://localhost:9001/metrics"
echo "  Prometheus:       http://localhost:9090"
echo "  Grafana:          http://localhost:3000"
echo ""
echo "Commands:"
echo "  View logs:    docker service logs ruvector-swarm_orchestrator"
echo "  Scale worker: docker service scale ruvector-swarm_worker-coder=2"
echo "  Stack status: docker stack ps ruvector-swarm"
echo ""
