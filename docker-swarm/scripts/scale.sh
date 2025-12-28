#!/bin/bash
# =============================================================================
# MANUAL SCALING SCRIPT
# =============================================================================

set -e

WORKER_TYPE="${1:-}"
REPLICAS="${2:-}"

if [ -z "${WORKER_TYPE}" ] || [ -z "${REPLICAS}" ]; then
    echo "Usage: $0 <worker-type> <replicas>"
    echo ""
    echo "Worker types:"
    echo "  researcher  - Exploration and analysis"
    echo "  coder       - Code implementation"
    echo "  tester      - Testing and validation"
    echo "  reviewer    - Code review and security"
    echo "  coordinator - Planning and orchestration"
    echo ""
    echo "Examples:"
    echo "  $0 coder 2        # Scale coders to 2 replicas"
    echo "  $0 tester 0       # Scale down testers to 0"
    echo "  $0 all 1          # Scale all workers to 1"
    exit 1
fi

STACK_NAME="ruvector-swarm"

if [ "${WORKER_TYPE}" == "all" ]; then
    echo "Scaling all workers to ${REPLICAS} replicas..."
    for type in researcher coder tester reviewer coordinator; do
        docker service scale "${STACK_NAME}_worker-${type}=${REPLICAS}"
    done
else
    echo "Scaling ${WORKER_TYPE} to ${REPLICAS} replicas..."
    docker service scale "${STACK_NAME}_worker-${WORKER_TYPE}=${REPLICAS}"
fi

echo ""
echo "Current worker status:"
docker service ls --filter "name=${STACK_NAME}_worker"
