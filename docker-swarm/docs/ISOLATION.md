# Network Isolation Architecture

## Overview

The RuVector Swarm uses **explicit network isolation** to enforce security boundaries between worker types. Each worker operates in its own isolated network segment while sharing access to the common agent-mesh for inter-worker communication.

## Network Topology

```
                    ┌─────────────────────────────────────────────────────────┐
                    │                    EXTERNAL ACCESS                       │
                    │                   (Ports: 9000, 3000)                    │
                    └─────────────────────────────────────────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────────────┐
                    │              CONTROL-PLANE NETWORK (10.10.0.0/24)         │
                    │  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐  │
                    │  │ Orchestrator │  │  Memory-Store  │  │Memory-Backup │  │
                    │  │   :9000      │  │    (Redis)     │  │              │  │
                    │  └──────────────┘  └────────────────┘  └──────────────┘  │
                    └─────────────────────────────────────────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────────────────┐
                    │               AGENT-MESH NETWORK (10.20.0.0/24)           │
                    │                   (Internal Only)                         │
                    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────┐ │
                    │  │Researcher│ │ Coder  │ │ Tester │ │Reviewer│ │Coord│ │
                    │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └──┬──┘ │
                    └───────┼───────────┼───────────┼───────────┼─────────┼────┘
                            │           │           │           │         │
            ┌───────────────▼──┐  ┌─────▼──────┐  ┌─▼────────┐  ┌▼───────┐│
            │ RESEARCH-ISOLATED│  │CODER-ISOL. │  │TESTER-IS.│  │REVIEW. ││
            │  (10.100.0.0/24) │  │(10.101.0/24│  │10.102.0/24│ │10.103. ││
            │    [Internal]    │  │ [Internal] │  │[Internal] │ │[Intern]││
            └──────────────────┘  └────────────┘  └───────────┘ └────────┘│
                                                                          │
                    ┌─────────────────────────────────────────────────────▼────┐
                    │              MONITORING NETWORK (10.30.0.0/24)           │
                    │  ┌──────────────┐            ┌──────────────┐            │
                    │  │  Prometheus  │            │   Grafana    │            │
                    │  │    :9090     │            │    :3000     │            │
                    │  └──────────────┘            └──────────────┘            │
                    └──────────────────────────────────────────────────────────┘
```

## Network Segments

### 1. Control-Plane Network (`10.10.0.0/24`)

**Purpose**: Management and coordination traffic

| Service | IP Range | Access |
|---------|----------|--------|
| Orchestrator | 10.10.0.10 | External + Internal |
| Memory-Store (Redis) | 10.10.0.20 | Internal Only |
| Memory-Backup | 10.10.0.30 | Internal Only |

**Security Properties**:
- Only orchestrator is exposed externally
- Redis has no external access
- All management commands flow through orchestrator

### 2. Agent-Mesh Network (`10.20.0.0/24`)

**Purpose**: Inter-worker communication

| Worker Type | IP Range |
|-------------|----------|
| Researcher | 10.20.0.10-19 |
| Coder | 10.20.0.20-29 |
| Tester | 10.20.0.30-39 |
| Reviewer | 10.20.0.40-49 |
| Coordinator | 10.20.0.50-59 |

**Security Properties**:
- Marked `internal: true` - no external routing
- All workers can communicate with each other
- Memory access via Redis on control-plane

### 3. Worker-Isolated Networks

Each worker type has its own isolated network:

| Network | Subnet | Purpose |
|---------|--------|---------|
| research-isolated | 10.100.0.0/24 | Web access, external APIs |
| coder-isolated | 10.101.0.0/24 | Workspace isolation |
| tester-isolated | 10.102.0.0/24 | Test execution sandbox |
| reviewer-isolated | 10.103.0.0/24 | Security scanning |

**Security Properties**:
- Each is `internal: true`
- Workers only see peers of same type
- Prevents cross-contamination during execution

### 4. Monitoring Network (`10.30.0.0/24`)

**Purpose**: Metrics collection and visualization

| Service | Role |
|---------|------|
| Prometheus | Metrics aggregation |
| Grafana | Dashboards |

## Isolation Decision Log

### Decision 1: Overlay Networks
```
Problem: Workers on different Swarm nodes need to communicate
Decision: Use Docker overlay networks with Swarm
Rationale: Native encrypted communication, automatic routing
Trade-off: Slight latency overhead (~1-2ms)
```

### Decision 2: Internal Networks
```
Problem: Prevent workers from accessing external resources directly
Decision: Mark worker networks as `internal: true`
Rationale: Forces all external access through orchestrator
Trade-off: Requires orchestrator proxy for external APIs
```

### Decision 3: Per-Worker-Type Isolation
```
Problem: Prevent compromised worker from affecting others
Decision: Separate network per worker type
Rationale: Blast radius limited to single worker type
Trade-off: Increased network complexity
```

### Decision 4: Redis on Control-Plane Only
```
Problem: Memory must be shared but protected
Decision: Redis only accessible from control-plane
Rationale: Central point of control for all memory access
Trade-off: All workers route through agent-mesh to access memory
```

## Network Rules Summary

| From → To | Control-Plane | Agent-Mesh | Isolated | Monitoring | External |
|-----------|---------------|------------|----------|------------|----------|
| Orchestrator | ✅ | ✅ | ❌ | ✅ | ✅ |
| Redis | ✅ | ❌ | ❌ | ❌ | ❌ |
| Workers | ❌ | ✅ | ✅ (own) | ❌ | ❌ |
| Prometheus | ✅ | ✅ | ❌ | ✅ | ❌ |
| Grafana | ❌ | ❌ | ❌ | ✅ | ✅ |

## Volume Isolation

| Volume | Access Mode | Shared By |
|--------|-------------|-----------|
| shared-memory | Read/Write | All workers, Orchestrator |
| redis-data | Read/Write | Redis only |
| backup-storage | Read/Write | Backup service only |
| workspace | Read/Write | Coder only, Read for Tester |
| test-results | Read/Write | Tester only |

## Security Recommendations

1. **Enable Swarm Encryption**: `docker swarm update --autolock=true`
2. **Rotate Encryption Keys**: Regular key rotation for overlay networks
3. **Limit Manager Nodes**: Only 1-3 manager nodes, rest as workers
4. **Use Secrets**: Store sensitive config in Docker secrets, not env vars
5. **Network Policies**: Consider Calico for additional network policies

## Disaster Recovery

### Network Failure
If overlay networks fail, workers will be isolated but safe:
- Memory persisted in Redis with AOF
- Backup service continues independently
- Orchestrator can rebuild networks on recovery

### Full Cluster Recovery
1. Restore Redis from backup: `./scripts/restore.sh latest`
2. Redeploy stack: `./scripts/deploy.sh`
3. Workers will auto-reconnect to Redis
4. Memory state restored from backup
