# RuVector Docker Swarm Architecture

## System Overview

This Docker Swarm implementation provides a **self-scaling, isolated, memory-persistent** orchestration system for RuVector agents.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DOCKER SWARM CLUSTER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      ALWAYS-ON SERVICES                              │   │
│  │  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ ORCHESTRATOR│  │ MEMORY-STORE │  │MEMORY-BACKUP │               │   │
│  │  │   (Node.js) │  │   (Redis)    │  │  (Scripts)   │               │   │
│  │  │             │  │              │  │              │               │   │
│  │  │ - API       │  │ - State      │  │ - Hourly     │               │   │
│  │  │ - Scaling   │  │ - Tasks      │  │ - Daily      │               │   │
│  │  │ - Routing   │  │ - Memory     │  │ - S3 Sync    │               │   │
│  │  └─────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    DYNAMIC WORKERS (1:2 RATIO)                       │   │
│  │                                                                       │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │RESEARCHER│ │  CODER   │ │  TESTER  │ │ REVIEWER │ │COORDINATOR│ │   │
│  │  │ min:1    │ │ min:1    │ │ min:0    │ │ min:0    │ │ min:1    │  │   │
│  │  │ max:2    │ │ max:2    │ │ max:2    │ │ max:2    │ │ max:2    │  │   │
│  │  │          │ │          │ │          │ │          │ │          │  │   │
│  │  │ explore  │ │ write    │ │ test     │ │ review   │ │ plan     │  │   │
│  │  │ analyze  │ │ edit     │ │ validate │ │ audit    │ │ delegate │  │   │
│  │  │ search   │ │ refactor │ │ benchmark│ │ lint     │ │ merge    │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         MONITORING                                   │   │
│  │  ┌─────────────┐              ┌─────────────┐                       │   │
│  │  │ PROMETHEUS  │──────────────│   GRAFANA   │                       │   │
│  │  │   :9090     │              │    :3000    │                       │   │
│  │  └─────────────┘              └─────────────┘                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Design Decisions

### Decision 1: 1:2 Worker Ratio

**Problem**: How many tasks can one worker handle simultaneously?

**Decision**: Each worker handles up to 2 concurrent tasks (1:2 ratio)

**Rationale**:
- Workers are I/O bound, not CPU bound
- Waiting for file operations = idle time
- 2 tasks allows overlap of wait times
- Higher ratios cause context switching overhead

**Implementation**:
```javascript
const neededWorkers = Math.ceil(pendingTasks / 2);
const desired = Math.max(minReplicas, Math.min(maxReplicas, neededWorkers));
```

### Decision 2: Core Workers Always Running

**Problem**: Some workers are needed immediately, others on-demand

**Decision**: 3 workers always running (researcher, coder, coordinator)

| Worker | Min Replicas | Reason |
|--------|--------------|--------|
| Researcher | 1 | Initial exploration always needed |
| Coder | 1 | Most common operation |
| Coordinator | 1 | Task routing requires instant response |
| Tester | 0 | Only needed after code changes |
| Reviewer | 0 | Only needed before commits |

**Trade-off**: Higher baseline cost for faster response time

### Decision 3: Redis for Memory

**Problem**: Workers need shared state that survives restarts

**Decision**: Redis with AOF persistence + periodic RDB snapshots

**Rationale**:
- Sub-millisecond latency for memory operations
- AOF ensures no data loss on crash
- RDB snapshots for efficient backup/restore
- Native pub/sub for event broadcasting

**Configuration**:
```
appendonly yes          # Write every operation
appendfsync everysec    # Fsync every second
save 60 1000           # Snapshot if 1000 writes in 60s
save 300 100           # Snapshot if 100 writes in 5min
```

### Decision 4: Task Queue per Worker Type

**Problem**: How to route tasks to the right worker?

**Decision**: Separate Redis list per worker type

```
tasks:researcher  ← [task1, task2, ...]
tasks:coder       ← [task3, task4, ...]
tasks:tester      ← [task5, ...]
```

**Rationale**:
- No contention between worker types
- Workers only poll their own queue
- BRPOP provides efficient blocking wait
- Easy to see queue depths per type

### Decision 5: Idle Timeout Auto-Shutdown

**Problem**: Dynamic workers should despawn when not needed

**Decision**: Workers self-terminate after idle timeout

| Worker | Idle Timeout | Reason |
|--------|--------------|--------|
| Researcher | 120s | Exploration comes in bursts |
| Coder | 180s | Code sessions are longer |
| Tester | 90s | Tests run quickly |
| Reviewer | 90s | Reviews are quick |
| Coordinator | 300s | Must stay available longer |

**Implementation**:
```javascript
if (Date.now() - this.lastActivity > CONFIG.idleTimeout * 1000) {
    this.running = false;  // Signal graceful shutdown
}
```

## Data Flow

### Task Submission
```
Client
  │
  ▼
┌──────────────┐
│ Orchestrator │ ←── POST /task
│   API        │
└──────┬───────┘
       │
       ▼ Analyze capabilities
┌──────────────┐
│ Task Router  │
└──────┬───────┘
       │
       ▼ LPUSH to appropriate queue
┌──────────────┐
│    Redis     │ ←── tasks:coder
└──────────────┘
```

### Task Execution
```
┌──────────────┐
│    Worker    │
└──────┬───────┘
       │
       ▼ BRPOP with 5s timeout
┌──────────────┐
│    Redis     │ ←── tasks:coder
└──────┬───────┘
       │
       ▼ Execute task
┌──────────────┐
│  Task Logic  │
└──────┬───────┘
       │
       ▼ Store result
┌──────────────┐
│    Redis     │ ←── task:{id}:result
└──────┬───────┘
       │
       ▼ Notify orchestrator
┌──────────────┐
│ Orchestrator │ ←── PUT /task/{id}
└──────────────┘
```

### Memory Coordination
```
Worker A                    Worker B
   │                           │
   ▼ Write to shared memory    │
┌──────────────┐              │
│    Redis     │ SET memory:plan:v1
└──────────────┘              │
                              ▼ Read shared memory
                         ┌──────────────┐
                         │    Redis     │ GET memory:plan:v1
                         └──────────────┘
```

## Scaling Algorithm

```
Every 30 seconds:
  FOR each worker_type:
    pending = count(tasks in queue)
    active = count(running workers)

    needed = ceil(pending / 2)  # 1:2 ratio
    desired = clamp(needed, min_replicas, max_replicas)

    IF desired != active AND cooldown_elapsed:
      scale(worker_type, desired)
      reset_cooldown(worker_type)
```

### Scale-Up Triggers
- Task queue depth > active_workers * 2
- Cooldown: 60 seconds between scale events

### Scale-Down Triggers
- Worker idle timeout reached
- Task queue empty for > idle_timeout
- Worker self-terminates gracefully

## Backup Strategy

### Continuous
- Redis AOF: Every write appended to log
- Memory: File changes tracked by inotify

### Periodic (Every 5 minutes)
- Redis BGSAVE triggered
- Memory directory tarball created
- Compressed with gzip

### Retention
| Type | Retention | Storage |
|------|-----------|---------|
| Hourly | 24 backups | Local volume |
| Daily | 7 days | Local volume |
| Weekly | 4 weeks | Local + S3 (optional) |

### Recovery Time
- **Redis restore**: < 30 seconds for 1GB dataset
- **Memory restore**: < 10 seconds for typical state
- **Full cluster recovery**: < 5 minutes

## Resource Allocation

### Per-Worker Limits

| Worker | CPU Limit | Memory Limit | Reservation |
|--------|-----------|--------------|-------------|
| Orchestrator | 2 | 2GB | 0.5 CPU, 512MB |
| Researcher | 1 | 1GB | 0.25 CPU, 256MB |
| Coder | 2 | 2GB | 0.5 CPU, 512MB |
| Tester | 1 | 1GB | 0.25 CPU, 256MB |
| Reviewer | 1 | 512MB | 0.25 CPU, 128MB |
| Coordinator | 1 | 1GB | 0.25 CPU, 256MB |

### Cluster Sizing

| Swarm Size | Max Workers | Recommended Node Spec |
|------------|-------------|----------------------|
| Small (dev) | 5 | 4 CPU, 8GB RAM |
| Medium | 10 | 8 CPU, 16GB RAM per node x 2 |
| Large | 20 | 16 CPU, 32GB RAM per node x 3 |

## Failure Modes

### Worker Crash
1. Docker Swarm detects unhealthy container
2. Replaces with new instance
3. New worker reconnects to Redis
4. Resumes polling task queue
5. In-flight task may be re-queued

### Redis Crash
1. Memory-backup has latest RDB
2. Redis restarts, loads RDB
3. AOF replay recovers recent writes
4. Workers reconnect automatically

### Orchestrator Crash
1. Swarm restarts orchestrator
2. State loaded from Redis
3. API available within seconds
4. No task loss (queued in Redis)

### Network Partition
1. Workers in partition continue locally
2. Memory writes queue until reconnect
3. Orchestrator maintains last-known state
4. Reconciliation on partition heal
