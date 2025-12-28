# Expected System Logs - Simulation

This document shows the expected log output when the Docker Swarm is deployed and running.

---

## 1. Stack Deployment Logs

```
==============================================
RUVECTOR SWARM DEPLOYMENT
==============================================
[1/7] Checking prerequisites...
  Docker: OK
  Swarm: OK
[2/7] Creating storage directories...
  Storage: /var/lib/ruvector-swarm
[3/7] Building Docker images...
  Building orchestrator...
  Building worker-researcher...
  Building worker-coder...
  Building worker-tester...
  Building worker-reviewer...
  Building worker-coordinator...
  Building memory-backup...
  All images built
[4/7] Creating overlay networks...
  Created: control-plane
  Created: agent-mesh
  Created: monitoring
  Created: research-isolated
  Created: coder-isolated
  Created: tester-isolated
  Created: reviewer-isolated
[5/7] Deploying stack...
Creating service ruvector-swarm_memory-store
Creating service ruvector-swarm_memory-backup
Creating service ruvector-swarm_orchestrator
Creating service ruvector-swarm_worker-researcher
Creating service ruvector-swarm_worker-coder
Creating service ruvector-swarm_worker-coordinator
Creating service ruvector-swarm_prometheus
Creating service ruvector-swarm_grafana
  Stack deployed
[6/7] Waiting for services to start...
  Waiting for ruvector-swarm_orchestrator... OK
  Waiting for ruvector-swarm_memory-store... OK
  Waiting for ruvector-swarm_memory-backup... OK
[7/7] Verifying deployment...

Services:
ID             NAME                              MODE         REPLICAS   IMAGE
abc123         ruvector-swarm_orchestrator       replicated   1/1        ruvector-swarm/orchestrator:latest
def456         ruvector-swarm_memory-store       replicated   1/1        redis:7-alpine
ghi789         ruvector-swarm_memory-backup      replicated   1/1        ruvector-swarm/memory-backup:latest
jkl012         ruvector-swarm_worker-researcher  replicated   1/1        ruvector-swarm/worker-researcher:latest
mno345         ruvector-swarm_worker-coder       replicated   1/1        ruvector-swarm/worker-coder:latest
pqr678         ruvector-swarm_worker-coordinator replicated   1/1        ruvector-swarm/worker-coordinator:latest
stu901         ruvector-swarm_worker-tester      replicated   0/0        ruvector-swarm/worker-tester:latest
vwx234         ruvector-swarm_worker-reviewer    replicated   0/0        ruvector-swarm/worker-reviewer:latest
yza567         ruvector-swarm_prometheus         replicated   1/1        prom/prometheus:latest
bcd890         ruvector-swarm_grafana            replicated   1/1        grafana/grafana:latest

Networks:
NETWORK ID     NAME              DRIVER    SCOPE
net001         control-plane     overlay   swarm
net002         agent-mesh        overlay   swarm
net003         monitoring        overlay   swarm
net004         research-isolated overlay   swarm
net005         coder-isolated    overlay   swarm
net006         tester-isolated   overlay   swarm
net007         reviewer-isolated overlay   swarm

==============================================
DEPLOYMENT COMPLETE
==============================================

Endpoints:
  Orchestrator API: http://localhost:9000
  Metrics:          http://localhost:9001/metrics
  Prometheus:       http://localhost:9090
  Grafana:          http://localhost:3000
```

---

## 2. Orchestrator Logs

```
============================================================
RUVECTOR SWARM ORCHESTRATOR
============================================================
Worker Ratio: 1:2
Min Workers: 1
Max Workers: 10
============================================================
[STATE] Connected to Redis
[STATE] No persisted state found, starting fresh
[API] Server listening on port 9000
[METRICS] Server listening on port 9001
[MAIN] Orchestrator ready

# Task submission
[ROUTER] Task task-1703800000000-abc123def routed to coder
[SCALE] coder: current=1, pending=1, desired=1
[ROUTER] Task task-1703800001000-ghi456jkl routed to researcher
[ROUTER] Task task-1703800002000-mno789pqr routed to coder
[SCALE] coder: current=1, pending=2, desired=1

# Task completion
[ROUTER] Task task-1703800000000-abc123def completed
[ROUTER] Task task-1703800001000-ghi456jkl completed

# Scaling events
[SCALE] Evaluating scaling...
[SCALE] tester: current=0, pending=3, desired=2
[SCALE] Scaled ruvector-swarm_worker-tester to 2 replicas

# State persistence
[STATE] State persisted

# Graceful shutdown
[MAIN] Received SIGTERM, shutting down...
[STATE] State persisted
```

---

## 3. Worker Logs (Researcher Example)

```
============================================================
RUVECTOR SWARM WORKER - RESEARCHER
============================================================
Capabilities: explore, analyze, search, read, web
Priority: high
Idle Timeout: 120s
Max Concurrent Tasks: 2
============================================================
[WORKER:researcher-node1-1703800000000] Connected to Redis
[WORKER:researcher-node1-1703800000000] Registered with swarm
[HEALTH] Server listening on port 8080
[MAIN] Worker researcher-node1-1703800000000 ready, polling for tasks...

# Task execution
[EXECUTOR] Starting task task-1703800001000-ghi456jkl: Analyze codebase structure
[EXECUTOR] Task task-1703800001000-ghi456jkl completed in 2341ms
[POLLER] Task task-1703800001000-ghi456jkl completed

# Idle timeout
[POLLER] No tasks received for 120 seconds
[POLLER] Idle timeout reached, signaling shutdown
[MAIN] Received IDLE_TIMEOUT, shutting down...
[WORKER:researcher-node1-1703800000000] Deregistered from swarm
```

---

## 4. Memory Backup Logs

```
==============================================
RUVECTOR MEMORY BACKUP SERVICE
==============================================
Backup Interval: 300s
Retention: 7 days
Compression: gzip
Redis: memory-store:6379
==============================================
[INIT] Waiting for Redis...
[INIT] Redis is available

# Initial backup
[BACKUP] Starting backup at 20241228_120000
[BACKUP] Backing up Redis...
[BACKUP] Redis backup: /backups/hourly/redis_20241228_120000.rdb.gz
[BACKUP] Backing up shared memory...
[BACKUP] Memory backup: /backups/hourly/memory_20241228_120000.tar.gz
[BACKUP] Cleaning up old backups...
[BACKUP] Backup complete at Sat Dec 28 12:00:05 UTC 2024

# Scheduled backups
[BACKUP] Starting backup at 20241228_120500
[BACKUP] Backing up Redis...
[BACKUP] Redis backup: /backups/hourly/redis_20241228_120500.rdb.gz
[BACKUP] Backing up shared memory...
[BACKUP] Memory backup: /backups/hourly/memory_20241228_120500.tar.gz
[BACKUP] Cleaning up old backups...
[BACKUP] Removed 0 old hourly backups
[BACKUP] Backup complete at Sat Dec 28 12:05:03 UTC 2024

# Daily promotion (at midnight)
[BACKUP] Starting backup at 20241229_000000
[BACKUP] Backing up Redis...
[BACKUP] Redis backup: /backups/hourly/redis_20241229_000000.rdb.gz
[BACKUP] Backing up shared memory...
[BACKUP] Memory backup: /backups/hourly/memory_20241229_000000.tar.gz
[BACKUP] Promoting to daily backup...
[BACKUP] Cleaning up old backups...
[BACKUP] Removed 24 old hourly backups
[BACKUP] Backup complete at Sun Dec 29 00:00:07 UTC 2024
```

---

## 5. Scaling Event Logs

```
# Scale up due to task load
[SCALE] Evaluating scaling...
[SCALE] worker-tester: current=0, pending=4, desired=2
[SCALE] Cooldown active for worker-tester: false
[SCALE] Scaling worker-tester from 0 to 2
[SCALE] Scaled ruvector-swarm_worker-tester to 2 replicas
[SCALE] Scale event recorded: {service: worker-tester, replicas: 2, reason: demand}

# Scale down due to idle
[SCALE] Evaluating scaling...
[SCALE] worker-tester: current=2, pending=0, desired=0
[SCALE] Cooldown active for worker-tester: false
[SCALE] Note: worker-tester min replicas is 0, allowing scale to 0
[SCALE] Workers will self-terminate after idle timeout

# Worker self-termination
[POLLER] Idle timeout reached (90s), signaling shutdown
[MAIN] Worker tester-node1-1703800500000 shutting down...
[WORKER:tester-node1-1703800500000] Deregistered from swarm
```

---

## 6. API Request/Response Logs

```
# Health check
[API] GET /health 200 {"status":"healthy","uptime":3600000}

# Submit task
[API] POST /task 201 {"taskId":"task-1703800000000-abc123def","status":"pending"}
[ROUTER] Task task-1703800000000-abc123def routed to coder

# Get task status
[API] GET /task/task-1703800000000-abc123def 200
{
  "id": "task-1703800000000-abc123def",
  "workerType": "coder",
  "status": "completed",
  "createdAt": 1703800000000,
  "completedAt": 1703800002341,
  "result": {"success": true, "duration": 2341}
}

# Memory operations
[API] PUT /memory/swarm/plan/v1 200 {"success":true}
[API] GET /memory/swarm/plan/v1 200 {"steps":[...]}

# Manual scaling
[API] POST /scale/tester 200 {"success":true,"replicas":2}

# Trigger backup
[API] POST /backup 200 {"success":true,"timestamp":1703800300000}

# Swarm status
[API] GET /status 200
{
  "metrics": {
    "tasksCompleted": 47,
    "tasksActive": 3,
    "workersActive": 5,
    "uptime": 1703796400000
  },
  "workers": {
    "researcher": {"active": 1},
    "coder": {"active": 2},
    "tester": {"active": 1},
    "reviewer": {"active": 0},
    "coordinator": {"active": 1}
  },
  "tasks": {
    "total": 50,
    "pending": 3,
    "completed": 47
  }
}
```

---

## 7. Prometheus Metrics Output

```
# HELP swarm_tasks_total Total tasks processed
# TYPE swarm_tasks_total counter
swarm_tasks_total 47

# HELP swarm_tasks_active Currently active tasks
# TYPE swarm_tasks_active gauge
swarm_tasks_active 3

# HELP swarm_uptime_seconds Orchestrator uptime
# TYPE swarm_uptime_seconds gauge
swarm_uptime_seconds 3600

# HELP swarm_workers_active Active workers by type
# TYPE swarm_workers_active gauge
swarm_workers_active{type="researcher"} 1
swarm_workers_active{type="coder"} 2
swarm_workers_active{type="tester"} 1
swarm_workers_active{type="reviewer"} 0
swarm_workers_active{type="coordinator"} 1
```

---

## 8. Error Scenarios

### Redis Connection Failure
```
[STATE] Connecting to Redis...
[STATE] Redis connection failed: ECONNREFUSED
[STATE] Retrying in 1s... (attempt 1/3)
[STATE] Retrying in 2s... (attempt 2/3)
[STATE] Retrying in 4s... (attempt 3/3)
[STATE] Connected to Redis
```

### Task Execution Failure
```
[EXECUTOR] Starting task task-1703800000000-fail123: Process unknown file
[EXECUTOR] Task task-1703800000000-fail123 failed: File not found
[POLLER] Reporting failure to orchestrator
[API] PUT /task/task-1703800000000-fail123 200 {"success":false,"error":"File not found"}
```

### Network Partition Recovery
```
[POLLER] Redis connection lost
[POLLER] Queuing local operations...
[POLLER] Reconnecting to Redis...
[POLLER] Redis connection restored
[POLLER] Flushing 3 queued operations
[POLLER] Operations synced successfully
```
