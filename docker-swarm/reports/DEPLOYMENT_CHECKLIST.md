# Deployment Checklist

Use this checklist before deploying the RuVector Docker Swarm to production.

---

## Pre-Deployment Requirements

### Infrastructure
- [ ] Docker Engine 20.10+ installed
- [ ] Docker Swarm initialized (`docker swarm init`)
- [ ] Minimum 1 manager node, recommended 3 for HA
- [ ] Worker nodes joined to swarm (if multi-node)
- [ ] At least 8GB RAM available per node
- [ ] At least 4 CPU cores available per node
- [ ] 50GB+ disk space for volumes and backups

### Networking
- [ ] Port 9000 open for Orchestrator API
- [ ] Port 9001 open for Metrics endpoint
- [ ] Port 9090 open for Prometheus (optional)
- [ ] Port 3000 open for Grafana (optional)
- [ ] Port 6379 blocked externally (Redis internal only)
- [ ] Firewall rules configured for swarm ports (2377, 7946, 4789)

### Storage
- [ ] `/var/lib/ruvector-swarm/memory` directory created
- [ ] `/var/lib/ruvector-swarm/redis` directory created
- [ ] `/var/lib/ruvector-swarm/backups` directory created
- [ ] `/var/lib/ruvector-swarm/orchestrator` directory created
- [ ] Directories have correct permissions (1001:1001)
- [ ] Sufficient disk space for backups (10GB+ recommended)

---

## Configuration

### Environment Variables
- [ ] `BACKUP_S3_ENABLED` set if using S3 backups
- [ ] `BACKUP_S3_BUCKET` configured if S3 enabled
- [ ] `AWS_ACCESS_KEY_ID` set if S3 enabled
- [ ] `AWS_SECRET_ACCESS_KEY` set if S3 enabled
- [ ] `GRAFANA_PASSWORD` changed from default

### Docker Compose Overrides (if needed)
- [ ] Resource limits adjusted for cluster capacity
- [ ] Replica counts adjusted for workload
- [ ] Volume paths adjusted for infrastructure
- [ ] Network subnets adjusted if conflicting

---

## Deployment Steps

### 1. Build Images
```bash
cd docker-swarm
./scripts/deploy.sh  # Handles build automatically
```
- [ ] Orchestrator image built successfully
- [ ] All 5 worker images built successfully
- [ ] Memory-backup image built successfully

### 2. Create Networks
- [ ] control-plane network created
- [ ] agent-mesh network created
- [ ] monitoring network created
- [ ] research-isolated network created
- [ ] coder-isolated network created
- [ ] tester-isolated network created
- [ ] reviewer-isolated network created

### 3. Deploy Stack
```bash
docker stack deploy -c docker-compose.yml ruvector-swarm
```
- [ ] Stack deployed without errors
- [ ] All services created

### 4. Verify Services
```bash
docker service ls --filter "name=ruvector-swarm"
```
- [ ] orchestrator: 1/1 replicas
- [ ] memory-store: 1/1 replicas
- [ ] memory-backup: 1/1 replicas
- [ ] worker-researcher: 1/1 replicas
- [ ] worker-coder: 1/1 replicas
- [ ] worker-coordinator: 1/1 replicas
- [ ] worker-tester: 0/0 replicas (on-demand)
- [ ] worker-reviewer: 0/0 replicas (on-demand)
- [ ] prometheus: 1/1 replicas (if enabled)
- [ ] grafana: 1/1 replicas (if enabled)

---

## Post-Deployment Verification

### Health Checks
```bash
curl http://localhost:9000/health
```
- [ ] Returns `{"status":"healthy",...}`

### API Functionality
```bash
# Submit test task
curl -X POST http://localhost:9000/task \
  -H "Content-Type: application/json" \
  -d '{"description":"test","capabilities":["explore"]}'
```
- [ ] Returns task ID
- [ ] Task routes to researcher worker

### Memory Operations
```bash
# Write memory
curl -X PUT http://localhost:9000/memory/test \
  -H "Content-Type: application/json" \
  -d '{"value":"test"}'

# Read memory
curl http://localhost:9000/memory/test
```
- [ ] Memory write succeeds
- [ ] Memory read returns correct value

### Backup Verification
```bash
ls -la /var/lib/ruvector-swarm/backups/hourly/
cat /var/lib/ruvector-swarm/backups/latest.json
```
- [ ] Backup files exist
- [ ] latest.json contains recent timestamp

### Metrics Endpoint
```bash
curl http://localhost:9001/metrics
```
- [ ] Returns Prometheus-format metrics
- [ ] `swarm_tasks_total` present
- [ ] `swarm_workers_active` present

### Scaling Test
```bash
# Scale up
./scripts/scale.sh tester 2

# Verify
docker service ls --filter "name=ruvector-swarm_worker-tester"

# Scale down
./scripts/scale.sh tester 0
```
- [ ] Scale up succeeds
- [ ] Workers start within 30s
- [ ] Scale down succeeds
- [ ] Workers terminate after idle timeout

---

## Monitoring Setup (Optional)

### Prometheus
- [ ] Prometheus UI accessible at :9090
- [ ] Targets show UP status
- [ ] Swarm metrics visible in explorer

### Grafana
- [ ] Grafana UI accessible at :3000
- [ ] Default password changed
- [ ] Prometheus data source configured
- [ ] Dashboard imported/created

---

## Security Verification

### Network Isolation
```bash
# Verify internal networks
docker network inspect agent-mesh | grep -A5 "Internal"
```
- [ ] agent-mesh is Internal: true
- [ ] isolated networks are Internal: true
- [ ] control-plane allows external (for API)

### Redis Security
- [ ] Redis not accessible from outside swarm
- [ ] Redis only on control-plane network

### Container Security
- [ ] Containers run as non-root (uid 1001)
- [ ] Docker socket only mounted to orchestrator (read-only)
- [ ] No privileged containers

---

## Rollback Plan

If deployment fails:
```bash
# Remove stack
docker stack rm ruvector-swarm

# Remove networks
docker network rm control-plane agent-mesh monitoring \
  research-isolated coder-isolated tester-isolated reviewer-isolated

# Restore from backup if needed
./memory/scripts/restore.sh latest
```

---

## Maintenance Schedule

### Daily
- [ ] Check backup files exist
- [ ] Review orchestrator logs for errors
- [ ] Verify all core workers running

### Weekly
- [ ] Review scaling patterns
- [ ] Check disk space for backups
- [ ] Update container images if needed

### Monthly
- [ ] Test restore procedure
- [ ] Review and rotate secrets
- [ ] Performance review from Prometheus

---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| Deployer | | | |
| Reviewer | | | |
| Approver | | | |
