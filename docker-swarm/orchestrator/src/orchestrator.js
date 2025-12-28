/**
 * =============================================================================
 * RUVECTOR SWARM ORCHESTRATOR
 * =============================================================================
 * Central command for:
 * - Worker spawn/despawn with 1:2 ratio
 * - Memory coordination and persistence
 * - Task routing and load balancing
 * - Health monitoring and auto-recovery
 * =============================================================================
 */

const http = require('http');
const { exec, spawn } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');
const Redis = require('ioredis');

const execAsync = promisify(exec);

// =============================================================================
// CONFIGURATION
// =============================================================================
const CONFIG = {
  port: parseInt(process.env.PORT || '9000'),
  metricsPort: parseInt(process.env.METRICS_PORT || '9001'),
  workerRatio: process.env.WORKER_RATIO || '1:2',
  minWorkers: parseInt(process.env.MIN_WORKERS || '1'),
  maxWorkers: parseInt(process.env.MAX_WORKERS || '10'),
  scaleCooldown: parseInt(process.env.SCALE_COOLDOWN || '60'),
  memoryPersistence: process.env.MEMORY_PERSISTENCE === 'true',
  redisUrl: process.env.MEMORY_REDIS_URL || 'redis://memory-store:6379',
  backupPath: process.env.BACKUP_LOCAL_PATH || '/app/backups',
  statePath: '/app/state',
};

// Worker type configurations
const WORKER_TYPES = {
  researcher: {
    service: 'worker-researcher',
    minReplicas: 1,
    maxReplicas: 2,
    priority: 'high',
    idleTimeout: 120,
    capabilities: ['explore', 'analyze', 'search', 'read', 'web'],
  },
  coder: {
    service: 'worker-coder',
    minReplicas: 1,
    maxReplicas: 2,
    priority: 'high',
    idleTimeout: 180,
    capabilities: ['write', 'edit', 'create', 'refactor', 'implement'],
  },
  tester: {
    service: 'worker-tester',
    minReplicas: 0,
    maxReplicas: 2,
    priority: 'medium',
    idleTimeout: 90,
    capabilities: ['test', 'validate', 'benchmark', 'coverage'],
  },
  reviewer: {
    service: 'worker-reviewer',
    minReplicas: 0,
    maxReplicas: 2,
    priority: 'medium',
    idleTimeout: 90,
    capabilities: ['review', 'audit', 'security', 'lint'],
  },
  coordinator: {
    service: 'worker-coordinator',
    minReplicas: 1,
    maxReplicas: 2,
    priority: 'critical',
    idleTimeout: 300,
    capabilities: ['plan', 'orchestrate', 'delegate', 'merge'],
  },
};

// =============================================================================
// STATE MANAGEMENT
// =============================================================================
class SwarmState {
  constructor() {
    this.workers = new Map();
    this.tasks = new Map();
    this.metrics = {
      tasksCompleted: 0,
      tasksActive: 0,
      workersActive: 0,
      lastScaleEvent: null,
      uptime: Date.now(),
    };
    this.scaleHistory = [];
    this.redis = null;
  }

  async init() {
    // Connect to Redis
    this.redis = new Redis(CONFIG.redisUrl, {
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
    });

    await this.redis.connect();
    console.log('[STATE] Connected to Redis');

    // Load persisted state
    await this.loadState();
  }

  async loadState() {
    try {
      const stateFile = path.join(CONFIG.statePath, 'swarm-state.json');
      const data = await fs.readFile(stateFile, 'utf8');
      const state = JSON.parse(data);

      this.metrics = { ...this.metrics, ...state.metrics };
      this.scaleHistory = state.scaleHistory || [];

      console.log('[STATE] Loaded persisted state');
    } catch (err) {
      console.log('[STATE] No persisted state found, starting fresh');
    }
  }

  async saveState() {
    try {
      const stateFile = path.join(CONFIG.statePath, 'swarm-state.json');
      const state = {
        metrics: this.metrics,
        scaleHistory: this.scaleHistory.slice(-100), // Keep last 100 events
        workers: Array.from(this.workers.entries()),
        tasks: Array.from(this.tasks.entries()),
        timestamp: Date.now(),
      };

      await fs.writeFile(stateFile, JSON.stringify(state, null, 2));

      // Also persist to Redis
      await this.redis.set('swarm:state', JSON.stringify(state));

      console.log('[STATE] State persisted');
    } catch (err) {
      console.error('[STATE] Failed to persist state:', err.message);
    }
  }

  async setMemory(key, value, ttl = null) {
    const fullKey = `memory:${key}`;
    if (ttl) {
      await this.redis.setex(fullKey, ttl, JSON.stringify(value));
    } else {
      await this.redis.set(fullKey, JSON.stringify(value));
    }
  }

  async getMemory(key) {
    const fullKey = `memory:${key}`;
    const data = await this.redis.get(fullKey);
    return data ? JSON.parse(data) : null;
  }

  async getAllMemoryKeys(pattern = '*') {
    const keys = await this.redis.keys(`memory:${pattern}`);
    return keys.map(k => k.replace('memory:', ''));
  }
}

// =============================================================================
// AUTOSCALER
// =============================================================================
class AutoScaler {
  constructor(state) {
    this.state = state;
    this.lastScaleTime = new Map();
    this.pendingTasks = new Map();
  }

  parseRatio(ratio) {
    const [min, max] = ratio.split(':').map(Number);
    return { min, max };
  }

  async getServiceReplicas(serviceName) {
    try {
      const { stdout } = await execAsync(
        `docker service ls --filter name=${serviceName} --format "{{.Replicas}}"`
      );
      const [current, desired] = stdout.trim().split('/').map(Number);
      return { current, desired };
    } catch (err) {
      console.error(`[SCALE] Failed to get replicas for ${serviceName}:`, err.message);
      return { current: 0, desired: 0 };
    }
  }

  async scaleService(serviceName, replicas) {
    const lastScale = this.lastScaleTime.get(serviceName) || 0;
    const now = Date.now();

    // Enforce cooldown
    if (now - lastScale < CONFIG.scaleCooldown * 1000) {
      console.log(`[SCALE] Cooldown active for ${serviceName}, skipping`);
      return false;
    }

    try {
      await execAsync(`docker service scale ${serviceName}=${replicas}`);
      this.lastScaleTime.set(serviceName, now);

      this.state.scaleHistory.push({
        service: serviceName,
        replicas,
        timestamp: now,
        reason: replicas > 0 ? 'demand' : 'idle',
      });

      console.log(`[SCALE] Scaled ${serviceName} to ${replicas} replicas`);
      return true;
    } catch (err) {
      console.error(`[SCALE] Failed to scale ${serviceName}:`, err.message);
      return false;
    }
  }

  async evaluateScaling() {
    const ratio = this.parseRatio(CONFIG.workerRatio);

    for (const [type, config] of Object.entries(WORKER_TYPES)) {
      const { service, minReplicas, maxReplicas } = config;
      const { current } = await this.getServiceReplicas(service);

      // Get pending tasks for this worker type
      const pending = this.pendingTasks.get(type) || 0;
      const active = this.state.workers.get(type)?.active || 0;

      // Calculate desired replicas based on 1:2 ratio
      // 1 worker can handle up to 2 tasks
      const neededWorkers = Math.ceil(pending / ratio.max);
      const desired = Math.max(minReplicas, Math.min(maxReplicas, neededWorkers));

      if (desired !== current) {
        console.log(`[SCALE] ${type}: current=${current}, pending=${pending}, desired=${desired}`);
        await this.scaleService(service, desired);
      }
    }
  }

  addPendingTask(workerType) {
    const current = this.pendingTasks.get(workerType) || 0;
    this.pendingTasks.set(workerType, current + 1);
  }

  removePendingTask(workerType) {
    const current = this.pendingTasks.get(workerType) || 0;
    this.pendingTasks.set(workerType, Math.max(0, current - 1));
  }
}

// =============================================================================
// TASK ROUTER
// =============================================================================
class TaskRouter {
  constructor(state, scaler) {
    this.state = state;
    this.scaler = scaler;
    this.taskQueue = [];
  }

  findBestWorker(task) {
    const requiredCapabilities = task.capabilities || [];

    for (const [type, config] of Object.entries(WORKER_TYPES)) {
      const hasCapabilities = requiredCapabilities.every(
        cap => config.capabilities.includes(cap)
      );

      if (hasCapabilities) {
        return type;
      }
    }

    // Default to coordinator for unknown tasks
    return 'coordinator';
  }

  async routeTask(task) {
    const taskId = `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    const workerType = this.findBestWorker(task);

    // Add to pending tasks
    this.scaler.addPendingTask(workerType);

    // Store task in state
    this.state.tasks.set(taskId, {
      ...task,
      id: taskId,
      workerType,
      status: 'pending',
      createdAt: Date.now(),
    });

    // Publish to Redis for worker pickup
    await this.state.redis.lpush(`tasks:${workerType}`, JSON.stringify({
      id: taskId,
      ...task,
    }));

    // Trigger scaling evaluation
    await this.scaler.evaluateScaling();

    console.log(`[ROUTER] Task ${taskId} routed to ${workerType}`);

    return taskId;
  }

  async completeTask(taskId, result) {
    const task = this.state.tasks.get(taskId);
    if (!task) {
      throw new Error(`Task ${taskId} not found`);
    }

    // Update task status
    task.status = 'completed';
    task.completedAt = Date.now();
    task.result = result;

    // Remove from pending
    this.scaler.removePendingTask(task.workerType);

    // Store result in memory
    await this.state.setMemory(`task:${taskId}:result`, result);

    // Update metrics
    this.state.metrics.tasksCompleted++;

    console.log(`[ROUTER] Task ${taskId} completed`);

    return task;
  }

  async getTaskStatus(taskId) {
    return this.state.tasks.get(taskId);
  }
}

// =============================================================================
// HTTP API SERVER
// =============================================================================
class APIServer {
  constructor(state, router, scaler) {
    this.state = state;
    this.router = router;
    this.scaler = scaler;
  }

  async handleRequest(req, res) {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const method = req.method;

    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Content-Type', 'application/json');

    try {
      // Health check
      if (url.pathname === '/health' && method === 'GET') {
        return this.sendJson(res, 200, { status: 'healthy', uptime: Date.now() - this.state.metrics.uptime });
      }

      // Submit task
      if (url.pathname === '/task' && method === 'POST') {
        const body = await this.readBody(req);
        const task = JSON.parse(body);
        const taskId = await this.router.routeTask(task);
        return this.sendJson(res, 201, { taskId, status: 'pending' });
      }

      // Get task status
      if (url.pathname.startsWith('/task/') && method === 'GET') {
        const taskId = url.pathname.split('/')[2];
        const status = await this.router.getTaskStatus(taskId);
        if (!status) {
          return this.sendJson(res, 404, { error: 'Task not found' });
        }
        return this.sendJson(res, 200, status);
      }

      // Complete task
      if (url.pathname.startsWith('/task/') && method === 'PUT') {
        const taskId = url.pathname.split('/')[2];
        const body = await this.readBody(req);
        const result = JSON.parse(body);
        const task = await this.router.completeTask(taskId, result);
        return this.sendJson(res, 200, task);
      }

      // Get/Set memory
      if (url.pathname.startsWith('/memory/')) {
        const key = url.pathname.replace('/memory/', '');

        if (method === 'GET') {
          const value = await this.state.getMemory(key);
          return this.sendJson(res, value ? 200 : 404, value || { error: 'Not found' });
        }

        if (method === 'PUT') {
          const body = await this.readBody(req);
          const value = JSON.parse(body);
          await this.state.setMemory(key, value);
          return this.sendJson(res, 200, { success: true });
        }
      }

      // List memory keys
      if (url.pathname === '/memory' && method === 'GET') {
        const pattern = url.searchParams.get('pattern') || '*';
        const keys = await this.state.getAllMemoryKeys(pattern);
        return this.sendJson(res, 200, { keys });
      }

      // Scale worker
      if (url.pathname.startsWith('/scale/') && method === 'POST') {
        const workerType = url.pathname.split('/')[2];
        const body = await this.readBody(req);
        const { replicas } = JSON.parse(body);

        const config = WORKER_TYPES[workerType];
        if (!config) {
          return this.sendJson(res, 404, { error: 'Unknown worker type' });
        }

        await this.scaler.scaleService(config.service, replicas);
        return this.sendJson(res, 200, { success: true, replicas });
      }

      // Get swarm status
      if (url.pathname === '/status' && method === 'GET') {
        const status = {
          metrics: this.state.metrics,
          workers: Object.fromEntries(this.state.workers),
          tasks: {
            total: this.state.tasks.size,
            pending: Array.from(this.state.tasks.values()).filter(t => t.status === 'pending').length,
            completed: this.state.metrics.tasksCompleted,
          },
          scaleHistory: this.state.scaleHistory.slice(-10),
        };
        return this.sendJson(res, 200, status);
      }

      // Backup trigger
      if (url.pathname === '/backup' && method === 'POST') {
        await this.state.saveState();
        return this.sendJson(res, 200, { success: true, timestamp: Date.now() });
      }

      // 404
      return this.sendJson(res, 404, { error: 'Not found' });

    } catch (err) {
      console.error('[API] Error:', err.message);
      return this.sendJson(res, 500, { error: err.message });
    }
  }

  sendJson(res, status, data) {
    res.writeHead(status);
    res.end(JSON.stringify(data));
  }

  readBody(req) {
    return new Promise((resolve, reject) => {
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => resolve(body));
      req.on('error', reject);
    });
  }

  start() {
    const server = http.createServer((req, res) => this.handleRequest(req, res));
    server.listen(CONFIG.port, () => {
      console.log(`[API] Server listening on port ${CONFIG.port}`);
    });
    return server;
  }
}

// =============================================================================
// METRICS SERVER
// =============================================================================
class MetricsServer {
  constructor(state) {
    this.state = state;
  }

  generatePrometheusMetrics() {
    const metrics = [];

    metrics.push(`# HELP swarm_tasks_total Total tasks processed`);
    metrics.push(`# TYPE swarm_tasks_total counter`);
    metrics.push(`swarm_tasks_total ${this.state.metrics.tasksCompleted}`);

    metrics.push(`# HELP swarm_tasks_active Currently active tasks`);
    metrics.push(`# TYPE swarm_tasks_active gauge`);
    metrics.push(`swarm_tasks_active ${this.state.tasks.size}`);

    metrics.push(`# HELP swarm_uptime_seconds Orchestrator uptime`);
    metrics.push(`# TYPE swarm_uptime_seconds gauge`);
    metrics.push(`swarm_uptime_seconds ${(Date.now() - this.state.metrics.uptime) / 1000}`);

    for (const [type, data] of this.state.workers) {
      metrics.push(`swarm_workers_active{type="${type}"} ${data.active || 0}`);
    }

    return metrics.join('\n');
  }

  start() {
    const server = http.createServer((req, res) => {
      if (req.url === '/metrics') {
        res.writeHead(200, { 'Content-Type': 'text/plain' });
        res.end(this.generatePrometheusMetrics());
      } else {
        res.writeHead(404);
        res.end();
      }
    });

    server.listen(CONFIG.metricsPort, () => {
      console.log(`[METRICS] Server listening on port ${CONFIG.metricsPort}`);
    });

    return server;
  }
}

// =============================================================================
// MAIN
// =============================================================================
async function main() {
  console.log('='.repeat(60));
  console.log('RUVECTOR SWARM ORCHESTRATOR');
  console.log('='.repeat(60));
  console.log(`Worker Ratio: ${CONFIG.workerRatio}`);
  console.log(`Min Workers: ${CONFIG.minWorkers}`);
  console.log(`Max Workers: ${CONFIG.maxWorkers}`);
  console.log('='.repeat(60));

  // Initialize state
  const state = new SwarmState();
  await state.init();

  // Initialize scaler and router
  const scaler = new AutoScaler(state);
  const router = new TaskRouter(state, scaler);

  // Start API server
  const api = new APIServer(state, router, scaler);
  api.start();

  // Start metrics server
  const metrics = new MetricsServer(state);
  metrics.start();

  // Periodic state persistence
  setInterval(() => state.saveState(), 60000);

  // Periodic scaling evaluation
  setInterval(() => scaler.evaluateScaling(), 30000);

  // Graceful shutdown
  const shutdown = async (signal) => {
    console.log(`\n[MAIN] Received ${signal}, shutting down...`);
    await state.saveState();
    process.exit(0);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  console.log('[MAIN] Orchestrator ready');
}

main().catch(err => {
  console.error('[MAIN] Fatal error:', err);
  process.exit(1);
});
