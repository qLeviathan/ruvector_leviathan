/**
 * =============================================================================
 * RUVECTOR SWARM WORKER
 * =============================================================================
 * Generic worker runtime that:
 * - Polls for tasks from Redis queue
 * - Executes tasks based on worker type
 * - Reports results back to orchestrator
 * - Self-terminates after idle timeout
 * =============================================================================
 */

const http = require('http');
const Redis = require('ioredis');
const os = require('os');

// =============================================================================
// CONFIGURATION
// =============================================================================
const CONFIG = {
  workerType: process.env.WORKER_TYPE || 'generic',
  capabilities: (process.env.WORKER_CAPABILITIES || '').split(','),
  priority: process.env.SPAWN_PRIORITY || 'medium',
  idleTimeout: parseInt(process.env.IDLE_TIMEOUT || '120'),
  maxConcurrentTasks: parseInt(process.env.MAX_CONCURRENT_TASKS || '2'),
  redisUrl: process.env.MEMORY_REDIS_URL || 'redis://memory-store:6379',
  orchestratorUrl: process.env.ORCHESTRATOR_URL || 'http://orchestrator:9000',
  healthPort: parseInt(process.env.HEALTH_PORT || '8080'),
};

// =============================================================================
// WORKER STATE
// =============================================================================
class WorkerState {
  constructor() {
    this.id = `${CONFIG.workerType}-${os.hostname()}-${Date.now()}`;
    this.redis = null;
    this.activeTasks = new Map();
    this.completedTasks = 0;
    this.lastActivity = Date.now();
    this.running = true;
  }

  async init() {
    this.redis = new Redis(CONFIG.redisUrl, {
      retryDelayOnFailover: 100,
      maxRetriesPerRequest: 3,
      lazyConnect: true,
    });

    await this.redis.connect();
    console.log(`[WORKER:${this.id}] Connected to Redis`);

    // Register with orchestrator
    await this.register();
  }

  async register() {
    const info = {
      id: this.id,
      type: CONFIG.workerType,
      capabilities: CONFIG.capabilities,
      priority: CONFIG.priority,
      hostname: os.hostname(),
      startedAt: Date.now(),
    };

    await this.redis.hset('workers:active', this.id, JSON.stringify(info));
    console.log(`[WORKER:${this.id}] Registered with swarm`);
  }

  async deregister() {
    await this.redis.hdel('workers:active', this.id);
    console.log(`[WORKER:${this.id}] Deregistered from swarm`);
  }

  updateActivity() {
    this.lastActivity = Date.now();
  }

  isIdle() {
    return (Date.now() - this.lastActivity) > (CONFIG.idleTimeout * 1000);
  }
}

// =============================================================================
// TASK EXECUTOR
// =============================================================================
class TaskExecutor {
  constructor(state) {
    this.state = state;
  }

  async execute(task) {
    console.log(`[EXECUTOR] Starting task ${task.id}: ${task.description || 'no description'}`);

    const startTime = Date.now();

    try {
      // Route to appropriate handler based on capability
      let result;

      switch (CONFIG.workerType) {
        case 'researcher':
          result = await this.executeResearch(task);
          break;
        case 'coder':
          result = await this.executeCode(task);
          break;
        case 'tester':
          result = await this.executeTest(task);
          break;
        case 'reviewer':
          result = await this.executeReview(task);
          break;
        case 'coordinator':
          result = await this.executeCoordination(task);
          break;
        default:
          result = await this.executeGeneric(task);
      }

      const duration = Date.now() - startTime;

      return {
        success: true,
        result,
        duration,
        workerId: this.state.id,
      };

    } catch (err) {
      console.error(`[EXECUTOR] Task ${task.id} failed:`, err.message);

      return {
        success: false,
        error: err.message,
        duration: Date.now() - startTime,
        workerId: this.state.id,
      };
    }
  }

  async executeResearch(task) {
    // Research-specific logic
    const { action, target, query } = task;

    switch (action) {
      case 'explore':
        return this.exploreCodebase(target, query);
      case 'analyze':
        return this.analyzePattern(target, query);
      case 'search':
        return this.searchContent(target, query);
      default:
        return { action, status: 'completed' };
    }
  }

  async executeCode(task) {
    // Coding-specific logic
    const { action, file, content, changes } = task;

    switch (action) {
      case 'write':
        return this.writeFile(file, content);
      case 'edit':
        return this.editFile(file, changes);
      case 'refactor':
        return this.refactorCode(file, changes);
      default:
        return { action, status: 'completed' };
    }
  }

  async executeTest(task) {
    // Testing-specific logic
    const { action, target, config } = task;

    switch (action) {
      case 'test':
        return this.runTests(target, config);
      case 'benchmark':
        return this.runBenchmark(target, config);
      case 'coverage':
        return this.getCoverage(target);
      default:
        return { action, status: 'completed' };
    }
  }

  async executeReview(task) {
    // Review-specific logic
    const { action, target, rules } = task;

    switch (action) {
      case 'review':
        return this.reviewCode(target, rules);
      case 'audit':
        return this.securityAudit(target);
      case 'lint':
        return this.lintCode(target, rules);
      default:
        return { action, status: 'completed' };
    }
  }

  async executeCoordination(task) {
    // Coordination-specific logic
    const { action, subtasks, strategy } = task;

    switch (action) {
      case 'plan':
        return this.createPlan(subtasks);
      case 'delegate':
        return this.delegateTasks(subtasks);
      case 'merge':
        return this.mergeResults(subtasks);
      default:
        return { action, status: 'completed' };
    }
  }

  async executeGeneric(task) {
    return { task, status: 'completed' };
  }

  // Placeholder implementations - would be filled with actual logic
  async exploreCodebase(target, query) {
    return { explored: target, matches: [] };
  }

  async analyzePattern(target, query) {
    return { analyzed: target, patterns: [] };
  }

  async searchContent(target, query) {
    return { searched: target, results: [] };
  }

  async writeFile(file, content) {
    return { written: file, bytes: content?.length || 0 };
  }

  async editFile(file, changes) {
    return { edited: file, changes: changes?.length || 0 };
  }

  async refactorCode(file, changes) {
    return { refactored: file };
  }

  async runTests(target, config) {
    return { tested: target, passed: true };
  }

  async runBenchmark(target, config) {
    return { benchmarked: target, results: {} };
  }

  async getCoverage(target) {
    return { target, coverage: 0 };
  }

  async reviewCode(target, rules) {
    return { reviewed: target, issues: [] };
  }

  async securityAudit(target) {
    return { audited: target, vulnerabilities: [] };
  }

  async lintCode(target, rules) {
    return { linted: target, errors: 0, warnings: 0 };
  }

  async createPlan(subtasks) {
    return { planned: subtasks?.length || 0 };
  }

  async delegateTasks(subtasks) {
    return { delegated: subtasks?.length || 0 };
  }

  async mergeResults(subtasks) {
    return { merged: subtasks?.length || 0 };
  }
}

// =============================================================================
// TASK POLLER
// =============================================================================
class TaskPoller {
  constructor(state, executor) {
    this.state = state;
    this.executor = executor;
  }

  async poll() {
    const queueKey = `tasks:${CONFIG.workerType}`;

    while (this.state.running) {
      try {
        // Check if we can take more tasks
        if (this.state.activeTasks.size >= CONFIG.maxConcurrentTasks) {
          await this.sleep(1000);
          continue;
        }

        // Block-pop from queue (5 second timeout)
        const result = await this.state.redis.brpop(queueKey, 5);

        if (result) {
          const [, taskData] = result;
          const task = JSON.parse(taskData);

          this.state.updateActivity();
          this.processTask(task); // Don't await - process concurrently
        }

        // Check idle timeout
        if (this.state.isIdle() && this.state.activeTasks.size === 0) {
          console.log(`[POLLER] Idle timeout reached, signaling shutdown`);
          this.state.running = false;
        }

      } catch (err) {
        console.error('[POLLER] Error:', err.message);
        await this.sleep(1000);
      }
    }
  }

  async processTask(task) {
    this.state.activeTasks.set(task.id, task);

    try {
      const result = await this.executor.execute(task);

      // Report result to orchestrator
      await this.reportResult(task.id, result);

      this.state.completedTasks++;
      this.state.updateActivity();

    } catch (err) {
      console.error(`[POLLER] Failed to process task ${task.id}:`, err.message);
    } finally {
      this.state.activeTasks.delete(task.id);
    }
  }

  async reportResult(taskId, result) {
    try {
      // Store in Redis
      await this.state.redis.set(
        `task:${taskId}:result`,
        JSON.stringify(result),
        'EX',
        3600 // 1 hour TTL
      );

      // Notify orchestrator
      const response = await fetch(`${CONFIG.orchestratorUrl}/task/${taskId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(result),
      });

      if (!response.ok) {
        console.warn(`[POLLER] Failed to notify orchestrator: ${response.status}`);
      }

    } catch (err) {
      console.error(`[POLLER] Failed to report result:`, err.message);
    }
  }

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// =============================================================================
// HEALTH SERVER
// =============================================================================
class HealthServer {
  constructor(state) {
    this.state = state;
  }

  start() {
    const server = http.createServer((req, res) => {
      if (req.url === '/health') {
        const health = {
          status: this.state.running ? 'healthy' : 'shutting_down',
          workerId: this.state.id,
          workerType: CONFIG.workerType,
          activeTasks: this.state.activeTasks.size,
          completedTasks: this.state.completedTasks,
          idleSeconds: Math.floor((Date.now() - this.state.lastActivity) / 1000),
          uptime: process.uptime(),
        };

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(health));
      } else if (req.url === '/ready') {
        res.writeHead(this.state.running ? 200 : 503);
        res.end();
      } else {
        res.writeHead(404);
        res.end();
      }
    });

    server.listen(CONFIG.healthPort, () => {
      console.log(`[HEALTH] Server listening on port ${CONFIG.healthPort}`);
    });

    return server;
  }
}

// =============================================================================
// MEMORY COORDINATION
// =============================================================================
class MemoryCoordinator {
  constructor(state) {
    this.state = state;
  }

  async get(key) {
    const data = await this.state.redis.get(`memory:${key}`);
    return data ? JSON.parse(data) : null;
  }

  async set(key, value, ttl = null) {
    const fullKey = `memory:${key}`;
    if (ttl) {
      await this.state.redis.setex(fullKey, ttl, JSON.stringify(value));
    } else {
      await this.state.redis.set(fullKey, JSON.stringify(value));
    }
  }

  async append(key, value) {
    const existing = await this.get(key) || [];
    existing.push(value);
    await this.set(key, existing);
  }

  async getSwarmState() {
    const workers = await this.state.redis.hgetall('workers:active');
    return Object.fromEntries(
      Object.entries(workers).map(([k, v]) => [k, JSON.parse(v)])
    );
  }
}

// =============================================================================
// MAIN
// =============================================================================
async function main() {
  console.log('='.repeat(60));
  console.log(`RUVECTOR SWARM WORKER - ${CONFIG.workerType.toUpperCase()}`);
  console.log('='.repeat(60));
  console.log(`Capabilities: ${CONFIG.capabilities.join(', ')}`);
  console.log(`Priority: ${CONFIG.priority}`);
  console.log(`Idle Timeout: ${CONFIG.idleTimeout}s`);
  console.log(`Max Concurrent Tasks: ${CONFIG.maxConcurrentTasks}`);
  console.log('='.repeat(60));

  // Initialize state
  const state = new WorkerState();
  await state.init();

  // Initialize components
  const executor = new TaskExecutor(state);
  const poller = new TaskPoller(state, executor);
  const health = new HealthServer(state);
  const memory = new MemoryCoordinator(state);

  // Attach memory to state for use in executor
  state.memory = memory;

  // Start health server
  health.start();

  // Graceful shutdown
  const shutdown = async (signal) => {
    console.log(`\n[MAIN] Received ${signal}, shutting down...`);
    state.running = false;

    // Wait for active tasks to complete (max 30 seconds)
    const timeout = Date.now() + 30000;
    while (state.activeTasks.size > 0 && Date.now() < timeout) {
      await new Promise(r => setTimeout(r, 1000));
    }

    await state.deregister();
    await state.redis.quit();
    process.exit(0);
  };

  process.on('SIGTERM', () => shutdown('SIGTERM'));
  process.on('SIGINT', () => shutdown('SIGINT'));

  // Start polling for tasks
  console.log(`[MAIN] Worker ${state.id} ready, polling for tasks...`);
  await poller.poll();

  // Clean shutdown after idle timeout
  await shutdown('IDLE_TIMEOUT');
}

main().catch(err => {
  console.error('[MAIN] Fatal error:', err);
  process.exit(1);
});
