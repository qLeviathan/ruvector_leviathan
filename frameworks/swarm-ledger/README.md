# 🌊 Swarm Ledger

> **Deterministic Swarm Hosting Framework for Fractal Information Economy**

A revolutionary distributed development platform combining:
- 🐙 **GitHub's collaboration** - Version control, pull requests, code review
- 🎮 **Game forums' dynamics** - Discussion, reputation, achievements
- ₿ **Bitcoin's ledger** - Immutable transactions, cryptographic verification

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue)](https://www.typescriptlang.org/)
[![RuVector](https://img.shields.io/badge/RuVector-Powered-purple)](https://github.com/ruvnet/ruvector)

---

## 🚀 What is Swarm Ledger?

**Swarm Ledger** creates a **deterministic, auditable, fractal information economy** where:

- ✅ Every development action is **cryptographically signed** and **immutably stored**
- ✅ Tasks **self-organize** into fractal hierarchies (infinitely decomposable)
- ✅ Contributors earn **reputation** and **credits** through validated work
- ✅ Anyone can deploy a **reverse swarm** from their device via API

### Key Innovation: Reverse Swarm Deployment

**Traditional**: Deploy code TO infrastructure
**Swarm Ledger**: Deploy infrastructure FROM your device

```typescript
// Deploy a swarm instance from your phone, laptop, or Raspberry Pi
const swarm = new ReverseSwarm();
const instance = await swarm.deploy(apiKey, {
  topology: 'mesh',
  maxAgents: 5,
  taskMarket: { acceptTypes: ['code-review', 'testing'] }
});

// Your swarm now participates in the global task economy
console.log(`Swarm deployed: ${instance.publicEndpoint}`);
```

---

## 🎯 Core Principles

### 1. Deterministic Execution
Every action produces the same result given the same inputs. All operations are reproducible and verifiable.

### 2. Fractal Task Decomposition
Tasks self-organize into hierarchical structures:

```
Epic Task (10,000 credits)
├── Feature A (3,000 credits)
│   ├── Subtask A.1 (1,000 credits)
│   │   ├── Micro-task A.1.a (250 credits)
│   │   └── Micro-task A.1.b (250 credits)
│   └── Subtask A.2 (2,000 credits)
└── Feature B (7,000 credits)
```

### 3. Cryptographic Audit Trail
Every transaction is:
- 🔐 Cryptographically signed (Ed25519)
- 🕒 Timestamped with block height
- 🌳 Merkle-tree verified
- 💾 Immutably stored

### 4. Economic Incentive Alignment
Contributors earn through:
- 💻 Code contributions (validated by tests)
- 👁️ Code reviews (validated by consensus)
- 🐛 Bug reports (validated by fixes)
- 📝 Documentation (validated by usage)

---

## 📦 Installation

```bash
npm install @ruvector/swarm-ledger
```

Or try instantly:

```bash
npx @ruvector/swarm-ledger deploy-swarm
```

---

## 🏁 Quick Start

### 1. Deploy Your First Swarm

```typescript
import { ReverseSwarm, generateKeyPair } from '@ruvector/swarm-ledger';

// Generate cryptographic identity
const identity = await generateKeyPair();

// Deploy swarm from your device
const swarm = new ReverseSwarm();
const instance = await swarm.deploy(
  process.env.SWARM_API_KEY,
  {
    topology: 'mesh',
    maxAgents: 5,
    ledgerEndpoint: 'wss://ledger.swarm-network.io',
    nodeIdentity: identity,

    // Task market preferences
    taskMarket: {
      acceptTypes: ['code-review', 'testing', 'documentation'],
      minValue: 100,  // Minimum credits
      maxConcurrent: 3,
    },
  }
);

console.log(`✅ Swarm deployed at: ${instance.publicEndpoint}`);
console.log(`📜 Ledger transaction: ${instance.deploymentTxHash}`);
```

### 2. Create a Fractal Task

```typescript
import { FractalTask, Ledger } from '@ruvector/swarm-ledger';

const ledger = new Ledger({ endpoint: 'wss://ledger.swarm-network.io' });

// Create epic task
const epic = await ledger.submitTransaction({
  type: 'TaskCreate',
  data: {
    title: 'Build authentication system',
    description: 'OAuth2, JWT, session management',
    value: BigInt(10000),  // 10,000 credits
    proofType: { type: 'TestResults', config: { framework: 'jest', coverage: 0.8 } },
  },
});

// Decompose into subtasks
await ledger.submitTransaction({
  type: 'TaskDecompose',
  data: {
    parent: epic.id,
    children: [
      { title: 'OAuth2 provider integration', value: BigInt(3000) },
      { title: 'JWT token management', value: BigInt(2000) },
      { title: 'Session storage (Redis)', value: BigInt(2000) },
      { title: 'Security tests', value: BigInt(2000) },
      { title: 'Documentation', value: BigInt(1000) },
    ],
  },
});
```

### 3. Claim and Complete a Task

```typescript
// Claim task with stake
await ledger.submitTransaction({
  type: 'TaskClaim',
  data: {
    task: taskHash,
    stake: BigInt(200),  // Stake credits (refunded on success)
  },
});

// Complete task with proof
await ledger.submitTransaction({
  type: 'TaskComplete',
  data: {
    task: taskHash,
    proof: {
      type: 'TestResults',
      data: testResults,
      timestamp: Date.now(),
    },
  },
});

// Validators approve → Credits distributed automatically
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│         LAYER 5: USER INTERFACE              │
│   Web App | Mobile App | CLI | API Client   │
└─────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│      LAYER 4: COMMUNITY & FORUM              │
│   Discussions | Reputation | Achievements   │
└─────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│         LAYER 3: TASK ECONOMY                │
│   Fractal Tasks | Credits | Marketplace     │
└─────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│      LAYER 2: SWARM ORCHESTRATION            │
│   Mesh | Hierarchical | Ring | Adaptive     │
│   (Powered by RuVector swarm capabilities)  │
└─────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────┐
│          LAYER 1: LEDGER CORE                │
│   Transaction Log | Raft Consensus          │
│   Merkle Tree | Ed25519 Signatures          │
└─────────────────────────────────────────────┘
```

---

## 📊 Transaction Types

Swarm Ledger supports 17 transaction types across 5 categories:

### Code Operations
- `Commit` - Create/modify/delete files
- `Review` - Approve/request changes/reject
- `Merge` - Merge branches with various strategies

### Task Operations
- `TaskCreate` - Create new task with economic value
- `TaskDecompose` - Break down into subtasks (fractal)
- `TaskClaim` - Claim task with stake
- `TaskComplete` - Submit proof of completion
- `TaskValidate` - Validate completion proof

### Forum Operations
- `Post` - Create discussion posts
- `Vote` - Upvote/downvote content
- `Badge` - Award achievements

### Swarm Operations
- `SwarmDeploy` - Deploy reverse swarm instance
- `AgentSpawn` - Spawn agent in swarm
- `AgentComplete` - Agent task completion

### Economic Operations
- `Transfer` - Transfer credits
- `Stake` - Stake credits on task
- `Reward` - Distribute rewards

---

## 💰 Economic Model

### Credit System

**Credits** are the internal currency for task valuation and incentives.

#### Credit Sources (How to Earn)
- ✅ Complete tasks → 60% of task value
- ✅ Review code → 20% of task value (split among reviewers)
- ✅ Validate work → 10% of task value (split among validators)
- ✅ Report bugs → Variable based on severity
- ✅ Run swarm agents → Proportional to compute time

#### Credit Sinks (How Credits Are Used)
- 📤 Post new tasks → 1% fee
- ⚡ Priority boost → Variable market price
- 🚀 Deploy swarms → Based on resources

#### Economic Conservation

The system enforces **economic conservation**:

```typescript
// Parent task must equal sum of children
parentTask.value === sum(childTasks.map(t => t.value))

// Distribution must sum to 1.0
split.implementation + split.review + split.validation + split.pool === 1.0
```

---

## 🎖️ Reputation System

### Reputation Score

Your reputation is calculated from multiple factors:

```typescript
interface ReputationScore {
  total: number;
  breakdown: {
    codeContributions: number;    // From merged commits
    codeReviews: number;          // From accepted reviews
    taskCompletion: number;       // From validated tasks
    communityEngagement: number;  // From posts, votes
    swarmOperations: number;      // From agent executions
  };
}
```

### Decay Mechanism

Reputation decays over time to encourage continued participation:

```typescript
decayRate: 0.01  // 1% per week of inactivity
```

### Badges & Achievements

Earn badges by meeting criteria:

| Badge | Requirement |
|-------|------------|
| 🎯 **First Commit** | Make your first code contribution |
| 💯 **Hundred Commits** | Make 100 code contributions |
| 👨‍🏫 **Code Mentor** | Get 100+ helpful reviews |
| 🎯 **Task Completer** | Complete 50+ tasks |
| 🧩 **Task Decomposer** | Excellent task breakdown (voted) |
| 🚀 **Swarm Master** | Deploy 10+ swarms |
| 🏆 **Veteran** | 1 year active |

---

## 🔒 Security

### Cryptographic Primitives

- **Identity**: Ed25519 (Curve25519 + Edwards curve)
- **Hashing**: SHA-256 for transaction IDs, Merkle trees
- **Signatures**: Ed25519 (~50,000 verifications/sec)

### Attack Mitigation

| Attack | Mitigation |
|--------|-----------|
| Sybil | Proof-of-stake for account creation |
| Spam | Rate limiting + reputation requirements |
| 51% Attack | Require >66% consensus for critical ops |
| Double-spending | UTXO-style credit tracking with Raft |
| Replay | Monotonic nonce per account |
| Malicious code | Sandboxed execution (Firecracker VMs) |
| Data tampering | Merkle tree verification |

---

## 🛠️ CLI Usage

```bash
# Deploy swarm
swarm-ledger deploy-swarm --topology mesh --max-agents 5

# Create task
swarm-ledger task create \
  --title "Fix auth bug" \
  --value 500 \
  --proof-type test

# Query ledger
swarm-ledger ledger query --tx-hash abc123...

# Check reputation
swarm-ledger reputation check --pubkey def456...

# Audit trail
swarm-ledger audit --from abc123 --to def456
```

---

## 📚 Documentation

- 📖 [Specification](./docs/SPECIFICATION.md) - Complete feature specification
- 🏗️ [Architecture](./docs/ARCHITECTURE.md) - Technical deep dive
- 📘 [API Reference](./docs/API.md) - TypeScript/REST API docs
- 🎓 [Examples](./examples/) - Code examples and tutorials

---

## 🗺️ Roadmap

### Phase 1: Foundation (Months 1-3)
- [x] Ledger core implementation
- [x] Transaction types
- [ ] Raft consensus integration
- [ ] CLI tools

### Phase 2: Task Economy (Months 4-6)
- [ ] Fractal task system
- [ ] Credit tracking
- [ ] Task marketplace
- [ ] Agent-task matching

### Phase 3: Reverse Swarm (Months 7-9)
- [ ] Reverse deployment API
- [ ] Device-to-swarm protocols
- [ ] Global network discovery
- [ ] Load balancing

### Phase 4: Community (Months 10-12)
- [ ] Forum implementation
- [ ] Reputation system
- [ ] Badge/achievement system
- [ ] Governance proposals

### Phase 5: Production (Months 13-15)
- [ ] Security audits
- [ ] Performance optimization
- [ ] Documentation
- [ ] Public beta launch

---

## 📈 Success Metrics

### Network Growth
- 🎯 1,000+ active swarm nodes
- 🎯 10,000+ deployed swarms
- 🎯 100,000+ transactions/day

### Economic Activity
- 💰 1M+ credits in circulation
- ✅ 500+ tasks completed daily
- 📊 90%+ task completion rate

### Community Engagement
- 👥 5,000+ registered users
- 💬 100+ forum posts/day
- ⭐ Average reputation: 250

### Performance
- ⚡ Transaction finality: <5 seconds
- 🔍 Audit verification: <100ms
- 🚀 Swarm deployment: <30 seconds

---

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md).

```bash
# Clone repository
git clone https://github.com/qLeviathan/ruvector_leviathan.git
cd frameworks/swarm-ledger

# Install dependencies
npm install

# Run tests
npm test

# Build
npm run build
```

---

## 📜 License

**MIT License** - see [LICENSE](./LICENSE) for details.

Free for commercial and personal use.

---

## 🙏 Acknowledgments

Built on top of:
- [RuVector](https://github.com/ruvnet/ruvector) - Distributed vector database with swarm capabilities
- [Raft Consensus](https://raft.github.io/) - Distributed consensus algorithm
- [@noble/ed25519](https://github.com/paulmillr/noble-ed25519) - Ed25519 signatures

---

<div align="center">

**Built by [rUv](https://ruv.io)**

[Documentation](./docs/) | [Examples](./examples/) | [GitHub](https://github.com/qLeviathan/ruvector_leviathan)

*Creating a deterministic, auditable future for collaborative development* 🌊

</div>
