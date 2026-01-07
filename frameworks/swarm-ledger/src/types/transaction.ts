/**
 * Core transaction types for Swarm Ledger
 * Provides Bitcoin-like immutability for all development operations
 */

import { Hash, PublicKey, Signature, Credits } from './primitives';

/**
 * Base transaction structure
 * Every action in the system is represented as a signed, timestamped transaction
 */
export interface Transaction {
  /** SHA-256 hash of transaction content */
  id: Hash;

  /** Unix timestamp in milliseconds */
  timestamp: number;

  /** Ed25519 public key of transaction author */
  author: PublicKey;

  /** Ed25519 signature over transaction content */
  signature: Signature;

  /** Type of transaction */
  txType: TxType;

  /** Serialized transaction payload */
  payload: Uint8Array;

  /** Hash of previous transaction (blockchain-style) */
  prevHash: Hash;

  /** Monotonic nonce for replay protection */
  nonce: bigint;
}

/**
 * Transaction types supported by Swarm Ledger
 */
export type TxType =
  // Code operations
  | { type: 'Commit'; data: CommitTx }
  | { type: 'Review'; data: ReviewTx }
  | { type: 'Merge'; data: MergeTx }

  // Task operations
  | { type: 'TaskCreate'; data: TaskCreateTx }
  | { type: 'TaskDecompose'; data: TaskDecomposeTx }
  | { type: 'TaskClaim'; data: TaskClaimTx }
  | { type: 'TaskComplete'; data: TaskCompleteTx }
  | { type: 'TaskValidate'; data: TaskValidateTx }

  // Forum operations
  | { type: 'Post'; data: PostTx }
  | { type: 'Vote'; data: VoteTx }
  | { type: 'Badge'; data: BadgeTx }

  // Swarm operations
  | { type: 'SwarmDeploy'; data: SwarmDeployTx }
  | { type: 'AgentSpawn'; data: AgentSpawnTx }
  | { type: 'AgentComplete'; data: AgentCompleteTx }

  // Economic operations
  | { type: 'Transfer'; data: TransferTx }
  | { type: 'Stake'; data: StakeTx }
  | { type: 'Reward'; data: RewardTx };

// =============================================================================
// Code Operations
// =============================================================================

export interface CommitTx {
  files: FileDelta[];
  message: string;
  branch?: string;
}

export interface FileDelta {
  path: string;
  operation: 'create' | 'modify' | 'delete';
  contentHash?: Hash;  // SHA-256 of file content
  diff?: string;       // Unified diff format
}

export interface ReviewTx {
  target: Hash;        // Transaction being reviewed
  verdict: Verdict;
  comments: Comment[];
}

export type Verdict = 'approve' | 'request_changes' | 'reject';

export interface Comment {
  file?: string;
  line?: number;
  content: string;
}

export interface MergeTx {
  source: Hash;        // Source commit
  target: Hash;        // Target branch/commit
  strategy: 'fast-forward' | 'merge-commit' | 'squash' | 'rebase';
}

// =============================================================================
// Task Operations
// =============================================================================

export interface TaskCreateTx {
  title: string;
  description: string;
  value: Credits;
  proofType: ProofType;
  tags?: string[];
  deadline?: number;  // Unix timestamp
}

export interface TaskDecomposeTx {
  parent: Hash;
  children: ChildTaskSpec[];
}

export interface ChildTaskSpec {
  title: string;
  description: string;
  value: Credits;
  economicSplit?: EconomicSplit;
}

export interface TaskClaimTx {
  task: Hash;
  stake: Credits;      // Credits staked to claim task
  estimatedCompletion?: number;  // Unix timestamp
}

export interface TaskCompleteTx {
  task: Hash;
  proof: Proof;
}

export interface TaskValidateTx {
  task: Hash;
  verdict: boolean;
  reason?: string;
}

// =============================================================================
// Forum Operations
// =============================================================================

export interface PostTx {
  thread: Hash;
  content: string;
  attachments?: string[];  // IPFS hashes
}

export interface VoteTx {
  target: Hash;        // Post or comment being voted on
  direction: 1 | -1;   // Upvote or downvote
}

export interface BadgeTx {
  recipient: PublicKey;
  badgeType: string;
  reason: string;
}

// =============================================================================
// Swarm Operations
// =============================================================================

export interface SwarmDeployTx {
  config: SwarmConfig;
  endpoint: string;    // Public endpoint of deployed swarm
  resources?: ResourceSpec;
}

export interface SwarmConfig {
  topology: 'mesh' | 'hierarchical' | 'ring' | 'star' | 'adaptive';
  maxAgents: number;
  consensusMode: 'raft' | 'byzantine' | 'gossip';
  taskMarket?: TaskMarketConfig;
  reputation?: ReputationConfig;
}

export interface ResourceSpec {
  cpu: number;         // CPU cores
  memory: number;      // MB
  storage: number;     // GB
  bandwidth: number;   // Mbps
}

export interface TaskMarketConfig {
  acceptTypes: string[];
  minValue: Credits;
  maxConcurrent: number;
}

export interface ReputationConfig {
  minScoreToValidate: number;
  specializations: string[];
}

export interface AgentSpawnTx {
  swarm: Hash;
  agentType: string;
  capabilities: string[];
  config?: Record<string, unknown>;
}

export interface AgentCompleteTx {
  agent: Hash;
  task: Hash;
  result: Uint8Array;  // Serialized result
  metrics?: AgentMetrics;
}

export interface AgentMetrics {
  duration: number;    // Milliseconds
  tokensUsed?: number;
  memoryPeak?: number; // MB
}

// =============================================================================
// Economic Operations
// =============================================================================

export interface TransferTx {
  from: PublicKey;
  to: PublicKey;
  amount: Credits;
  memo?: string;
}

export interface StakeTx {
  task: Hash;
  amount: Credits;
  duration?: number;   // Staking period in seconds
}

export interface RewardTx {
  recipient: PublicKey;
  amount: Credits;
  reason: string;
  source: 'task_completion' | 'validation' | 'review' | 'community_pool';
}

// =============================================================================
// Supporting Types
// =============================================================================

export interface EconomicSplit {
  implementation: number;  // 0.0-1.0
  review: number;
  validation: number;
  pool: number;
}

export type ProofType =
  | { type: 'TestResults'; config: TestConfig }
  | { type: 'BenchmarkResults'; metrics: string[] }
  | { type: 'StaticAnalysis'; tools: string[] }
  | { type: 'ConsensusVote'; minValidators: number; threshold: number }
  | { type: 'PeerReview'; minReviews: number }
  | { type: 'AIValidation'; model: string; threshold: number };

export interface TestConfig {
  framework: string;
  coverage?: number;   // Minimum code coverage (0.0-1.0)
  timeout?: number;    // Milliseconds
  environment?: Record<string, string>;
}

export interface Proof {
  type: string;
  data: Uint8Array;
  timestamp: number;
  verifier?: PublicKey;
}

/**
 * Serialize transaction for signing
 * Produces deterministic canonical representation
 */
export function serializeForSigning(tx: Omit<Transaction, 'signature'>): Uint8Array {
  // Canonical JSON serialization (sorted keys, no whitespace)
  const canonical = JSON.stringify({
    id: tx.id,
    timestamp: tx.timestamp,
    author: tx.author,
    txType: tx.txType,
    payload: Array.from(tx.payload),
    prevHash: tx.prevHash,
    nonce: tx.nonce.toString(),
  }, Object.keys({
    id: null,
    timestamp: null,
    author: null,
    txType: null,
    payload: null,
    prevHash: null,
    nonce: null,
  }));

  return new TextEncoder().encode(canonical);
}
