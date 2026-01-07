/**
 * Primitive cryptographic and economic types for Swarm Ledger
 */

/**
 * SHA-256 hash (32 bytes / 64 hex characters)
 */
export type Hash = string;

/**
 * Ed25519 public key (32 bytes)
 */
export type PublicKey = Uint8Array;

/**
 * Ed25519 signature (64 bytes)
 */
export type Signature = Uint8Array;

/**
 * Internal credit currency (integer, no fractional units)
 */
export type Credits = bigint;

/**
 * Task status in lifecycle
 */
export type TaskStatus =
  | 'open'
  | 'claimed'
  | 'in_progress'
  | 'pending_validation'
  | 'validated'
  | 'rejected'
  | 'cancelled';

/**
 * Node state in Raft consensus
 */
export type NodeState = 'follower' | 'candidate' | 'leader' | 'learner';

/**
 * Swarm topology pattern
 */
export type SwarmTopology = 'mesh' | 'hierarchical' | 'ring' | 'star' | 'adaptive';

/**
 * Consensus mode for transaction ordering
 */
export type ConsensusMode = 'raft' | 'byzantine' | 'gossip';

/**
 * Agent type in swarm
 */
export type AgentType =
  // Development agents
  | 'coder'
  | 'reviewer'
  | 'tester'
  | 'architect'

  // Task agents
  | 'task_decomposer'
  | 'task_validator'
  | 'task_executor'

  // Economic agents
  | 'market_maker'
  | 'reputation_oracle'

  // Infrastructure agents
  | 'storage_provider'
  | 'compute_provider'

  // Community agents
  | 'moderator'
  | 'content_curator';

/**
 * Badge rarity levels
 */
export type BadgeRarity = 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';

/**
 * Reputation breakdown by category
 */
export interface ReputationBreakdown {
  codeContributions: number;
  codeReviews: number;
  taskCompletion: number;
  communityEngagement: number;
  swarmOperations: number;
}

/**
 * Keypair for signing transactions
 */
export interface KeyPair {
  publicKey: PublicKey;
  privateKey: Uint8Array;
}

/**
 * Convert bytes to hex string
 */
export function bytesToHex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map(b => b.toString(16).padStart(2, '0'))
    .join('');
}

/**
 * Convert hex string to bytes
 */
export function hexToBytes(hex: string): Uint8Array {
  if (hex.length % 2 !== 0) {
    throw new Error('Invalid hex string');
  }
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < hex.length; i += 2) {
    bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
  }
  return bytes;
}

/**
 * Convert PublicKey to string representation
 */
export function publicKeyToString(pubkey: PublicKey): string {
  return bytesToHex(pubkey);
}

/**
 * Convert string to PublicKey
 */
export function stringToPublicKey(str: string): PublicKey {
  return hexToBytes(str);
}

/**
 * Convert Credits to number (for display)
 */
export function creditsToNumber(credits: Credits): number {
  return Number(credits);
}

/**
 * Convert number to Credits
 */
export function numberToCredits(num: number): Credits {
  return BigInt(Math.floor(num));
}
