/**
 * Basic usage example for Swarm Ledger
 * Demonstrates core concepts: reverse swarm deployment, task creation, and economic flow
 */

import { ReverseSwarm, Ledger, generateKeyPair, FractalTask } from '../src';

async function main() {
  console.log('🌊 Swarm Ledger - Basic Usage Example\n');

  // ==========================================================================
  // STEP 1: Generate cryptographic identity
  // ==========================================================================
  console.log('1️⃣  Generating cryptographic identity...');
  const identity = await generateKeyPair();
  console.log(`   ✅ Public key: ${identity.publicKey.toString('hex').slice(0, 16)}...\n`);

  // ==========================================================================
  // STEP 2: Deploy reverse swarm from this device
  // ==========================================================================
  console.log('2️⃣  Deploying reverse swarm from this device...');
  const swarm = new ReverseSwarm();

  const instance = await swarm.deploy(
    process.env.SWARM_API_KEY || 'demo-api-key',
    {
      topology: 'mesh',
      maxAgents: 5,
      ledgerEndpoint: 'ws://localhost:8080',  // Local ledger for demo
      nodeIdentity: identity,
      consensusMode: 'raft',

      // Task market preferences
      taskMarket: {
        acceptTypes: ['code-review', 'testing', 'documentation'],
        minValue: BigInt(100),
        maxConcurrent: 3,
      },

      // Reputation requirements
      reputation: {
        minScoreToValidate: 500,
        specializations: ['rust', 'typescript', 'distributed-systems'],
      },
    }
  );

  console.log(`   ✅ Swarm deployed at: ${instance.publicEndpoint}`);
  console.log(`   📜 Deployment TX: ${instance.deploymentTxHash}\n`);

  // ==========================================================================
  // STEP 3: Connect to ledger
  // ==========================================================================
  console.log('3️⃣  Connecting to ledger network...');
  const ledger = new Ledger({
    endpoint: 'ws://localhost:8080',
    identity,
  });

  await ledger.connect();
  console.log(`   ✅ Connected to ledger\n`);

  // ==========================================================================
  // STEP 4: Create an epic task (fractal root)
  // ==========================================================================
  console.log('4️⃣  Creating epic task...');
  const epicTx = await ledger.submitTransaction({
    type: 'TaskCreate',
    data: {
      title: 'Build authentication system',
      description: 'Complete OAuth2, JWT, and session management implementation',
      value: BigInt(10000),
      proofType: {
        type: 'TestResults',
        config: {
          framework: 'jest',
          coverage: 0.8,
          timeout: 30000,
        },
      },
      tags: ['security', 'authentication', 'backend'],
      deadline: Date.now() + (7 * 24 * 60 * 60 * 1000), // 1 week
    },
  });

  console.log(`   ✅ Epic task created: ${epicTx.id}`);
  console.log(`   💰 Value: 10,000 credits\n`);

  // ==========================================================================
  // STEP 5: Decompose into fractal subtasks
  // ==========================================================================
  console.log('5️⃣  Decomposing into subtasks (fractal structure)...');
  const decomposeTx = await ledger.submitTransaction({
    type: 'TaskDecompose',
    data: {
      parent: epicTx.id,
      children: [
        {
          title: 'OAuth2 provider integration',
          description: 'Integrate with Google, GitHub, and Microsoft OAuth2',
          value: BigInt(3000),
        },
        {
          title: 'JWT token management',
          description: 'Secure JWT generation, validation, and refresh',
          value: BigInt(2000),
        },
        {
          title: 'Session storage (Redis)',
          description: 'Implement session storage with Redis clustering',
          value: BigInt(2000),
        },
        {
          title: 'Security tests',
          description: 'Comprehensive security testing suite',
          value: BigInt(2000),
        },
        {
          title: 'Documentation',
          description: 'API docs and integration guides',
          value: BigInt(1000),
        },
      ],
    },
  });

  console.log(`   ✅ Task decomposed into 5 subtasks`);
  console.log(`   📊 Economic conservation verified: 10,000 = 3,000 + 2,000 + 2,000 + 2,000 + 1,000\n`);

  // ==========================================================================
  // STEP 6: Query task hierarchy
  // ==========================================================================
  console.log('6️⃣  Querying task hierarchy...');
  const epicTask = await ledger.getTask(epicTx.id);

  console.log(`   Epic: ${epicTask.title}`);
  console.log(`   └── Children: ${epicTask.children.length}`);
  for (const childHash of epicTask.children) {
    const child = await ledger.getTask(childHash);
    console.log(`       ├── ${child.title} (${child.value} credits)`);
  }
  console.log();

  // ==========================================================================
  // STEP 7: Simulate task claim
  // ==========================================================================
  console.log('7️⃣  Claiming first subtask...');
  const firstSubtask = epicTask.children[0];

  const claimTx = await ledger.submitTransaction({
    type: 'TaskClaim',
    data: {
      task: firstSubtask,
      stake: BigInt(300), // 10% stake (refunded on success)
      estimatedCompletion: Date.now() + (3 * 24 * 60 * 60 * 1000), // 3 days
    },
  });

  console.log(`   ✅ Task claimed`);
  console.log(`   💰 Staked: 300 credits (refunded on completion)\n`);

  // ==========================================================================
  // STEP 8: Simulate task completion
  // ==========================================================================
  console.log('8️⃣  Completing task with proof...');

  const completeTx = await ledger.submitTransaction({
    type: 'TaskComplete',
    data: {
      task: firstSubtask,
      proof: {
        type: 'TestResults',
        data: new TextEncoder().encode(JSON.stringify({
          framework: 'jest',
          passed: 45,
          failed: 0,
          coverage: 0.87,
          duration: 2345,
        })),
        timestamp: Date.now(),
      },
    },
  });

  console.log(`   ✅ Task completed with proof`);
  console.log(`   📋 Tests: 45 passed, 0 failed, 87% coverage\n`);

  // ==========================================================================
  // STEP 9: Validate and distribute rewards
  // ==========================================================================
  console.log('9️⃣  Validators reviewing proof...');

  // Simulate 3 validators approving
  for (let i = 0; i < 3; i++) {
    const validatorIdentity = await generateKeyPair();
    await ledger.submitTransaction({
      type: 'TaskValidate',
      data: {
        task: firstSubtask,
        verdict: true,
        reason: 'Tests pass, code quality good, coverage exceeds requirement',
      },
    });
  }

  console.log(`   ✅ 3/3 validators approved`);
  console.log(`   💸 Distributing rewards:\n`);

  // Economic split (default): 60% implementation, 20% review, 10% validation, 10% pool
  const taskValue = 3000;
  console.log(`      Implementation (60%): ${taskValue * 0.6} credits → Task completer`);
  console.log(`      Review (20%):         ${taskValue * 0.2} credits → Split among reviewers`);
  console.log(`      Validation (10%):     ${taskValue * 0.1} credits → Split among validators`);
  console.log(`      Community pool (10%): ${taskValue * 0.1} credits → Infrastructure\n`);

  // ==========================================================================
  // STEP 10: Check reputation update
  // ==========================================================================
  console.log('🔟  Checking reputation update...');
  const reputation = await ledger.getReputation(identity.publicKey);

  console.log(`   Total reputation: ${reputation.total.toFixed(2)}`);
  console.log(`   Breakdown:`);
  console.log(`      Code contributions:    ${reputation.breakdown.codeContributions.toFixed(2)}`);
  console.log(`      Code reviews:          ${reputation.breakdown.codeReviews.toFixed(2)}`);
  console.log(`      Task completion:       ${reputation.breakdown.taskCompletion.toFixed(2)}`);
  console.log(`      Community engagement:  ${reputation.breakdown.communityEngagement.toFixed(2)}`);
  console.log(`      Swarm operations:      ${reputation.breakdown.swarmOperations.toFixed(2)}\n`);

  // ==========================================================================
  // STEP 11: Audit trail
  // ==========================================================================
  console.log('1️⃣1️⃣  Generating audit trail...');
  const auditReport = await ledger.audit(epicTx.id, completeTx.id);

  console.log(`   ✅ Audit verified`);
  console.log(`   📜 Transactions in chain: ${auditReport.transactions.length}`);
  console.log(`   🔒 All signatures valid: ${auditReport.valid}`);
  console.log(`   🌳 Merkle proof verified: ${auditReport.merkleProofValid}\n`);

  // ==========================================================================
  // Summary
  // ==========================================================================
  console.log('📊 Summary:');
  console.log('   ✅ Deployed reverse swarm from this device');
  console.log('   ✅ Created epic task worth 10,000 credits');
  console.log('   ✅ Decomposed into 5 subtasks (fractal)');
  console.log('   ✅ Claimed, completed, and validated task');
  console.log('   ✅ Distributed 3,000 credits to participants');
  console.log('   ✅ Updated reputation scores');
  console.log('   ✅ Verified full audit trail\n');

  console.log('🌊 All operations recorded in immutable ledger!');

  // Cleanup
  await ledger.disconnect();
}

// Run example
main().catch(console.error);
