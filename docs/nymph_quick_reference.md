# NYMPH Encoding - Quick Reference Card

## Stage Comparison Matrix

| Aspect | Larval 🐛 | Nymph 🦋 | Adult 🦋✨ |
|--------|----------|---------|-----------|
| **Encoding** | 1-bit / 4-bit | 8-bit scalar | 32-bit f32 |
| **Memory (128D)** | 16-64 bytes | 128 bytes | 512 bytes |
| **Compression** | 8-32x | 4x | 1x |
| **Distance** | Hamming | Quantized L2 | Exact L2 |
| **Accuracy** | 85-90% | 95-97% | 100% |
| **Encode Time** | 50μs | 10μs | 0μs |
| **Decode Time** | 100μs | 5μs | 0μs |
| **Distance Time** | 2μs | 3μs | 8μs |
| **Access Count** | <10 | 10-100 | >100 |
| **Access Freq** | <0.01/sec | 0.01-1/sec | >1/sec |
| **Idle Timeout** | Never | 300sec | 60sec |
| **Use Case** | Archive | Regular | Frequent |

## Stage Transitions Cheat Sheet

```
PROMOTION RULES:
  Larval → Nymph:  access_count ≥ 10  OR  score > 5.0
  Nymph  → Adult:  access_count ≥ 100 OR  freq > 1/sec

DEMOTION RULES:
  Adult → Nymph:   idle > 60sec  AND  score < 10.0
  Nymph → Larval:  idle > 300sec AND  score < 2.0

ACCESS SCORE:
  score = (count × freq_wt) + (recency_wt / time_since) × exp(-decay × age)

  Default weights:
    freq_wt = 1.0
    recency_wt = 10.0
    decay = 0.01
```

## Memory Calculation Formulas

```
LARVAL (Binary):      dims × 1 bit  = dims / 8 bytes
LARVAL (PQ 4-bit):    dims × 4 bits = dims / 2 bytes
NYMPH (8-bit):        dims × 8 bits = dims bytes
ADULT (f32):          dims × 32 bits = dims × 4 bytes

Example (128 dimensions):
  Larval Binary:    128 / 8 = 16 bytes    (32x compression)
  Larval PQ:        128 / 2 = 64 bytes    (8x compression)
  Nymph:            128 = 128 bytes        (4x compression)
  Adult:            128 × 4 = 512 bytes    (1x, no compression)

TOTAL MEMORY (10M vectors, 128D):
  memory = (L_count × 16) + (N_count × 128) + (A_count × 512)

  Example (80/15/5 split):
    = (8M × 16) + (1.5M × 128) + (0.5M × 512)
    = 128MB + 192MB + 256MB
    = 576MB  (vs 5.12GB naive = 88.7% savings)
```

## Query Latency Estimation

```
THREE-PHASE SEARCH (10M vectors, 128D, top-10):

Phase 1: Larval Coarse Filter
  vectors = 8M (80% cold)
  distance_time = 2μs (Hamming SIMD)
  candidates = 1000
  latency = 8M × 2μs = 16ms

Phase 2: Nymph Warm Refinement
  vectors = 1.5M (15% warm)
  distance_time = 3μs (Quantized L2)
  candidates = 100
  latency = 1.5M × 3μs = 4.5ms

Phase 3: Adult Hot Exact
  vectors = 0.5M (5% hot)
  distance_time = 8μs (Exact L2)
  candidates = 10
  latency = 0.5M × 8μs = 4ms

TOTAL: 16ms + 4.5ms + 4ms = 24.5ms

Compare to naive: 10M × 8μs = 80ms
Speedup: 80ms / 24.5ms = 3.3x
```

## Code Snippets

### Basic Usage
```rust
// Initialize DB
let db = NymphVectorDB::new();

// Insert (starts as Nymph)
let id = db.insert(vec![1.0, 2.0, 3.0, ...]);

// Query (auto-promotes accessed vectors)
let results = db.query(&query_vec, k=10);
// returns: Vec<(id, distance)>

// Background compaction
db.compact(); // demotes cold vectors
```

### Custom Configuration
```rust
let policy = TransitionPolicy {
    larval_to_nymph: AccessThreshold {
        min_access_count: 20,      // More strict
        min_score: 10.0,
        ..Default::default()
    },
    max_adult_vectors: 1_000_000,  // Cap hot cache
    ..Default::default()
};

let db = NymphVectorDB::with_policy(policy);
```

### Manual Stage Control
```rust
// Manually promote vector
let mut vec = db.get_mut(id).unwrap();
vec.promote(&decompressed_data);

// Manually demote vector
vec.demote(&decompressed_data);

// Check current stage
match vec.current_stage() {
    MetamorphicStage::Larval => println!("Cold"),
    MetamorphicStage::Nymph => println!("Warm"),
    MetamorphicStage::Adult => println!("Hot"),
}
```

### AgentDB Integration
```rust
let memory = AgentEpisodeMemory::new();

// Store episode (auto-encoded as Nymph)
memory.store_episode(
    "Agent completed task successfully".to_string(),
    embedding,
    reward=1.0
);

// Retrieve similar (promotes frequently accessed)
let similar = memory.retrieve_similar(&query_emb, k=5);

// Compact old episodes to Larval
memory.compact_memory();
```

### Spike Train Integration
```rust
let spike_db = SpikeTrainDB::new();

// Encode spike pattern (stored as Larval binary)
let spikes = vec![true, false, true, true, ...];
spike_db.encode_spike_train(spikes, metadata);

// Fast Hamming similarity
let similar = spike_db.find_similar_patterns(query, k=10);
```

## Performance Tuning Guide

### Scenario: High-Throughput Query System
**Goal**: Minimize latency, maximize QPS

```rust
TransitionPolicy {
    larval_to_nymph: { min_access_count: 5 },   // Fast promotion
    nymph_to_adult: { min_access_count: 50 },   // More adult vectors
    adult_to_nymph: { inactivity_timeout_sec: 120 }, // Slower demotion
    max_adult_vectors: 20% of total,            // Large hot cache
    time_decay_factor: 0.005,                   // Slower decay
}

Expected:
  - 20% adult vectors (vs 5% default)
  - P95 latency: 15ms (vs 30ms default)
  - Memory: 1.5GB (vs 576MB default)
  - QPS: 60+ (vs 40 default)
```

### Scenario: Memory-Constrained Archive
**Goal**: Maximize compression, minimize memory

```rust
TransitionPolicy {
    larval_to_nymph: { min_access_count: 50 },  // Slow promotion
    nymph_to_adult: { min_access_count: 500 },  // Very few adults
    adult_to_nymph: { inactivity_timeout_sec: 10 }, // Fast demotion
    nymph_to_larval: { inactivity_timeout_sec: 60 }, // Fast demotion
    max_adult_vectors: 1% of total,             // Tiny hot cache
    max_nymph_vectors: 5% of total,             // Small warm cache
}

Expected:
  - 94% larval, 5% nymph, 1% adult
  - Memory: 250MB (vs 576MB default) = 95% savings
  - P95 latency: 60ms (vs 30ms default)
  - Trade-off: Higher latency for extreme compression
```

### Scenario: Balanced Production
**Goal**: Good latency and compression (default)

```rust
TransitionPolicy::default()
// 80% larval, 15% nymph, 5% adult

Expected:
  - Memory: 576MB (88% savings)
  - P95 latency: 30ms
  - Recall@10: >95%
  - QPS: 40+
```

## Monitoring Alerts

### Critical Alerts
```yaml
# Thrashing Detection
- alert: NymphHighTransitionRate
  expr: rate(nymph_transitions_total[5m]) > 100
  severity: critical
  message: "Vector thrashing detected (>100 transitions/sec)"

# Memory Pressure
- alert: NymphMemoryExhausted
  expr: nymph_adult_vectors_total > nymph_max_adult_vectors * 0.95
  severity: warning
  message: "Adult vector limit reached, forced demotions occurring"

# Performance Degradation
- alert: NymphQueryLatencyHigh
  expr: histogram_quantile(0.95, nymph_query_latency_seconds) > 0.060
  severity: warning
  message: "P95 query latency >60ms (target: <30ms)"
```

### Info Metrics
```yaml
# Stage Distribution
nymph_larval_percent = nymph_larval_vectors / total_vectors
nymph_nymph_percent = nymph_nymph_vectors / total_vectors
nymph_adult_percent = nymph_adult_vectors / total_vectors

# Compression Ratio
nymph_compression_ratio = (total_vectors × 512) / nymph_total_memory_bytes

# Memory Savings
nymph_memory_savings_gb = (total_vectors × 512 - nymph_total_memory_bytes) / 1e9
```

## Troubleshooting Guide

### Problem: Low Compression Ratio
**Symptoms**: Memory usage >2GB for 10M vectors

**Diagnosis**:
```bash
# Check stage distribution
curl localhost:9090/metrics | grep nymph_.*_vectors_total

# Expected: 80% larval, 15% nymph, 5% adult
# Actual: 20% larval, 30% nymph, 50% adult  ← TOO MANY HOT
```

**Fix**: Increase promotion thresholds
```rust
policy.nymph_to_adult.min_access_count = 200; // Was 100
policy.adult_to_nymph.inactivity_timeout_sec = 30; // Was 60
```

---

### Problem: High Query Latency
**Symptoms**: P95 latency >100ms

**Diagnosis**:
```bash
# Check phase breakdown
curl localhost:9090/metrics | grep nymph_query_latency_seconds

# Phase 1 (larval): 50ms  ← TOO SLOW (target: 20ms)
# Phase 2 (nymph): 20ms   ← TOO SLOW (target: 5ms)
# Phase 3 (adult): 5ms    ← OK
```

**Fix**: Promote more vectors to faster stages
```rust
policy.larval_to_nymph.min_access_count = 5; // Was 10
policy.nymph_to_adult.min_access_count = 50; // Was 100
```

---

### Problem: Thrashing (Rapid Transitions)
**Symptoms**: High CPU, many transitions/sec

**Diagnosis**:
```bash
# Check transition rate
curl localhost:9090/metrics | grep nymph_transitions_total

# Rate: 150 transitions/sec  ← TOO HIGH (target: <10/sec)
```

**Fix**: Add hysteresis to thresholds
```rust
// Increase gap between promotion/demotion thresholds
policy.nymph_to_adult.min_score = 60.0;     // Was 50.0
policy.adult_to_nymph.min_score = 8.0;      // Was 10.0
// Gap: 60.0 - 8.0 = 52.0 (prevents rapid flip-flopping)
```

---

### Problem: Low Recall
**Symptoms**: Query recall@10 <90%

**Diagnosis**:
```bash
# Check encoding accuracy
# Larval accuracy: 82%  ← TOO LOW (target: 85-90%)
```

**Fix**: Use PQ instead of binary for larval
```rust
LarvalEncodingType::ProductQuant {
    num_subspaces: 8,
    bits_per_code: 4,
}
// PQ has higher accuracy than binary (90% vs 85%)
```

Or increase candidate pool:
```rust
// Phase 1: Return top-2000 instead of top-1000
let candidates = self.search_larval_phase(query, k * 200); // Was k * 100
```

---

## Architecture Decision Quick Reference

| Decision | Chosen Approach | Rationale | Trade-off |
|----------|----------------|-----------|-----------|
| **Number of stages** | 3 (Larval/Nymph/Adult) | Balance complexity vs granularity | More stages = more overhead |
| **Larval encoding** | Binary OR PQ | Binary for speed, PQ for accuracy | PQ requires codebook training |
| **Nymph encoding** | 8-bit scalar | Simple, fast, good accuracy | Less compression than PQ |
| **Adult encoding** | Raw f32 | Maximum speed, no decompression | No memory savings |
| **Transition policy** | LFU+LRU hybrid | Adapts to workload | Requires tuning |
| **Distance metric** | Stage-specific | Optimize for encoding | Need multiple implementations |
| **Query strategy** | Three-phase | Balance speed/accuracy | More complex than single-pass |

---

## Compression Algorithm Reference

### Binary Encoding (1-bit)
```python
threshold = mean(vector)
binary[i] = 1 if vector[i] > threshold else 0

# Decompression
decompressed[i] = threshold if binary[i] == 1 else 0
```

### Scalar Quantization (8-bit)
```python
min_val = min(vector)
max_val = max(vector)
scale = (max_val - min_val) / 255

# Encode
quantized[i] = clamp((vector[i] - min_val) / scale, 0, 255)

# Decode
decompressed[i] = quantized[i] * scale + min_val
```

### Product Quantization (4-bit)
```python
# Training (k-means on subspaces)
for subspace in range(num_subspaces):
    centroids[subspace] = kmeans(vectors[:, subspace_dims])

# Encode
for subspace in range(num_subspaces):
    code[subspace] = argmin(distance(vector[subspace_dims], centroids[subspace]))

# Decode
decompressed = concat([centroids[s][code[s]] for s in range(num_subspaces)])
```

---

## SIMD Optimization Cheat Sheet

### Hamming Distance (AVX2)
```rust
#[cfg(target_feature = "avx2")]
unsafe fn hamming_distance_avx2(a: &[u8], b: &[u8]) -> u32 {
    use std::arch::x86_64::*;

    let mut count = 0;
    for chunk in a.chunks_exact(32).zip(b.chunks_exact(32)) {
        let va = _mm256_loadu_si256(chunk.0.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(chunk.1.as_ptr() as *const __m256i);
        let vxor = _mm256_xor_si256(va, vb);

        // Count bits (popcount)
        let vpop = _mm256_sad_epu8(vxor, _mm256_setzero_si256());
        count += _mm256_extract_epi64(vpop, 0) as u32;
        count += _mm256_extract_epi64(vpop, 2) as u32;
    }
    count
}
```

### Quantized L2 (AVX2)
```rust
#[cfg(target_feature = "avx2")]
unsafe fn quantized_l2_avx2(a: &[u8], b: &[u8]) -> f32 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_si256();
    for chunk in a.chunks_exact(32).zip(b.chunks_exact(32)) {
        let va = _mm256_loadu_si256(chunk.0.as_ptr() as *const __m256i);
        let vb = _mm256_loadu_si256(chunk.1.as_ptr() as *const __m256i);

        // Difference
        let vdiff = _mm256_sub_epi8(va, vb);

        // Square
        let vsq = _mm256_mullo_epi16(vdiff, vdiff);

        // Accumulate
        sum = _mm256_add_epi32(sum, vsq);
    }

    // Horizontal sum and sqrt
    let total = horizontal_sum(sum);
    (total as f32).sqrt()
}
```

---

## Related Technologies Comparison

| Technology | Purpose | Nymph Advantage | Complementary? |
|------------|---------|-----------------|----------------|
| **HNSW** | Fast ANN search | Nymph reduces memory | ✅ Yes - combine both |
| **IVF** | Clustering for search | Nymph adaptive to access | ✅ Yes - use together |
| **PQ** | Fixed quantization | Nymph multi-stage adaptive | ⚠️ Partial - Nymph uses PQ |
| **ScaNN** | Google's ANN library | Nymph metamorphic stages | ✅ Yes - different focus |
| **DiskANN** | SSD-based ANN | Nymph RAM optimization | ✅ Yes - Nymph for RAM tier |
| **FAISS** | Full ANN library | Nymph workload-adaptive | ✅ Yes - can integrate |

**Best Practice**: Use Nymph for memory reduction + HNSW/IVF for search acceleration

---

## Further Reading

- Architecture: `/home/user/ruvector_leviathan/docs/nymph_encoding_architecture.md`
- Type Definitions: `/home/user/ruvector_leviathan/docs/nymph_encoding_types.rs`
- Integration Examples: `/home/user/ruvector_leviathan/docs/nymph_integration_example.rs`
- Implementation Plan: `/home/user/ruvector_leviathan/docs/nymph_implementation_plan.md`
- Summary: `/home/user/ruvector_leviathan/docs/NYMPH_ARCHITECTURE_SUMMARY.md`
