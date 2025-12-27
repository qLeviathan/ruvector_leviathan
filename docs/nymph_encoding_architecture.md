# Nymph Encoding Structure - System Architecture

## Executive Summary

**NYMPH** (Neuromorphic Yield Memory Pattern Hierarchy) is a metamorphic vector encoding system that automatically adapts compression strategies based on access patterns, achieving 10-64x memory reduction while maintaining query performance for hot data.

## Architecture Decision Records (ADRs)

### ADR-1: Three-Stage Metamorphic Encoding

**Context**: Vector databases face the "cold start problem" where most vectors are rarely accessed but consume significant memory.

**Decision**: Implement three encoding stages with automatic promotion/demotion:
- **Larval**: Ultra-compressed (1-bit/4-bit) for cold data (~64x compression)
- **Nymph**: Balanced (8-bit scalar) for warm data (~4x compression)
- **Adult**: Full precision (f32) for hot data (1x, no compression)

**Rationale**:
1. **Pareto Principle**: 80% of queries access 20% of vectors
2. **Memory Hierarchy**: Match CPU cache (hot) → RAM (warm) → SSD (cold) patterns
3. **Workload Adaptation**: Automatically learns access patterns

**Trade-offs**:
- ✅ **Pro**: 10-64x memory reduction for cold data
- ✅ **Pro**: Sub-millisecond decompression for warm data
- ✅ **Pro**: No performance penalty for hot data
- ❌ **Con**: 5-15% accuracy loss for larval vectors
- ❌ **Con**: Stage transition overhead (~100μs)
- ❌ **Con**: Metadata tracking overhead (~24 bytes/vector)

**Alternatives Considered**:
1. **Two-stage (Binary + F32)**: Rejected - too aggressive, no middle ground
2. **Five-stage with 2-bit/16-bit**: Rejected - complexity not justified
3. **Static compression**: Rejected - doesn't adapt to workload

---

### ADR-2: Larval Stage Encoding Strategy

**Context**: Need extreme compression for rarely accessed vectors.

**Decision**: Support both Binary (1-bit) and Product Quantization (4-bit) with runtime selection.

**Rationale**:
- **Binary (1-bit)**: 32x compression, Hamming distance for fast similarity
- **Product Quantization (4-bit)**: 8x compression, better accuracy retention

**Implementation**:
```
Original: [128 dims × 4 bytes] = 512 bytes
Binary:   [128 dims × 1 bit] = 16 bytes (32x reduction)
PQ:       [128 dims ÷ 8 subs × 4 bits] = 64 bytes (8x reduction)
```

**Trade-offs**:
- ✅ **Pro**: Massive memory savings for archival data
- ✅ **Pro**: Binary enables SIMD Hamming distance
- ❌ **Con**: Lossy compression (5-15% accuracy loss)
- ❌ **Con**: PQ requires trained codebooks

---

### ADR-3: Nymph Stage Scalar Quantization

**Context**: Need fast, balanced encoding for warm data.

**Decision**: Use 8-bit scalar quantization with learned scale/offset.

**Formula**:
```
quantized = clamp((value - offset) / scale, 0, 255)
dequantized = quantized * scale + offset
```

**Rationale**:
- 4x compression vs f32
- O(1) decompression (single multiply-add)
- <5% accuracy degradation
- No codebook training required

**Trade-offs**:
- ✅ **Pro**: Simple, fast implementation
- ✅ **Pro**: Good accuracy/compression balance
- ✅ **Pro**: Cache-friendly (4x more vectors per cache line)
- ❌ **Con**: Less compression than PQ

---

### ADR-4: Access Pattern Tracking

**Context**: Need to decide when to promote/demote vectors between stages.

**Decision**: Hybrid LFU/LRU with exponential time decay.

**Algorithm**:
```
access_score = (access_count × frequency_weight) +
               (recency_weight / time_since_last_access) ×
               exp(-decay_factor × age)
```

**Promotion Triggers**:
- Larval → Nymph: `access_score > threshold_1` OR `access_count > 10`
- Nymph → Adult: `access_score > threshold_2` OR `access_frequency > 1/sec`

**Demotion Triggers**:
- Adult → Nymph: No access for 60 seconds AND `access_score < threshold_3`
- Nymph → Larval: No access for 300 seconds AND `access_score < threshold_4`

**Trade-offs**:
- ✅ **Pro**: Adapts to workload shifts
- ✅ **Pro**: Prevents thrashing with hysteresis
- ❌ **Con**: Requires tunable thresholds
- ❌ **Con**: Metadata storage overhead

---

### ADR-5: Distance Computation Strategy

**Context**: Different encodings require different distance metrics.

**Decision**: Implement stage-specific distance functions with early termination.

**Distance Functions**:
- **Larval Binary**: Hamming distance (SIMD popcount)
- **Larval PQ**: Asymmetric distance (lookup table)
- **Nymph**: Quantized L2 (8-bit SIMD)
- **Adult**: Full precision L2/Cosine

**Optimization**: Two-phase search
1. **Coarse filter**: Use larval/nymph distances for candidate selection
2. **Refinement**: Promote top-k to adult for exact distance

**Trade-offs**:
- ✅ **Pro**: 10-100x faster coarse search
- ✅ **Pro**: Minimal accuracy loss with proper k
- ❌ **Con**: Requires re-ranking overhead

---

## Component Diagrams

### C4 Level 1: System Context

```
┌─────────────────────────────────────────────────────────┐
│                   Vector Database                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │         Nymph Encoding System                     │  │
│  │                                                    │  │
│  │  Larval ◄──► Nymph ◄──► Adult                    │  │
│  │  (cold)      (warm)      (hot)                     │  │
│  └──────────────────────────────────────────────────┘  │
│         ▲                                 ▲              │
│         │                                 │              │
│    Query Engine                      Storage Layer      │
│                                                          │
└─────────────────────────────────────────────────────────┘
         ▲
         │
    Applications
  (AgentDB, VectorDB)
```

### C4 Level 2: Container Diagram

```
┌─────────────────────────────────────────────────────────┐
│              Nymph Encoding Container                    │
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Larval     │  │    Nymph     │  │    Adult     │ │
│  │   Encoder    │  │   Encoder    │  │   Encoder    │ │
│  │              │  │              │  │              │ │
│  │ • Binary     │  │ • Scalar     │  │ • Raw F32    │ │
│  │ • PQ         │  │   Quant      │  │ • No loss    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         ▲                 ▲                 ▲           │
│         └─────────────────┴─────────────────┘           │
│                           │                              │
│                 ┌─────────▼─────────┐                   │
│                 │  Stage Manager    │                   │
│                 │                   │                   │
│                 │ • Access Tracker  │                   │
│                 │ • Promotion Logic │                   │
│                 │ • Demotion Logic  │                   │
│                 └───────────────────┘                   │
│                           │                              │
│                 ┌─────────▼─────────┐                   │
│                 │   Metadata Store  │                   │
│                 │                   │                   │
│                 │ • Access counts   │                   │
│                 │ • Timestamps      │                   │
│                 │ • Stage history   │                   │
│                 └───────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

### C4 Level 3: Component Diagram - Stage Transitions

```
                    ┌──────────────────┐
                    │  Access Event    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Update Metadata  │
                    │ • access_count++ │
                    │ • last_access=now│
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Compute Score    │
                    │ score = f(count, │
                    │    recency, age) │
                    └────────┬─────────┘
                             │
                   ┌─────────┴─────────┐
                   │                   │
         ▼                             ▼
┌────────────────┐          ┌────────────────┐
│  Promote?      │          │  Demote?       │
│ score > thresh │          │ score < thresh │
└────────┬───────┘          └────────┬───────┘
         │ YES                       │ YES
         ▼                           ▼
┌────────────────┐          ┌────────────────┐
│ Decompress     │          │ Compress       │
│ Encode to      │          │ Encode to      │
│ Higher Stage   │          │ Lower Stage    │
└────────┬───────┘          └────────┬───────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
            ┌────────────────┐
            │ Update Stage   │
            │ in Metadata    │
            └────────────────┘
```

---

## Data Flow Diagrams

### Vector Insertion Flow

```
┌─────────┐
│ Insert  │
│ Vector  │
│ (f32[]) │
└────┬────┘
     │
     ▼
┌─────────────────┐
│ Initial Stage:  │
│ NYMPH (default) │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ Scalar Quantize │
│ to u8[]         │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ Create Metadata │
│ • count = 0     │
│ • timestamp=now │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ Store in Index  │
└─────────────────┘
```

### Query Search Flow

```
┌─────────┐
│ Query   │
│ Vector  │
└────┬────┘
     │
     ▼
┌─────────────────────────┐
│ Phase 1: Coarse Filter  │
│ • Search larval vectors │
│ • Use Hamming distance  │
│ • Get top-10k           │
└────┬────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ Phase 2: Warm Refinement│
│ • Search nymph vectors  │
│ • Use quantized L2      │
│ • Merge top-1k          │
└────┬────────────────────┘
     │
     ▼
┌─────────────────────────┐
│ Phase 3: Hot Refinement │
│ • Promote top-100 to    │
│   adult if not already  │
│ • Compute exact L2      │
│ • Return top-k          │
└─────────────────────────┘
```

---

## Performance Characteristics

### Memory Savings

| Stage  | Encoding | Bytes/Vector (128D) | Compression | Use Case |
|--------|----------|---------------------|-------------|----------|
| Larval | Binary   | 16 bytes           | 32x         | Archives, Cold data |
| Larval | PQ(4-bit)| 64 bytes           | 8x          | Infrequent access |
| Nymph  | u8       | 128 bytes          | 4x          | Regular access |
| Adult  | f32      | 512 bytes          | 1x          | Frequent queries |

**Example**: 10M vectors, 128 dims, 80% cold, 15% warm, 5% hot
- Naive f32: `10M × 512 = 5.12 GB`
- Nymph: `(8M×16) + (1.5M×128) + (0.5M×512) = 128MB + 192MB + 256MB = 576MB`
- **Savings**: 88.7% reduction

### Latency Analysis

| Operation | Larval | Nymph | Adult |
|-----------|--------|-------|-------|
| Encode    | 50μs   | 10μs  | 0μs   |
| Decode    | 100μs  | 5μs   | 0μs   |
| Distance  | 2μs    | 3μs   | 8μs   |
| Promote   | 150μs  | 15μs  | N/A   |
| Demote    | N/A    | 10μs  | 50μs  |

**Query Latency Budget** (10M vectors, top-10):
- Phase 1 (Larval): `10M × 2μs = 20ms`
- Phase 2 (Nymph): `1M × 3μs = 3ms`
- Phase 3 (Adult): `100 × 8μs = 0.8ms`
- **Total**: ~24ms (vs 80ms for full f32 scan)

---

## Integration Architecture

### VectorDB Storage Layer Integration

```rust
// Storage manager handles stage-aware persistence
impl VectorStorage {
    fn store(&mut self, vector: NymphVector) {
        match vector.stage() {
            Larval => self.cold_storage.append(vector),
            Nymph  => self.warm_storage.insert(vector),
            Adult  => self.hot_cache.insert(vector),
        }
    }

    fn query(&self, q: &[f32], k: usize) -> Vec<(VectorId, f32)> {
        // Two-phase search
        let candidates = self.cold_storage.search_larval(q, k * 100);
        let refined = self.warm_storage.refine(q, candidates, k * 10);
        self.hot_cache.exact_search(q, refined, k)
    }
}
```

### AgentDB Episode Memory Integration

```rust
// Episodes stored in nymph encoding by default
// Recent episodes promoted to adult automatically
impl AgentDB {
    fn store_episode(&mut self, episode: Episode) {
        let embedding = self.embed(episode);
        let nymph_vec = NymphVector::new_nymph(embedding);
        self.db.insert(nymph_vec);
    }

    fn retrieve_similar(&self, query: &str, k: usize) -> Vec<Episode> {
        let q_embed = self.embed(query);
        // Automatically promotes accessed episodes
        self.db.query(&q_embed, k)
            .iter()
            .map(|(id, _)| self.get_episode(*id))
            .collect()
    }
}
```

### Spike Train Temporal Encoding

```rust
// Spike trains benefit from binary larval encoding
impl SpikeTrainEncoder {
    fn encode_temporal(&self, spikes: &[Spike]) -> LarvalEncoded {
        let binary = spikes.iter()
            .map(|s| if s.fired { 1 } else { 0 })
            .collect::<BitVec>();
        LarvalEncoded::from_binary(binary)
    }

    fn hamming_similarity(&self, a: &LarvalEncoded, b: &LarvalEncoded) -> f32 {
        // SIMD-accelerated Hamming distance
        let xor = a.binary_xor(b);
        1.0 - (xor.count_ones() as f32 / a.len() as f32)
    }
}
```

---

## Operational Considerations

### Monitoring Metrics

```rust
pub struct NymphMetrics {
    // Stage distribution
    larval_count: AtomicU64,
    nymph_count: AtomicU64,
    adult_count: AtomicU64,

    // Transition rates
    promotions_per_sec: AtomicU64,
    demotions_per_sec: AtomicU64,

    // Performance
    avg_query_latency_ms: AtomicF64,
    cache_hit_rate: AtomicF64,

    // Memory
    total_memory_bytes: AtomicU64,
    compression_ratio: AtomicF64,
}
```

### Tuning Parameters

```rust
pub struct NymphConfig {
    // Promotion thresholds
    larval_to_nymph_accesses: u64,      // Default: 10
    nymph_to_adult_accesses: u64,       // Default: 100

    // Demotion timeouts (seconds)
    adult_to_nymph_timeout: u64,        // Default: 60
    nymph_to_larval_timeout: u64,       // Default: 300

    // Time decay
    access_decay_factor: f64,           // Default: 0.01

    // Memory limits
    max_adult_vectors: usize,           // Default: 10% of total
    max_nymph_vectors: usize,           // Default: 20% of total
}
```

### Failure Modes & Mitigation

| Failure Mode | Symptom | Mitigation |
|--------------|---------|------------|
| Thrashing | Frequent promote/demote cycles | Hysteresis in thresholds |
| Memory leak | Adult vectors never demote | Forced demotion on memory pressure |
| Cold start | All vectors larval, slow queries | Pre-promote representative subset |
| Accuracy loss | Poor recall on larval search | Increase Phase 1 candidate count |

---

## Future Extensions

### Planned Enhancements

1. **Adaptive PQ Codebooks**: Retrain codebooks based on access patterns
2. **GPU Acceleration**: CUDA kernels for batch encoding/distance
3. **Streaming Compression**: Compress during insertion without blocking
4. **Multi-tier Storage**: SSD for larval, RAM for nymph/adult
5. **Federated Nymph**: Distributed stage management across nodes

### Research Directions

1. **Neural Quantization**: Learn optimal quantization with neural networks
2. **Temporal Patterns**: Predict access patterns with LSTM
3. **Hierarchical Nymph**: Sub-stages within each metamorphic level
4. **Cross-modal Nymph**: Different encodings for different modalities

---

## References

- **Product Quantization**: Jégou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011
- **Scalar Quantization**: Guo et al., "Quantization based Fast Inner Product Search", AISTATS 2016
- **Binary Embeddings**: Rastegari et al., "XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks", ECCV 2016
- **Adaptive Compression**: Wu et al., "Multiscale Quantization for Fast Similarity Search", KDD 2017
