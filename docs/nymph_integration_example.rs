// Nymph Encoding Integration Example
// Demonstrates practical usage in ruvector_leviathan vector database

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

// Import nymph types (in actual implementation, these would be modules)
// use crate::nymph::{NymphVector, StageManager, TransitionPolicy, TransitionConfig};

/// Example: VectorDB with Nymph encoding integration
pub struct NymphVectorDB {
    /// Vectors organized by stage for efficient querying
    larval_vectors: Arc<RwLock<HashMap<u64, NymphVector>>>,
    nymph_vectors: Arc<RwLock<HashMap<u64, NymphVector>>>,
    adult_vectors: Arc<RwLock<HashMap<u64, NymphVector>>>,

    /// Stage transition manager
    stage_manager: Arc<StageManager>,

    /// Next vector ID
    next_id: Arc<RwLock<u64>>,
}

impl NymphVectorDB {
    pub fn new() -> Self {
        let policy = TransitionPolicy::default();
        let config = TransitionConfig::default();

        Self {
            larval_vectors: Arc::new(RwLock::new(HashMap::new())),
            nymph_vectors: Arc::new(RwLock::new(HashMap::new())),
            adult_vectors: Arc::new(RwLock::new(HashMap::new())),
            stage_manager: Arc::new(StageManager::new(policy, config)),
            next_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Inserts a vector (starts in Nymph stage by default)
    pub fn insert(&self, vector: Vec<f32>) -> u64 {
        let mut id_lock = self.next_id.write().unwrap();
        let id = *id_lock;
        *id_lock += 1;
        drop(id_lock);

        let nymph_vec = NymphVector::new_nymph(id, &vector);
        self.nymph_vectors.write().unwrap().insert(id, nymph_vec);

        id
    }

    /// Queries for k-nearest neighbors using three-phase search
    pub fn query(&self, query_vector: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Phase 1: Coarse filter with larval vectors (Hamming distance)
        let larval_candidates = self.search_larval_phase(query_vector, k * 100);

        // Phase 2: Warm refinement with nymph vectors (quantized L2)
        let nymph_candidates = self.search_nymph_phase(query_vector, k * 10);

        // Phase 3: Hot refinement with adult vectors (exact L2)
        let mut all_candidates = Vec::new();
        all_candidates.extend(larval_candidates);
        all_candidates.extend(nymph_candidates);

        // Add adult vectors
        let adult_results = self.search_adult_phase(query_vector);
        all_candidates.extend(adult_results);

        // Sort by distance and return top-k
        all_candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_candidates.truncate(k);

        // Promote accessed vectors based on access patterns
        for (id, _) in &all_candidates {
            self.maybe_promote(*id);
        }

        all_candidates
    }

    /// Phase 1: Larval search (fast, approximate)
    fn search_larval_phase(&self, query: &[f32], top_n: usize) -> Vec<(u64, f32)> {
        let larval_lock = self.larval_vectors.read().unwrap();
        let query_binary = LarvalEncoded::from_binary(query);

        let mut results: Vec<(u64, f32)> = larval_lock
            .iter()
            .map(|(id, vec)| {
                let distance = match &vec.encoding {
                    NymphEncoding::Larval(larval) => {
                        // Hamming distance (fast SIMD operation)
                        larval.hamming_distance(&query_binary) as f32
                    }
                    _ => unreachable!("Larval vectors should have larval encoding"),
                };

                // Record access
                vec.metadata.record_access(2); // ~2μs for Hamming

                (*id, distance)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_n);
        results
    }

    /// Phase 2: Nymph search (balanced)
    fn search_nymph_phase(&self, query: &[f32], top_n: usize) -> Vec<(u64, f32)> {
        let nymph_lock = self.nymph_vectors.read().unwrap();
        let query_nymph = NymphEncoded::from_f32(query);

        let mut results: Vec<(u64, f32)> = nymph_lock
            .iter()
            .map(|(id, vec)| {
                let distance = match &vec.encoding {
                    NymphEncoding::Nymph(nymph) => {
                        // Quantized L2 distance (fast SIMD)
                        nymph.quantized_l2_distance(&query_nymph)
                    }
                    _ => unreachable!("Nymph vectors should have nymph encoding"),
                };

                // Record access
                vec.metadata.record_access(3); // ~3μs for quantized L2

                (*id, distance)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(top_n);
        results
    }

    /// Phase 3: Adult search (exact, high quality)
    fn search_adult_phase(&self, query: &[f32]) -> Vec<(u64, f32)> {
        let adult_lock = self.adult_vectors.read().unwrap();

        adult_lock
            .iter()
            .map(|(id, vec)| {
                let distance = match &vec.encoding {
                    NymphEncoding::Adult(adult) => {
                        // Exact L2 distance
                        adult.l2_distance(&AdultEncoded::from_f32(query.to_vec()))
                    }
                    _ => unreachable!("Adult vectors should have adult encoding"),
                };

                // Record access
                vec.metadata.record_access(8); // ~8μs for full precision L2

                (*id, distance)
            })
            .collect()
    }

    /// Checks if vector should be promoted and performs promotion
    fn maybe_promote(&self, id: u64) {
        // Try to find vector in current stage and promote if needed
        if let Some(mut vector) = self.larval_vectors.write().unwrap().remove(&id) {
            if self.stage_manager.should_promote(&vector) {
                let decompressed = vector.decompress();
                let old_stage = vector.current_stage();

                vector.promote(&decompressed);
                let new_stage = vector.current_stage();

                self.stage_manager.record_transition(old_stage, new_stage);
                self.nymph_vectors.write().unwrap().insert(id, vector);

                println!("Promoted vector {} from Larval to Nymph", id);
            } else {
                self.larval_vectors.write().unwrap().insert(id, vector);
            }
        } else if let Some(mut vector) = self.nymph_vectors.write().unwrap().remove(&id) {
            if self.stage_manager.should_promote(&vector) {
                let decompressed = vector.decompress();
                let old_stage = vector.current_stage();

                vector.promote(&decompressed);
                let new_stage = vector.current_stage();

                self.stage_manager.record_transition(old_stage, new_stage);
                self.adult_vectors.write().unwrap().insert(id, vector);

                println!("Promoted vector {} from Nymph to Adult", id);
            } else {
                self.nymph_vectors.write().unwrap().insert(id, vector);
            }
        }
    }

    /// Background task: demote cold vectors to save memory
    pub fn compact(&self) {
        // Demote adult vectors that haven't been accessed recently
        let mut to_demote_adult = Vec::new();
        {
            let adult_lock = self.adult_vectors.read().unwrap();
            for (id, vector) in adult_lock.iter() {
                if self.stage_manager.should_demote(vector) {
                    to_demote_adult.push(*id);
                }
            }
        }

        for id in to_demote_adult {
            if let Some(mut vector) = self.adult_vectors.write().unwrap().remove(&id) {
                let decompressed = vector.decompress();
                let old_stage = vector.current_stage();

                vector.demote(&decompressed);
                let new_stage = vector.current_stage();

                self.stage_manager.record_transition(old_stage, new_stage);
                self.nymph_vectors.write().unwrap().insert(id, vector);

                println!("Demoted vector {} from Adult to Nymph", id);
            }
        }

        // Demote nymph vectors that are cold
        let mut to_demote_nymph = Vec::new();
        {
            let nymph_lock = self.nymph_vectors.read().unwrap();
            for (id, vector) in nymph_lock.iter() {
                if self.stage_manager.should_demote(vector) {
                    to_demote_nymph.push(*id);
                }
            }
        }

        for id in to_demote_nymph {
            if let Some(mut vector) = self.nymph_vectors.write().unwrap().remove(&id) {
                let decompressed = vector.decompress();
                let old_stage = vector.current_stage();

                vector.demote(&decompressed);
                let new_stage = vector.current_stage();

                self.stage_manager.record_transition(old_stage, new_stage);
                self.larval_vectors.write().unwrap().insert(id, vector);

                println!("Demoted vector {} from Nymph to Larval", id);
            }
        }
    }

    /// Returns memory statistics
    pub fn stats(&self) -> NymphStats {
        let (larval_count, nymph_count, adult_count) = self.stage_manager.stage_distribution();

        // Estimate memory usage (128 dimensions)
        let larval_bytes = larval_count * 16;  // Binary encoding
        let nymph_bytes = nymph_count * 128;   // u8 encoding
        let adult_bytes = adult_count * 512;   // f32 encoding

        let total_bytes = larval_bytes + nymph_bytes + adult_bytes;
        let total_vectors = larval_count + nymph_count + adult_count;
        let naive_bytes = total_vectors * 512; // All as f32

        NymphStats {
            larval_count,
            nymph_count,
            adult_count,
            total_memory_bytes: total_bytes,
            compression_ratio: naive_bytes as f64 / total_bytes as f64,
        }
    }
}

/// Memory and performance statistics
#[derive(Debug)]
pub struct NymphStats {
    pub larval_count: u64,
    pub nymph_count: u64,
    pub adult_count: u64,
    pub total_memory_bytes: u64,
    pub compression_ratio: f64,
}

// ============================================================================
// AGENTDB INTEGRATION EXAMPLE
// ============================================================================

/// Example: AgentDB episode memory with Nymph encoding
pub struct AgentEpisodeMemory {
    /// Episode embeddings stored as Nymph vectors
    embeddings: NymphVectorDB,

    /// Episode metadata (text, rewards, etc.)
    episodes: Arc<RwLock<HashMap<u64, Episode>>>,
}

#[derive(Debug, Clone)]
pub struct Episode {
    pub id: u64,
    pub text: String,
    pub reward: f32,
    pub timestamp: u64,
}

impl AgentEpisodeMemory {
    pub fn new() -> Self {
        Self {
            embeddings: NymphVectorDB::new(),
            episodes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Stores a new episode
    pub fn store_episode(&self, text: String, embedding: Vec<f32>, reward: f32) -> u64 {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Store embedding in Nymph DB (starts in Nymph stage)
        let id = self.embeddings.insert(embedding);

        // Store episode metadata
        let episode = Episode { id, text, reward, timestamp };
        self.episodes.write().unwrap().insert(id, episode);

        id
    }

    /// Retrieves similar episodes (automatically promotes frequently accessed)
    pub fn retrieve_similar(&self, query_embedding: &[f32], k: usize) -> Vec<Episode> {
        let results = self.embeddings.query(query_embedding, k);

        let episodes_lock = self.episodes.read().unwrap();
        results
            .into_iter()
            .filter_map(|(id, _distance)| episodes_lock.get(&id).cloned())
            .collect()
    }

    /// Background compaction (demote cold episodes)
    pub fn compact_memory(&self) {
        self.embeddings.compact();

        let stats = self.embeddings.stats();
        println!("Episode Memory Stats:");
        println!("  Larval (cold): {} episodes", stats.larval_count);
        println!("  Nymph (warm):  {} episodes", stats.nymph_count);
        println!("  Adult (hot):   {} episodes", stats.adult_count);
        println!("  Memory used:   {:.2} MB", stats.total_memory_bytes as f64 / 1_048_576.0);
        println!("  Compression:   {:.2}x", stats.compression_ratio);
    }
}

// ============================================================================
// SPIKE TRAIN INTEGRATION EXAMPLE
// ============================================================================

/// Example: Spike train encoding for neuromorphic patterns
pub struct SpikeTrainDB {
    /// Spike patterns stored as Larval binary vectors
    patterns: Arc<RwLock<HashMap<u64, NymphVector>>>,

    /// Pattern metadata
    metadata: Arc<RwLock<HashMap<u64, SpikeMetadata>>>,

    next_id: Arc<RwLock<u64>>,
}

#[derive(Debug, Clone)]
pub struct SpikeMetadata {
    pub neuron_ids: Vec<usize>,
    pub spike_times: Vec<f64>,
    pub pattern_label: String,
}

impl SpikeTrainDB {
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(RwLock::new(HashMap::new())),
            metadata: Arc::new(RwLock::new(HashMap::new())),
            next_id: Arc::new(RwLock::new(0)),
        }
    }

    /// Encodes spike train as binary larval vector
    pub fn encode_spike_train(&self, spikes: Vec<bool>, metadata: SpikeMetadata) -> u64 {
        let mut id_lock = self.next_id.write().unwrap();
        let id = *id_lock;
        *id_lock += 1;
        drop(id_lock);

        // Convert spikes to f32 (1.0 for spike, 0.0 for no spike)
        let spike_vector: Vec<f32> = spikes.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect();

        // Store as Larval binary (perfect for spike trains)
        let larval_vec = NymphVector::new_larval(id, &spike_vector);
        self.patterns.write().unwrap().insert(id, larval_vec);
        self.metadata.write().unwrap().insert(id, metadata);

        id
    }

    /// Finds similar spike patterns using Hamming distance
    pub fn find_similar_patterns(&self, query_spikes: Vec<bool>, k: usize) -> Vec<(u64, u32)> {
        let query_vector: Vec<f32> = query_spikes.iter().map(|&s| if s { 1.0 } else { 0.0 }).collect();
        let query_larval = LarvalEncoded::from_binary(&query_vector);

        let patterns_lock = self.patterns.read().unwrap();
        let mut results: Vec<(u64, u32)> = patterns_lock
            .iter()
            .map(|(id, vec)| {
                let hamming_dist = match &vec.encoding {
                    NymphEncoding::Larval(larval) => larval.hamming_distance(&query_larval),
                    _ => unreachable!(),
                };

                (*id, hamming_dist)
            })
            .collect();

        results.sort_by_key(|&(_, dist)| dist);
        results.truncate(k);
        results
    }

    /// Computes pattern similarity percentage
    pub fn pattern_similarity(&self, pattern1_id: u64, pattern2_id: u64) -> f32 {
        let patterns_lock = self.patterns.read().unwrap();

        let p1 = patterns_lock.get(&pattern1_id).expect("Pattern 1 not found");
        let p2 = patterns_lock.get(&pattern2_id).expect("Pattern 2 not found");

        match (&p1.encoding, &p2.encoding) {
            (NymphEncoding::Larval(l1), NymphEncoding::Larval(l2)) => {
                let hamming = l1.hamming_distance(l2);
                let max_dist = l1.dimensions as u32;
                1.0 - (hamming as f32 / max_dist as f32)
            }
            _ => unreachable!(),
        }
    }
}

// ============================================================================
// USAGE EXAMPLE
// ============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_nymph_db_workflow() {
        let db = NymphVectorDB::new();

        // Insert 1000 vectors
        for i in 0..1000 {
            let vector = vec![i as f32; 128];
            db.insert(vector);
        }

        // Query (should use three-phase search)
        let query = vec![500.0; 128];
        let results = db.query(&query, 10);

        assert_eq!(results.len(), 10);

        // Check memory savings
        let stats = db.stats();
        println!("Compression ratio: {:.2}x", stats.compression_ratio);
        assert!(stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_agentdb_episode_memory() {
        let memory = AgentEpisodeMemory::new();

        // Store episodes
        for i in 0..100 {
            let text = format!("Episode {}: Agent performed action", i);
            let embedding = vec![i as f32 * 0.1; 128];
            let reward = (i % 10) as f32;

            memory.store_episode(text, embedding, reward);
        }

        // Query similar episodes
        let query = vec![5.0 * 0.1; 128];
        let similar = memory.retrieve_similar(&query, 5);

        assert_eq!(similar.len(), 5);

        // Compact (demote cold episodes)
        memory.compact_memory();
    }

    #[test]
    fn test_spike_train_encoding() {
        let spike_db = SpikeTrainDB::new();

        // Create spike pattern
        let pattern1 = vec![
            true, false, true, true, false, false, true, false,
            true, false, true, true, false, false, true, false,
        ];

        let metadata = SpikeMetadata {
            neuron_ids: vec![1, 2, 3],
            spike_times: vec![0.1, 0.2, 0.3],
            pattern_label: "Pattern A".to_string(),
        };

        let id1 = spike_db.encode_spike_train(pattern1.clone(), metadata);

        // Create similar pattern (1 bit different)
        let mut pattern2 = pattern1.clone();
        pattern2[0] = false;

        let metadata2 = SpikeMetadata {
            neuron_ids: vec![1, 2, 3],
            spike_times: vec![0.1, 0.2, 0.3],
            pattern_label: "Pattern B".to_string(),
        };

        let id2 = spike_db.encode_spike_train(pattern2.clone(), metadata2);

        // Compute similarity
        let similarity = spike_db.pattern_similarity(id1, id2);
        assert!(similarity > 0.9); // Should be >90% similar

        // Find similar patterns
        let similar = spike_db.find_similar_patterns(pattern1, 5);
        assert!(similar.len() > 0);
    }
}
