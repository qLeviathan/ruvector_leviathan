// Nymph Encoding Type System
// Complete Rust type definitions for the Neuromorphic Yield Memory Pattern Hierarchy

use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// CORE METAMORPHIC STAGE ENUM
// ============================================================================

/// Represents the three metamorphic stages of vector encoding.
///
/// Design Rationale:
/// - Larval: Cold data that's rarely accessed → aggressive compression
/// - Nymph: Warm data with moderate access → balanced compression
/// - Adult: Hot data with frequent access → no compression for speed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetamorphicStage {
    /// Ultra-compressed encoding (1-bit or 4-bit per dimension)
    /// Compression ratio: 8-32x
    /// Use case: Cold vectors accessed <10 times
    Larval,

    /// Balanced encoding (8-bit scalar quantization)
    /// Compression ratio: 4x
    /// Use case: Warm vectors accessed 10-100 times
    Nymph,

    /// Full precision encoding (32-bit f32)
    /// Compression ratio: 1x (no compression)
    /// Use case: Hot vectors accessed >100 times or >1/sec
    Adult,
}

impl MetamorphicStage {
    /// Returns the compression ratio for this stage
    pub fn compression_ratio(&self) -> f32 {
        match self {
            Self::Larval => 16.0, // Average of binary (32x) and PQ (8x)
            Self::Nymph => 4.0,
            Self::Adult => 1.0,
        }
    }

    /// Returns the typical decompression latency in microseconds
    pub fn decompression_latency_us(&self) -> u64 {
        match self {
            Self::Larval => 100,
            Self::Nymph => 5,
            Self::Adult => 0,
        }
    }

    /// Returns the next higher stage (if any)
    pub fn promote(&self) -> Option<Self> {
        match self {
            Self::Larval => Some(Self::Nymph),
            Self::Nymph => Some(Self::Adult),
            Self::Adult => None,
        }
    }

    /// Returns the next lower stage (if any)
    pub fn demote(&self) -> Option<Self> {
        match self {
            Self::Larval => None,
            Self::Nymph => Some(Self::Larval),
            Self::Adult => Some(Self::Nymph),
        }
    }
}

// ============================================================================
// LARVAL ENCODING - Ultra-Compressed (Cold Data)
// ============================================================================

/// Encoding type for larval stage.
///
/// Design Rationale:
/// - Binary: Best for Hamming-based similarity, SIMD-friendly
/// - ProductQuant: Better accuracy, requires trained codebooks
#[derive(Debug, Clone, PartialEq)]
pub enum LarvalEncodingType {
    /// Binary encoding (1 bit per dimension)
    /// Memory: 128 dims → 16 bytes (32x compression)
    /// Distance: Hamming (popcount)
    Binary,

    /// Product Quantization (4 bits per dimension)
    /// Memory: 128 dims → 64 bytes (8x compression)
    /// Distance: Asymmetric (lookup table)
    ProductQuant {
        /// Number of subspaces (typically 8 or 16)
        num_subspaces: usize,
        /// Bits per subspace code (typically 4 or 8)
        bits_per_code: usize,
    },
}

/// Larval-stage encoded vector (ultra-compressed).
///
/// Design Rationale:
/// - Pack bits tightly to minimize memory footprint
/// - Store decompression metadata (codebooks, centroids) separately
/// - Trade accuracy for 8-32x memory savings
#[derive(Debug, Clone)]
pub struct LarvalEncoded {
    /// Packed binary or PQ codes
    /// For binary: bits packed into bytes (big-endian)
    /// For PQ: 4-bit codes packed into bytes
    pub data: Vec<u8>,

    /// Original vector dimensionality
    pub dimensions: usize,

    /// Encoding variant and parameters
    pub encoding_type: LarvalEncodingType,

    /// For ProductQuant: codebook of centroids
    /// Shape: [num_subspaces, num_centroids (2^bits), subspace_dims]
    pub codebook: Option<Vec<Vec<f32>>>,

    /// For Binary: learned centroids for decompression
    /// Shape: [dimensions]
    pub centroids: Option<Vec<f32>>,

    /// Quantization scale for normalization
    pub scale: f32,
}

impl LarvalEncoded {
    /// Compresses a float vector to binary encoding
    pub fn from_binary(vector: &[f32]) -> Self {
        let threshold = vector.iter().sum::<f32>() / vector.len() as f32;
        let bits: Vec<u8> = vector.chunks(8)
            .map(|chunk| {
                chunk.iter().enumerate()
                    .fold(0u8, |acc, (i, &v)| {
                        acc | (((v > threshold) as u8) << i)
                    })
            })
            .collect();

        Self {
            data: bits,
            dimensions: vector.len(),
            encoding_type: LarvalEncodingType::Binary,
            codebook: None,
            centroids: Some(vec![threshold; vector.len()]),
            scale: 1.0,
        }
    }

    /// Decompresses to approximate float vector
    pub fn decompress(&self) -> Vec<f32> {
        match &self.encoding_type {
            LarvalEncodingType::Binary => {
                let centroids = self.centroids.as_ref().unwrap();
                self.data.iter()
                    .enumerate()
                    .flat_map(|(byte_idx, &byte)| {
                        (0..8).map(move |bit_idx| {
                            let dim_idx = byte_idx * 8 + bit_idx;
                            if dim_idx < self.dimensions {
                                let bit = (byte >> bit_idx) & 1;
                                if bit == 1 { centroids[dim_idx] } else { 0.0 }
                            } else {
                                0.0
                            }
                        })
                    })
                    .take(self.dimensions)
                    .collect()
            }
            LarvalEncodingType::ProductQuant { num_subspaces, .. } => {
                // PQ decompression: lookup codes in codebook
                let codebook = self.codebook.as_ref().unwrap();
                let mut result = Vec::with_capacity(self.dimensions);

                for (subspace_idx, &code) in self.data.iter().enumerate().take(*num_subspaces) {
                    let centroid = &codebook[subspace_idx][code as usize];
                    result.extend_from_slice(centroid);
                }

                result
            }
        }
    }

    /// Computes Hamming distance to another larval binary vector (fast)
    #[cfg(target_arch = "x86_64")]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        debug_assert_eq!(self.encoding_type, LarvalEncodingType::Binary);

        self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Memory footprint in bytes
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() +
        self.data.len() +
        self.codebook.as_ref().map_or(0, |cb| cb.len() * cb[0].len() * 4) +
        self.centroids.as_ref().map_or(0, |c| c.len() * 4)
    }
}

// ============================================================================
// NYMPH ENCODING - Balanced (Warm Data)
// ============================================================================

/// Nymph-stage encoded vector (8-bit scalar quantization).
///
/// Design Rationale:
/// - Simple linear quantization: fast to compress/decompress
/// - 4x compression with <5% accuracy loss
/// - SIMD-friendly for vectorized distance computation
#[derive(Debug, Clone)]
pub struct NymphEncoded {
    /// 8-bit quantized values
    /// Quantization: q = clamp((v - offset) / scale, 0, 255)
    pub data: Vec<u8>,

    /// Original vector dimensionality
    pub dimensions: usize,

    /// Quantization scale factor
    /// Computed as: (max - min) / 255
    pub scale: f32,

    /// Quantization offset
    /// Computed as: min value in original vector
    pub offset: f32,
}

impl NymphEncoded {
    /// Compresses a float vector to 8-bit quantization
    pub fn from_f32(vector: &[f32]) -> Self {
        let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = (max - min) / 255.0;

        let data = vector.iter()
            .map(|&v| {
                let normalized = (v - min) / scale;
                normalized.clamp(0.0, 255.0) as u8
            })
            .collect();

        Self {
            data,
            dimensions: vector.len(),
            scale,
            offset: min,
        }
    }

    /// Decompresses to float vector
    pub fn decompress(&self) -> Vec<f32> {
        self.data.iter()
            .map(|&q| (q as f32) * self.scale + self.offset)
            .collect()
    }

    /// Computes L2 distance in quantized space (fast approximation)
    pub fn quantized_l2_distance(&self, other: &Self) -> f32 {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| {
                let diff = (a as i16 - b as i16) as f32;
                diff * diff
            })
            .sum::<f32>()
            .sqrt() * self.scale.max(other.scale)
    }

    /// Memory footprint in bytes
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len()
    }
}

// ============================================================================
// ADULT ENCODING - Full Precision (Hot Data)
// ============================================================================

/// Adult-stage encoded vector (full f32 precision).
///
/// Design Rationale:
/// - No compression for maximum speed
/// - Direct memory access for distance computation
/// - Cache-friendly for frequently accessed vectors
#[derive(Debug, Clone)]
pub struct AdultEncoded {
    /// Full precision float values
    pub data: Vec<f32>,

    /// Vector dimensionality
    pub dimensions: usize,
}

impl AdultEncoded {
    /// Creates adult encoding from float vector (no compression)
    pub fn from_f32(vector: Vec<f32>) -> Self {
        let dimensions = vector.len();
        Self { data: vector, dimensions }
    }

    /// Returns reference to underlying data (zero-copy)
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Computes exact L2 distance
    pub fn l2_distance(&self, other: &Self) -> f32 {
        self.data.iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Computes cosine similarity
    pub fn cosine_similarity(&self, other: &Self) -> f32 {
        let dot: f32 = self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).sum();
        let norm_a: f32 = self.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let norm_b: f32 = other.data.iter().map(|&v| v * v).sum::<f32>().sqrt();
        dot / (norm_a * norm_b)
    }

    /// Memory footprint in bytes
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.data.len() * 4
    }
}

// ============================================================================
// STAGE METADATA - Access Pattern Tracking
// ============================================================================

/// Metadata for tracking vector access patterns and stage transitions.
///
/// Design Rationale:
/// - Lightweight tracking (24 bytes) vs vector size (16-512 bytes)
/// - Lock-free atomic operations for concurrency
/// - Exponential time decay for recency weighting
#[derive(Debug)]
pub struct StageMetadata {
    /// Total number of times this vector was accessed
    access_count: AtomicU64,

    /// Timestamp of last access (Unix epoch microseconds)
    last_access_timestamp: AtomicU64,

    /// Timestamp when vector was created (Unix epoch microseconds)
    creation_timestamp: u64,

    /// Cumulative access time in microseconds (for avg computation)
    total_access_time: AtomicU64,

    /// Number of times this vector changed stages
    stage_transition_count: AtomicU32,

    /// Current metamorphic stage
    current_stage: MetamorphicStage,
}

impl StageMetadata {
    /// Creates new metadata for a freshly inserted vector
    pub fn new(initial_stage: MetamorphicStage) -> Self {
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        Self {
            access_count: AtomicU64::new(0),
            last_access_timestamp: AtomicU64::new(now_us),
            creation_timestamp: now_us,
            total_access_time: AtomicU64::new(0),
            stage_transition_count: AtomicU32::new(0),
            current_stage: initial_stage,
        }
    }

    /// Records an access event (called on every query hit)
    pub fn record_access(&self, access_latency_us: u64) {
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        self.access_count.fetch_add(1, Ordering::Relaxed);
        self.last_access_timestamp.store(now_us, Ordering::Relaxed);
        self.total_access_time.fetch_add(access_latency_us, Ordering::Relaxed);
    }

    /// Computes access score for promotion/demotion decisions
    ///
    /// Formula: score = (count × freq_weight) + (recency_weight / time_since) × exp(-decay × age)
    pub fn access_score(&self, config: &TransitionConfig) -> f64 {
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let count = self.access_count.load(Ordering::Relaxed) as f64;
        let last_access = self.last_access_timestamp.load(Ordering::Relaxed);
        let age_sec = (now_us - self.creation_timestamp) as f64 / 1_000_000.0;
        let time_since_access_sec = (now_us - last_access) as f64 / 1_000_000.0;

        // Frequency component
        let freq_component = count * config.frequency_weight;

        // Recency component with exponential decay
        let recency_component = if time_since_access_sec > 0.0 {
            (config.recency_weight / time_since_access_sec) *
            (-config.time_decay_factor * age_sec).exp()
        } else {
            config.recency_weight
        };

        freq_component + recency_component
    }

    /// Returns average access latency in microseconds
    pub fn avg_access_latency_us(&self) -> u64 {
        let total = self.total_access_time.load(Ordering::Relaxed);
        let count = self.access_count.load(Ordering::Relaxed);
        if count > 0 { total / count } else { 0 }
    }

    /// Returns seconds since last access
    pub fn seconds_since_last_access(&self) -> u64 {
        let now_us = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        let last = self.last_access_timestamp.load(Ordering::Relaxed);
        (now_us - last) / 1_000_000
    }

    /// Records a stage transition (promotion or demotion)
    pub fn record_transition(&mut self, new_stage: MetamorphicStage) {
        self.current_stage = new_stage;
        self.stage_transition_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Returns current stage
    pub fn current_stage(&self) -> MetamorphicStage {
        self.current_stage
    }

    /// Returns total access count
    pub fn access_count(&self) -> u64 {
        self.access_count.load(Ordering::Relaxed)
    }
}

// ============================================================================
// NYMPH VECTOR - Unified Wrapper
// ============================================================================

/// Unified wrapper for metamorphic vectors.
///
/// Design Rationale:
/// - Single type that can represent any stage
/// - Automatic stage transitions based on access patterns
/// - Thread-safe metadata updates
#[derive(Debug)]
pub struct NymphVector {
    /// Unique vector identifier
    pub id: u64,

    /// Current encoding (varies by stage)
    pub encoding: NymphEncoding,

    /// Access pattern metadata
    pub metadata: StageMetadata,
}

/// Enum holding the actual encoded data for each stage
#[derive(Debug, Clone)]
pub enum NymphEncoding {
    Larval(LarvalEncoded),
    Nymph(NymphEncoded),
    Adult(AdultEncoded),
}

impl NymphVector {
    /// Creates a new vector in nymph stage (default for insertions)
    pub fn new_nymph(id: u64, vector: &[f32]) -> Self {
        Self {
            id,
            encoding: NymphEncoding::Nymph(NymphEncoded::from_f32(vector)),
            metadata: StageMetadata::new(MetamorphicStage::Nymph),
        }
    }

    /// Creates a new vector in larval stage (for cold imports)
    pub fn new_larval(id: u64, vector: &[f32]) -> Self {
        Self {
            id,
            encoding: NymphEncoding::Larval(LarvalEncoded::from_binary(vector)),
            metadata: StageMetadata::new(MetamorphicStage::Larval),
        }
    }

    /// Creates a new vector in adult stage (for hot data)
    pub fn new_adult(id: u64, vector: Vec<f32>) -> Self {
        Self {
            id,
            encoding: NymphEncoding::Adult(AdultEncoded::from_f32(vector)),
            metadata: StageMetadata::new(MetamorphicStage::Adult),
        }
    }

    /// Promotes vector to next higher stage (if appropriate)
    pub fn promote(&mut self, vector: &[f32]) {
        if let Some(new_stage) = self.metadata.current_stage().promote() {
            self.encoding = match new_stage {
                MetamorphicStage::Nymph => NymphEncoding::Nymph(NymphEncoded::from_f32(vector)),
                MetamorphicStage::Adult => NymphEncoding::Adult(AdultEncoded::from_f32(vector.to_vec())),
                MetamorphicStage::Larval => unreachable!(),
            };
            self.metadata.record_transition(new_stage);
        }
    }

    /// Demotes vector to next lower stage (if appropriate)
    pub fn demote(&mut self, vector: &[f32]) {
        if let Some(new_stage) = self.metadata.current_stage().demote() {
            self.encoding = match new_stage {
                MetamorphicStage::Larval => NymphEncoding::Larval(LarvalEncoded::from_binary(vector)),
                MetamorphicStage::Nymph => NymphEncoding::Nymph(NymphEncoded::from_f32(vector)),
                MetamorphicStage::Adult => unreachable!(),
            };
            self.metadata.record_transition(new_stage);
        }
    }

    /// Decompresses to full precision (may be cached)
    pub fn decompress(&self) -> Vec<f32> {
        match &self.encoding {
            NymphEncoding::Larval(l) => l.decompress(),
            NymphEncoding::Nymph(n) => n.decompress(),
            NymphEncoding::Adult(a) => a.data.clone(),
        }
    }

    /// Returns current stage
    pub fn current_stage(&self) -> MetamorphicStage {
        self.metadata.current_stage()
    }

    /// Returns memory footprint in bytes
    pub fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() +
        match &self.encoding {
            NymphEncoding::Larval(l) => l.memory_bytes(),
            NymphEncoding::Nymph(n) => n.memory_bytes(),
            NymphEncoding::Adult(a) => a.memory_bytes(),
        }
    }
}

// ============================================================================
// STAGE TRANSITION POLICY
// ============================================================================

/// Configuration for stage transition thresholds and policies.
///
/// Design Rationale:
/// - Tunable thresholds for different workloads
/// - Hysteresis to prevent thrashing
/// - Time-based and count-based triggers
#[derive(Debug, Clone)]
pub struct TransitionPolicy {
    /// Promotion thresholds
    pub larval_to_nymph: AccessThreshold,
    pub nymph_to_adult: AccessThreshold,

    /// Demotion thresholds
    pub adult_to_nymph: AccessThreshold,
    pub nymph_to_larval: AccessThreshold,

    /// Time decay factor for exponential recency weighting
    pub time_decay_factor: f64,

    /// Memory pressure triggers
    pub max_adult_vectors: usize,
    pub max_nymph_vectors: usize,
}

impl Default for TransitionPolicy {
    fn default() -> Self {
        Self {
            larval_to_nymph: AccessThreshold {
                min_access_count: 10,
                min_access_frequency: 0.01, // 1 access per 100 seconds
                min_score: 5.0,
                inactivity_timeout_sec: None,
            },
            nymph_to_adult: AccessThreshold {
                min_access_count: 100,
                min_access_frequency: 1.0, // 1 access per second
                min_score: 50.0,
                inactivity_timeout_sec: None,
            },
            adult_to_nymph: AccessThreshold {
                min_access_count: 0,
                min_access_frequency: 0.0,
                min_score: 10.0, // Must drop below this score
                inactivity_timeout_sec: Some(60), // 1 minute of inactivity
            },
            nymph_to_larval: AccessThreshold {
                min_access_count: 0,
                min_access_frequency: 0.0,
                min_score: 2.0,
                inactivity_timeout_sec: Some(300), // 5 minutes of inactivity
            },
            time_decay_factor: 0.01,
            max_adult_vectors: usize::MAX / 10, // 10% of address space
            max_nymph_vectors: usize::MAX / 5,  // 20% of address space
        }
    }
}

/// Threshold configuration for stage transitions
#[derive(Debug, Clone)]
pub struct AccessThreshold {
    /// Minimum number of accesses required
    pub min_access_count: u64,

    /// Minimum access frequency (accesses per second)
    pub min_access_frequency: f64,

    /// Minimum computed access score
    pub min_score: f64,

    /// Inactivity timeout in seconds (for demotion)
    pub inactivity_timeout_sec: Option<u64>,
}

impl AccessThreshold {
    /// Checks if metadata meets this threshold for promotion
    pub fn meets_promotion_criteria(&self, metadata: &StageMetadata, config: &TransitionConfig) -> bool {
        let count = metadata.access_count() as u64;
        let score = metadata.access_score(config);

        count >= self.min_access_count && score >= self.min_score
    }

    /// Checks if metadata meets this threshold for demotion
    pub fn meets_demotion_criteria(&self, metadata: &StageMetadata, config: &TransitionConfig) -> bool {
        let score = metadata.access_score(config);
        let inactive_sec = metadata.seconds_since_last_access();

        score < self.min_score &&
        self.inactivity_timeout_sec.map_or(true, |timeout| inactive_sec >= timeout)
    }
}

/// Configuration for access score computation
#[derive(Debug, Clone)]
pub struct TransitionConfig {
    /// Weight for access frequency in score computation
    pub frequency_weight: f64,

    /// Weight for access recency in score computation
    pub recency_weight: f64,

    /// Time decay factor for exponential aging
    pub time_decay_factor: f64,
}

impl Default for TransitionConfig {
    fn default() -> Self {
        Self {
            frequency_weight: 1.0,
            recency_weight: 10.0,
            time_decay_factor: 0.01,
        }
    }
}

// ============================================================================
// NYMPH ENCODABLE TRAIT
// ============================================================================

/// Trait for types that can be encoded in metamorphic stages.
///
/// Design Rationale:
/// - Abstraction for different vector types (dense, sparse, temporal)
/// - Pluggable encoding strategies
/// - Metadata tracking for all encodable types
pub trait NymphEncodable: Send + Sync {
    /// Compresses to larval stage
    fn to_larval(&self) -> LarvalEncoded;

    /// Compresses to nymph stage
    fn to_nymph(&self) -> NymphEncoded;

    /// Converts to adult stage (no compression)
    fn to_adult(&self) -> AdultEncoded;

    /// Returns current metamorphic stage
    fn current_stage(&self) -> MetamorphicStage;

    /// Returns access pattern metadata
    fn stage_metadata(&self) -> &StageMetadata;

    /// Decompresses to full precision
    fn decompress(&self) -> Vec<f32>;
}

impl NymphEncodable for NymphVector {
    fn to_larval(&self) -> LarvalEncoded {
        let decompressed = self.decompress();
        LarvalEncoded::from_binary(&decompressed)
    }

    fn to_nymph(&self) -> NymphEncoded {
        let decompressed = self.decompress();
        NymphEncoded::from_f32(&decompressed)
    }

    fn to_adult(&self) -> AdultEncoded {
        let decompressed = self.decompress();
        AdultEncoded::from_f32(decompressed)
    }

    fn current_stage(&self) -> MetamorphicStage {
        self.metadata.current_stage()
    }

    fn stage_metadata(&self) -> &StageMetadata {
        &self.metadata
    }

    fn decompress(&self) -> Vec<f32> {
        self.decompress()
    }
}

// ============================================================================
// STAGE MANAGER - Orchestrates Transitions
// ============================================================================

/// Manages stage transitions for a collection of nymph vectors.
///
/// Design Rationale:
/// - Centralized policy enforcement
/// - Batch promotion/demotion for efficiency
/// - Memory pressure handling
#[derive(Debug)]
pub struct StageManager {
    /// Transition policy configuration
    policy: TransitionPolicy,

    /// Access score configuration
    config: TransitionConfig,

    /// Current counts per stage
    larval_count: AtomicU64,
    nymph_count: AtomicU64,
    adult_count: AtomicU64,
}

impl StageManager {
    pub fn new(policy: TransitionPolicy, config: TransitionConfig) -> Self {
        Self {
            policy,
            config,
            larval_count: AtomicU64::new(0),
            nymph_count: AtomicU64::new(0),
            adult_count: AtomicU64::new(0),
        }
    }

    /// Evaluates if a vector should be promoted
    pub fn should_promote(&self, vector: &NymphVector) -> bool {
        match vector.current_stage() {
            MetamorphicStage::Larval => {
                self.policy.larval_to_nymph.meets_promotion_criteria(&vector.metadata, &self.config)
            }
            MetamorphicStage::Nymph => {
                let count = self.adult_count.load(Ordering::Relaxed) as usize;
                count < self.policy.max_adult_vectors &&
                self.policy.nymph_to_adult.meets_promotion_criteria(&vector.metadata, &self.config)
            }
            MetamorphicStage::Adult => false,
        }
    }

    /// Evaluates if a vector should be demoted
    pub fn should_demote(&self, vector: &NymphVector) -> bool {
        match vector.current_stage() {
            MetamorphicStage::Larval => false,
            MetamorphicStage::Nymph => {
                self.policy.nymph_to_larval.meets_demotion_criteria(&vector.metadata, &self.config)
            }
            MetamorphicStage::Adult => {
                self.policy.adult_to_nymph.meets_demotion_criteria(&vector.metadata, &self.config)
            }
        }
    }

    /// Updates stage counts after a transition
    pub fn record_transition(&self, old_stage: MetamorphicStage, new_stage: MetamorphicStage) {
        match old_stage {
            MetamorphicStage::Larval => self.larval_count.fetch_sub(1, Ordering::Relaxed),
            MetamorphicStage::Nymph => self.nymph_count.fetch_sub(1, Ordering::Relaxed),
            MetamorphicStage::Adult => self.adult_count.fetch_sub(1, Ordering::Relaxed),
        };

        match new_stage {
            MetamorphicStage::Larval => self.larval_count.fetch_add(1, Ordering::Relaxed),
            MetamorphicStage::Nymph => self.nymph_count.fetch_add(1, Ordering::Relaxed),
            MetamorphicStage::Adult => self.adult_count.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// Returns current stage distribution
    pub fn stage_distribution(&self) -> (u64, u64, u64) {
        (
            self.larval_count.load(Ordering::Relaxed),
            self.nymph_count.load(Ordering::Relaxed),
            self.adult_count.load(Ordering::Relaxed),
        )
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_progression() {
        assert_eq!(MetamorphicStage::Larval.promote(), Some(MetamorphicStage::Nymph));
        assert_eq!(MetamorphicStage::Nymph.promote(), Some(MetamorphicStage::Adult));
        assert_eq!(MetamorphicStage::Adult.promote(), None);

        assert_eq!(MetamorphicStage::Larval.demote(), None);
        assert_eq!(MetamorphicStage::Nymph.demote(), Some(MetamorphicStage::Larval));
        assert_eq!(MetamorphicStage::Adult.demote(), Some(MetamorphicStage::Nymph));
    }

    #[test]
    fn test_nymph_encoding_roundtrip() {
        let vector = vec![1.0, 2.5, 3.7, -1.2, 0.0];
        let encoded = NymphEncoded::from_f32(&vector);
        let decoded = encoded.decompress();

        // Check approximate equality (quantization error)
        for (orig, dec) in vector.iter().zip(decoded.iter()) {
            assert!((orig - dec).abs() < 0.1, "Expected {}, got {}", orig, dec);
        }
    }

    #[test]
    fn test_larval_binary_compression() {
        let vector = vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let encoded = LarvalEncoded::from_binary(&vector);

        // 8 dims → 1 byte
        assert_eq!(encoded.data.len(), 1);
        assert_eq!(encoded.dimensions, 8);
    }

    #[test]
    fn test_metadata_access_tracking() {
        let metadata = StageMetadata::new(MetamorphicStage::Nymph);

        metadata.record_access(100);
        assert_eq!(metadata.access_count(), 1);

        metadata.record_access(150);
        assert_eq!(metadata.access_count(), 2);
        assert_eq!(metadata.avg_access_latency_us(), 125);
    }
}
