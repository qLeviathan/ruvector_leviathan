//! Nymph Encoding - Metamorphic Vector Compression System
//!
//! A three-stage adaptive compression system that dynamically adjusts vector precision
//! based on access patterns (temperature). Vectors "metamorphose" between stages:
//!
//! - **Larval**: Ultra-compressed (1-4 bit) - Cold, rarely accessed vectors
//! - **Nymph**: Medium compressed (8-bit) - Warm, occasionally accessed vectors
//! - **Adult**: Full precision (f32) - Hot, frequently accessed vectors
//!
//! This provides adaptive memory optimization with minimal accuracy loss for hot data.

use crate::error::{Result, RuvectorError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Vector lifecycle stage in the metamorphic encoding system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VectorStage {
    /// Ultra-compressed 1-4 bit representation (cold data)
    Larval,
    /// 8-bit scalar quantized representation (warm data)
    Nymph,
    /// Full f32 precision representation (hot data)
    Adult,
}

impl VectorStage {
    /// Get compression ratio relative to f32 (4 bytes per dimension)
    pub fn compression_ratio(&self, bits_per_dim: u8) -> f32 {
        match self {
            VectorStage::Larval => 32.0 / (bits_per_dim as f32),
            VectorStage::Nymph => 4.0,
            VectorStage::Adult => 1.0,
        }
    }

    /// Get memory size in bytes for a vector of given dimensions
    pub fn memory_size(&self, dimensions: usize, bits_per_dim: u8) -> usize {
        match self {
            VectorStage::Larval => ((dimensions * bits_per_dim as usize) + 7) / 8,
            VectorStage::Nymph => dimensions,
            VectorStage::Adult => dimensions * std::mem::size_of::<f32>(),
        }
    }
}

/// Metadata tracking access patterns for a single vector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorMetadata {
    /// Current stage of the vector
    pub stage: VectorStage,
    /// Total number of accesses
    pub access_count: u32,
    /// Timestamp of last access (seconds since UNIX epoch)
    pub last_access_time: u64,
    /// Timestamp of creation (seconds since UNIX epoch)
    pub created_time: u64,
    /// Number of accesses in the current hot window
    pub recent_access_count: u32,
    /// Start of the current hot window
    pub hot_window_start: u64,
    /// Bits per dimension for Larval stage (1-4)
    pub larval_bits: u8,
}

impl VectorMetadata {
    /// Create new metadata for a vector starting in Adult stage
    pub fn new() -> Self {
        let now = current_timestamp();
        Self {
            stage: VectorStage::Adult,
            access_count: 0,
            last_access_time: now,
            created_time: now,
            recent_access_count: 0,
            hot_window_start: now,
            larval_bits: 2,
        }
    }

    /// Record an access to this vector
    pub fn record_access(&mut self) {
        let now = current_timestamp();
        self.access_count += 1;
        self.recent_access_count += 1;
        self.last_access_time = now;
    }

    /// Get seconds since last access
    pub fn seconds_since_access(&self) -> u64 {
        current_timestamp().saturating_sub(self.last_access_time)
    }

    /// Reset the hot window counter
    pub fn reset_hot_window(&mut self) {
        self.recent_access_count = 0;
        self.hot_window_start = current_timestamp();
    }
}

impl Default for VectorMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Temperature-based policy for stage transitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperaturePolicy {
    /// Demote to Larval after this many seconds without access
    pub cold_threshold_secs: u64,
    /// Promote to Nymph after this many total accesses
    pub warm_access_count: u32,
    /// Promote to Adult after this many accesses in hot window
    pub hot_access_count: u32,
    /// Hot window duration in seconds
    pub hot_window_secs: u64,
    /// Bits per dimension for Larval stage (1-4)
    pub larval_bits: u8,
}

impl TemperaturePolicy {
    /// Create a conservative policy (favors precision)
    pub fn conservative() -> Self {
        Self {
            cold_threshold_secs: 3600, // 1 hour
            warm_access_count: 3,
            hot_access_count: 5,
            hot_window_secs: 300, // 5 minutes
            larval_bits: 4,
        }
    }

    /// Create an aggressive policy (favors compression)
    pub fn aggressive() -> Self {
        Self {
            cold_threshold_secs: 300, // 5 minutes
            warm_access_count: 10,
            hot_access_count: 20,
            hot_window_secs: 60, // 1 minute
            larval_bits: 1,
        }
    }

    /// Create a balanced policy (default)
    pub fn balanced() -> Self {
        Self {
            cold_threshold_secs: 1800, // 30 minutes
            warm_access_count: 5,
            hot_access_count: 10,
            hot_window_secs: 120, // 2 minutes
            larval_bits: 2,
        }
    }

    /// Determine the target stage for a vector based on its metadata
    pub fn evaluate_target_stage(&self, metadata: &VectorMetadata) -> VectorStage {
        let secs_since_access = metadata.seconds_since_access();
        let window_duration = current_timestamp().saturating_sub(metadata.hot_window_start);

        // Check if should be Adult (hot)
        if metadata.recent_access_count >= self.hot_access_count
            && window_duration <= self.hot_window_secs
        {
            return VectorStage::Adult;
        }

        // Check if should be Larval (cold)
        if secs_since_access >= self.cold_threshold_secs {
            return VectorStage::Larval;
        }

        // Check if should be Nymph (warm)
        if metadata.access_count >= self.warm_access_count {
            return VectorStage::Nymph;
        }

        // Default to current stage if no transition triggered
        metadata.stage
    }
}

impl Default for TemperaturePolicy {
    fn default() -> Self {
        Self::balanced()
    }
}

/// Trait for metamorphic policy implementations
pub trait MetamorphicPolicy: Send + Sync {
    /// Evaluate whether a vector should transition stages
    fn should_transition(&self, metadata: &VectorMetadata) -> Option<VectorStage>;

    /// Get the target bits per dimension for Larval stage
    fn larval_bits(&self) -> u8;
}

impl MetamorphicPolicy for TemperaturePolicy {
    fn should_transition(&self, metadata: &VectorMetadata) -> Option<VectorStage> {
        let target = self.evaluate_target_stage(metadata);
        if target != metadata.stage {
            Some(target)
        } else {
            None
        }
    }

    fn larval_bits(&self) -> u8 {
        self.larval_bits
    }
}

/// Encoded vector data in one of three stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodedVector {
    /// Larval stage: 1-4 bit per dimension
    Larval {
        data: Vec<u8>,
        min: f32,
        scale: f32,
        bits_per_dim: u8,
        dimensions: usize,
    },
    /// Nymph stage: 8-bit scalar quantized
    Nymph { data: Vec<u8>, min: f32, scale: f32 },
    /// Adult stage: full f32 precision
    Adult { data: Vec<f32> },
}

impl EncodedVector {
    /// Get the current stage
    pub fn stage(&self) -> VectorStage {
        match self {
            EncodedVector::Larval { .. } => VectorStage::Larval,
            EncodedVector::Nymph { .. } => VectorStage::Nymph,
            EncodedVector::Adult { .. } => VectorStage::Adult,
        }
    }

    /// Get the number of dimensions
    pub fn dimensions(&self) -> usize {
        match self {
            EncodedVector::Larval { dimensions, .. } => *dimensions,
            EncodedVector::Nymph { data, .. } => data.len(),
            EncodedVector::Adult { data } => data.len(),
        }
    }

    /// Decompress to full f32 precision
    pub fn decompress(&self) -> Vec<f32> {
        match self {
            EncodedVector::Larval {
                data,
                min,
                scale,
                bits_per_dim,
                dimensions,
            } => decompress_larval(data, *min, *scale, *bits_per_dim, *dimensions),
            EncodedVector::Nymph { data, min, scale } => {
                decompress_nymph(data, *min, *scale)
            }
            EncodedVector::Adult { data } => data.clone(),
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            EncodedVector::Larval { data, .. } => {
                data.len() + std::mem::size_of::<f32>() * 2 + std::mem::size_of::<u8>() * 2
            }
            EncodedVector::Nymph { data, .. } => {
                data.len() + std::mem::size_of::<f32>() * 2
            }
            EncodedVector::Adult { data } => data.len() * std::mem::size_of::<f32>(),
        }
    }
}

/// Stage transition functions
pub struct StageTransitioner {
    pub(crate) policy: Arc<dyn MetamorphicPolicy>,
}

impl StageTransitioner {
    /// Create a new transitioner with a policy
    pub fn new(policy: Arc<dyn MetamorphicPolicy>) -> Self {
        Self { policy }
    }

    /// Transition a vector from Larval to Nymph stage
    pub fn larval_to_nymph(&self, vector: &EncodedVector) -> Result<EncodedVector> {
        match vector {
            EncodedVector::Larval {
                data,
                min,
                scale,
                bits_per_dim,
                dimensions,
            } => {
                // Decompress from 1-4 bit to full precision
                let full = decompress_larval(data, *min, *scale, *bits_per_dim, *dimensions);
                // Re-compress to 8-bit
                Ok(compress_nymph(&full))
            }
            _ => Err(RuvectorError::InvalidInput(
                "Vector is not in Larval stage".into(),
            )),
        }
    }

    /// Transition a vector from Nymph to Adult stage
    pub fn nymph_to_adult(&self, vector: &EncodedVector) -> Result<EncodedVector> {
        match vector {
            EncodedVector::Nymph { data, min, scale } => {
                // Decompress to full precision
                let full = decompress_nymph(data, *min, *scale);
                Ok(EncodedVector::Adult { data: full })
            }
            _ => Err(RuvectorError::InvalidInput(
                "Vector is not in Nymph stage".into(),
            )),
        }
    }

    /// Transition a vector from Adult to Nymph stage (lossy)
    pub fn adult_to_nymph(&self, vector: &EncodedVector) -> Result<EncodedVector> {
        match vector {
            EncodedVector::Adult { data } => Ok(compress_nymph(data)),
            _ => Err(RuvectorError::InvalidInput(
                "Vector is not in Adult stage".into(),
            )),
        }
    }

    /// Transition a vector from Nymph to Larval stage (very lossy)
    pub fn nymph_to_larval(&self, vector: &EncodedVector) -> Result<EncodedVector> {
        match vector {
            EncodedVector::Nymph { data, min, scale } => {
                // Decompress to full precision first
                let full = decompress_nymph(data, *min, *scale);
                // Compress to larval with policy-defined bits
                Ok(compress_larval(&full, self.policy.larval_bits()))
            }
            _ => Err(RuvectorError::InvalidInput(
                "Vector is not in Nymph stage".into(),
            )),
        }
    }

    /// Transition a vector to a target stage
    pub fn transition_to_stage(
        &self,
        vector: &EncodedVector,
        target: VectorStage,
    ) -> Result<EncodedVector> {
        let current = vector.stage();

        if current == target {
            return Ok(vector.clone());
        }

        // Handle all possible transitions
        match (current, target) {
            (VectorStage::Larval, VectorStage::Nymph) => self.larval_to_nymph(vector),
            (VectorStage::Larval, VectorStage::Adult) => {
                let nymph = self.larval_to_nymph(vector)?;
                self.nymph_to_adult(&nymph)
            }
            (VectorStage::Nymph, VectorStage::Adult) => self.nymph_to_adult(vector),
            (VectorStage::Nymph, VectorStage::Larval) => self.nymph_to_larval(vector),
            (VectorStage::Adult, VectorStage::Nymph) => self.adult_to_nymph(vector),
            (VectorStage::Adult, VectorStage::Larval) => {
                let nymph = self.adult_to_nymph(vector)?;
                self.nymph_to_larval(&nymph)
            }
            _ => unreachable!(),
        }
    }

    /// Process batch transitions
    #[cfg(feature = "parallel")]
    pub fn batch_transition(
        &self,
        vectors: &[(String, EncodedVector, VectorMetadata)],
    ) -> Vec<(String, Result<EncodedVector>)> {
        vectors
            .par_iter()
            .map(|(id, vector, metadata)| {
                let result = if let Some(target) = self.policy.should_transition(metadata) {
                    self.transition_to_stage(vector, target)
                } else {
                    Ok(vector.clone())
                };
                (id.clone(), result)
            })
            .collect()
    }

    /// Process batch transitions (serial version)
    #[cfg(not(feature = "parallel"))]
    pub fn batch_transition(
        &self,
        vectors: &[(String, EncodedVector, VectorMetadata)],
    ) -> Vec<(String, Result<EncodedVector>)> {
        vectors
            .iter()
            .map(|(id, vector, metadata)| {
                let result = if let Some(target) = self.policy.should_transition(metadata) {
                    self.transition_to_stage(vector, target)
                } else {
                    Ok(vector.clone())
                };
                (id.clone(), result)
            })
            .collect()
    }
}

/// Metrics for the metamorphic encoding system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NymphMetrics {
    /// Number of vectors in each stage
    pub stage_distribution: HashMap<VectorStage, u64>,
    /// Total transitions in the last period
    pub transitions_per_second: f64,
    /// Memory saved (bytes)
    pub memory_saved_bytes: u64,
    /// Estimated precision loss (0.0 = no loss, 1.0 = total loss)
    pub avg_precision_loss: f32,
    /// Total number of vectors
    pub total_vectors: u64,
    /// Timestamp of last update
    pub last_update: u64,
}

impl NymphMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        let mut stage_distribution = HashMap::new();
        stage_distribution.insert(VectorStage::Larval, 0);
        stage_distribution.insert(VectorStage::Nymph, 0);
        stage_distribution.insert(VectorStage::Adult, 0);

        Self {
            stage_distribution,
            transitions_per_second: 0.0,
            memory_saved_bytes: 0,
            avg_precision_loss: 0.0,
            total_vectors: 0,
            last_update: current_timestamp(),
        }
    }

    /// Update stage distribution
    pub fn record_stage(&mut self, stage: VectorStage) {
        *self.stage_distribution.entry(stage).or_insert(0) += 1;
        self.total_vectors += 1;
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.total_vectors == 0 {
            return 1.0;
        }

        let total_vectors = self.total_vectors as f32;
        let adult_ratio = *self.stage_distribution.get(&VectorStage::Adult).unwrap_or(&0) as f32
            / total_vectors;
        let nymph_ratio = *self.stage_distribution.get(&VectorStage::Nymph).unwrap_or(&0) as f32
            / total_vectors;
        let larval_ratio =
            *self.stage_distribution.get(&VectorStage::Larval).unwrap_or(&0) as f32
                / total_vectors;

        // Weighted average: Adult=1x, Nymph=4x, Larval=16x (assuming 2-bit)
        1.0 / (adult_ratio * 1.0 + nymph_ratio * 0.25 + larval_ratio * 0.0625)
    }
}

impl Default for NymphMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Atomic metrics collector for thread-safe updates
#[derive(Debug)]
pub struct AtomicNymphMetrics {
    larval_count: AtomicU64,
    nymph_count: AtomicU64,
    adult_count: AtomicU64,
    total_transitions: AtomicU64,
    memory_saved: AtomicU64,
    last_update: AtomicU64,
}

impl AtomicNymphMetrics {
    /// Create new atomic metrics
    pub fn new() -> Self {
        Self {
            larval_count: AtomicU64::new(0),
            nymph_count: AtomicU64::new(0),
            adult_count: AtomicU64::new(0),
            total_transitions: AtomicU64::new(0),
            memory_saved: AtomicU64::new(0),
            last_update: AtomicU64::new(current_timestamp()),
        }
    }

    /// Record a stage
    pub fn record_stage(&self, stage: VectorStage) {
        match stage {
            VectorStage::Larval => self.larval_count.fetch_add(1, Ordering::Relaxed),
            VectorStage::Nymph => self.nymph_count.fetch_add(1, Ordering::Relaxed),
            VectorStage::Adult => self.adult_count.fetch_add(1, Ordering::Relaxed),
        };
    }

    /// Record a transition
    pub fn record_transition(&self) {
        self.total_transitions.fetch_add(1, Ordering::Relaxed);
    }

    /// Record memory saved
    pub fn record_memory_saved(&self, bytes: u64) {
        self.memory_saved.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Get snapshot of metrics
    pub fn snapshot(&self) -> NymphMetrics {
        let larval = self.larval_count.load(Ordering::Relaxed);
        let nymph = self.nymph_count.load(Ordering::Relaxed);
        let adult = self.adult_count.load(Ordering::Relaxed);
        let transitions = self.total_transitions.load(Ordering::Relaxed);
        let memory_saved = self.memory_saved.load(Ordering::Relaxed);
        let last_update = self.last_update.load(Ordering::Relaxed);

        let mut stage_distribution = HashMap::new();
        stage_distribution.insert(VectorStage::Larval, larval);
        stage_distribution.insert(VectorStage::Nymph, nymph);
        stage_distribution.insert(VectorStage::Adult, adult);

        let total_vectors = larval + nymph + adult;
        let elapsed = current_timestamp().saturating_sub(last_update);
        let transitions_per_second = if elapsed > 0 {
            transitions as f64 / elapsed as f64
        } else {
            0.0
        };

        NymphMetrics {
            stage_distribution,
            transitions_per_second,
            memory_saved_bytes: memory_saved,
            avg_precision_loss: 0.0, // TODO: calculate based on actual data
            total_vectors,
            last_update: current_timestamp(),
        }
    }

    /// Reset all counters
    pub fn reset(&self) {
        self.larval_count.store(0, Ordering::Relaxed);
        self.nymph_count.store(0, Ordering::Relaxed);
        self.adult_count.store(0, Ordering::Relaxed);
        self.total_transitions.store(0, Ordering::Relaxed);
        self.memory_saved.store(0, Ordering::Relaxed);
        self.last_update
            .store(current_timestamp(), Ordering::Relaxed);
    }
}

impl Default for AtomicNymphMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions for compression/decompression

/// Compress a vector to Nymph stage (8-bit scalar quantization)
fn compress_nymph(vector: &[f32]) -> EncodedVector {
    let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let scale = if (max - min).abs() < f32::EPSILON {
        1.0
    } else {
        (max - min) / 255.0
    };

    let data = vector
        .iter()
        .map(|&v| ((v - min) / scale).round().clamp(0.0, 255.0) as u8)
        .collect();

    EncodedVector::Nymph { data, min, scale }
}

/// Decompress from Nymph stage (8-bit) to full precision
fn decompress_nymph(data: &[u8], min: f32, scale: f32) -> Vec<f32> {
    data.iter()
        .map(|&v| min + (v as f32) * scale)
        .collect()
}

/// Compress a vector to Larval stage (1-4 bit quantization)
fn compress_larval(vector: &[f32], bits_per_dim: u8) -> EncodedVector {
    assert!(bits_per_dim >= 1 && bits_per_dim <= 4, "bits_per_dim must be 1-4");

    let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
    let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let max_value = (1 << bits_per_dim) - 1;
    let scale = if (max - min).abs() < f32::EPSILON {
        1.0
    } else {
        (max - min) / (max_value as f32)
    };

    // Pack multiple values into bytes
    let dimensions = vector.len();
    let values_per_byte = 8 / bits_per_dim as usize;
    let num_bytes = (dimensions + values_per_byte - 1) / values_per_byte;
    let mut data = vec![0u8; num_bytes];

    for (i, &v) in vector.iter().enumerate() {
        let quantized = ((v - min) / scale)
            .round()
            .clamp(0.0, max_value as f32) as u8;
        let byte_idx = i / values_per_byte;
        let bit_offset = (i % values_per_byte) * bits_per_dim as usize;
        data[byte_idx] |= quantized << bit_offset;
    }

    EncodedVector::Larval {
        data,
        min,
        scale,
        bits_per_dim,
        dimensions,
    }
}

/// Decompress from Larval stage (1-4 bit) to full precision
fn decompress_larval(
    data: &[u8],
    min: f32,
    scale: f32,
    bits_per_dim: u8,
    dimensions: usize,
) -> Vec<f32> {
    let values_per_byte = 8 / bits_per_dim as usize;
    let mask = (1u8 << bits_per_dim) - 1;
    let mut result = Vec::with_capacity(dimensions);

    for i in 0..dimensions {
        let byte_idx = i / values_per_byte;
        let bit_offset = (i % values_per_byte) * bits_per_dim as usize;
        let quantized = (data[byte_idx] >> bit_offset) & mask;
        result.push(min + (quantized as f32) * scale);
    }

    result
}

/// Get current timestamp in seconds since UNIX epoch
pub(crate) fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::from_secs(0))
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_compression_ratios() {
        let larval_1bit = VectorStage::Larval;
        let larval_4bit = VectorStage::Larval;
        let nymph = VectorStage::Nymph;
        let adult = VectorStage::Adult;

        assert_eq!(larval_1bit.compression_ratio(1), 32.0);
        assert_eq!(larval_4bit.compression_ratio(4), 8.0);
        assert_eq!(nymph.compression_ratio(2), 4.0);
        assert_eq!(adult.compression_ratio(2), 1.0);
    }

    #[test]
    fn test_vector_metadata() {
        let mut metadata = VectorMetadata::new();
        assert_eq!(metadata.stage, VectorStage::Adult);
        assert_eq!(metadata.access_count, 0);

        metadata.record_access();
        assert_eq!(metadata.access_count, 1);
        assert_eq!(metadata.recent_access_count, 1);
    }

    #[test]
    fn test_temperature_policy_hot() {
        let policy = TemperaturePolicy::balanced();
        let mut metadata = VectorMetadata::new();

        // Simulate hot access pattern
        for _ in 0..15 {
            metadata.record_access();
        }

        let target = policy.evaluate_target_stage(&metadata);
        assert_eq!(target, VectorStage::Adult);
    }

    #[test]
    fn test_nymph_compression() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let encoded = compress_nymph(&vector);

        match &encoded {
            EncodedVector::Nymph { data, .. } => {
                assert_eq!(data.len(), 5);
            }
            _ => panic!("Expected Nymph encoding"),
        }

        let decompressed = encoded.decompress();
        for (orig, recon) in vector.iter().zip(decompressed.iter()) {
            assert!((orig - recon).abs() < 0.1);
        }
    }

    #[test]
    fn test_larval_compression_2bit() {
        let vector = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let encoded = compress_larval(&vector, 2);

        match &encoded {
            EncodedVector::Larval {
                data,
                bits_per_dim,
                dimensions,
                ..
            } => {
                assert_eq!(*bits_per_dim, 2);
                assert_eq!(*dimensions, 8);
                // 2 bits per value, 4 values per byte
                assert_eq!(data.len(), 2);
            }
            _ => panic!("Expected Larval encoding"),
        }

        let decompressed = encoded.decompress();
        assert_eq!(decompressed.len(), 8);
        // Larval is very lossy with 2-bit
        for (orig, recon) in vector.iter().zip(decompressed.iter()) {
            assert!((orig - recon).abs() < 3.0); // Allow higher error
        }
    }

    #[test]
    fn test_larval_compression_1bit() {
        let vector = vec![0.0, 1.0, 0.5, 0.8, 0.2, 0.9, 0.1, 0.7];
        let encoded = compress_larval(&vector, 1);

        match &encoded {
            EncodedVector::Larval {
                data,
                bits_per_dim,
                dimensions,
                ..
            } => {
                assert_eq!(*bits_per_dim, 1);
                assert_eq!(*dimensions, 8);
                // 1 bit per value, 8 values per byte
                assert_eq!(data.len(), 1);
            }
            _ => panic!("Expected Larval encoding"),
        }
    }

    #[test]
    fn test_stage_transitions() {
        let policy = Arc::new(TemperaturePolicy::balanced());
        let transitioner = StageTransitioner::new(policy);

        // Start with Adult
        let adult = EncodedVector::Adult {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };

        // Adult -> Nymph
        let nymph = transitioner.adult_to_nymph(&adult).unwrap();
        assert_eq!(nymph.stage(), VectorStage::Nymph);

        // Nymph -> Larval
        let larval = transitioner.nymph_to_larval(&nymph).unwrap();
        assert_eq!(larval.stage(), VectorStage::Larval);

        // Larval -> Nymph
        let nymph2 = transitioner.larval_to_nymph(&larval).unwrap();
        assert_eq!(nymph2.stage(), VectorStage::Nymph);

        // Nymph -> Adult
        let adult2 = transitioner.nymph_to_adult(&nymph2).unwrap();
        assert_eq!(adult2.stage(), VectorStage::Adult);
    }

    #[test]
    fn test_transition_to_stage() {
        let policy = Arc::new(TemperaturePolicy::balanced());
        let transitioner = StageTransitioner::new(policy);

        let adult = EncodedVector::Adult {
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };

        // Direct Adult -> Larval (should go through Nymph)
        let larval = transitioner
            .transition_to_stage(&adult, VectorStage::Larval)
            .unwrap();
        assert_eq!(larval.stage(), VectorStage::Larval);

        // Direct Larval -> Adult (should go through Nymph)
        let adult2 = transitioner
            .transition_to_stage(&larval, VectorStage::Adult)
            .unwrap();
        assert_eq!(adult2.stage(), VectorStage::Adult);
    }

    #[test]
    fn test_nymph_metrics() {
        let mut metrics = NymphMetrics::new();
        assert_eq!(metrics.total_vectors, 0);

        metrics.record_stage(VectorStage::Adult);
        metrics.record_stage(VectorStage::Nymph);
        metrics.record_stage(VectorStage::Larval);

        assert_eq!(metrics.total_vectors, 3);
        assert_eq!(*metrics.stage_distribution.get(&VectorStage::Adult).unwrap(), 1);
        assert_eq!(*metrics.stage_distribution.get(&VectorStage::Nymph).unwrap(), 1);
        assert_eq!(*metrics.stage_distribution.get(&VectorStage::Larval).unwrap(), 1);
    }

    #[test]
    fn test_atomic_metrics() {
        let metrics = AtomicNymphMetrics::new();

        metrics.record_stage(VectorStage::Adult);
        metrics.record_stage(VectorStage::Nymph);
        metrics.record_transition();
        metrics.record_memory_saved(1024);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.total_vectors, 2);
        assert_eq!(snapshot.memory_saved_bytes, 1024);
    }

    #[test]
    fn test_compression_ratio_calculation() {
        let mut metrics = NymphMetrics::new();

        // All Adult (1x compression)
        metrics.record_stage(VectorStage::Adult);
        assert!((metrics.compression_ratio() - 1.0).abs() < 0.01);

        // All Nymph (4x compression)
        metrics = NymphMetrics::new();
        for _ in 0..10 {
            metrics.record_stage(VectorStage::Nymph);
        }
        assert!((metrics.compression_ratio() - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_memory_usage() {
        let adult = EncodedVector::Adult {
            data: vec![1.0; 128],
        };
        let nymph = compress_nymph(&vec![1.0; 128]);
        let larval = compress_larval(&vec![1.0; 128], 2);

        let adult_mem = adult.memory_usage();
        let nymph_mem = nymph.memory_usage();
        let larval_mem = larval.memory_usage();

        // Adult should use most memory
        assert!(adult_mem > nymph_mem);
        assert!(nymph_mem > larval_mem);
    }

    #[test]
    fn test_policy_evaluation() {
        let policy = TemperaturePolicy::aggressive();
        let mut metadata = VectorMetadata::new();

        // Initially should stay Adult
        assert_eq!(
            policy.evaluate_target_stage(&metadata),
            VectorStage::Adult
        );

        // After cold threshold, should become Larval
        metadata.last_access_time = current_timestamp() - 600; // 10 minutes ago
        let target = policy.evaluate_target_stage(&metadata);
        assert_eq!(target, VectorStage::Larval);
    }
}
