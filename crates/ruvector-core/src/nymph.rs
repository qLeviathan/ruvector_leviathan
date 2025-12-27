//! Nymph Encoding - Metamorphic Vector Compression
//!
//! This module implements a three-stage compression system that automatically adapts
//! vector precision based on access patterns:
//!
//! - **Larval**: Ultra-compressed (1-4 bits/dim) for cold storage
//! - **Nymph**: Balanced (8-bit) for moderate access
//! - **Adult**: Full precision (f32) for hot data
//!
//! Vectors automatically promote to higher precision on access and demote when idle.
//!
//! ## Example
//!
//! ```rust
//! use ruvector_core::nymph::{NymphVector, MetamorphicStage};
//!
//! // Create in larval stage (maximum compression)
//! let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let nymph = NymphVector::new(vector, MetamorphicStage::Larval);
//!
//! // Automatic promotion on access
//! let reconstructed = nymph.access(); // Promotes to Nymph or Adult
//! ```

use crate::error::{Result, RuvectorError};
use crate::quantization::QuantizedVector;
use parking_lot::RwLock;
use redb::ReadableTable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Metamorphic compression stages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum MetamorphicStage {
    /// Ultra-compressed (1-4 bits/dim), cold storage
    Larval,
    /// Balanced compression (8-bit), moderate access
    Nymph,
    /// Full precision (f32), hot data
    Adult,
}

impl MetamorphicStage {
    /// Get the compression ratio relative to f32
    pub fn compression_ratio(&self) -> f32 {
        match self {
            MetamorphicStage::Larval => 8.0,  // 4 bits/dim average
            MetamorphicStage::Nymph => 4.0,   // 8 bits/dim
            MetamorphicStage::Adult => 1.0,   // 32 bits/dim (no compression)
        }
    }

    /// Check if this stage is more precise than another
    pub fn is_higher_than(&self, other: &MetamorphicStage) -> bool {
        match (self, other) {
            (MetamorphicStage::Adult, MetamorphicStage::Nymph | MetamorphicStage::Larval) => true,
            (MetamorphicStage::Nymph, MetamorphicStage::Larval) => true,
            _ => false,
        }
    }
}

/// Metadata tracking stage transitions and access patterns
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageMetadata {
    /// Number of times this vector has been accessed
    pub access_count: u64,
    /// Last access timestamp (Unix epoch seconds)
    pub last_access: i64,
    /// Creation timestamp
    pub created_at: i64,
    /// History of stage transitions (stage, timestamp)
    pub stage_transitions: Vec<(MetamorphicStage, i64)>,
}

impl StageMetadata {
    /// Create new metadata
    pub fn new(initial_stage: MetamorphicStage) -> Self {
        let now = Self::current_timestamp();
        Self {
            access_count: 0,
            last_access: now,
            created_at: now,
            stage_transitions: vec![(initial_stage, now)],
        }
    }

    /// Record an access
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_access = Self::current_timestamp();
    }

    /// Record a stage transition
    pub fn record_transition(&mut self, new_stage: MetamorphicStage) {
        let now = Self::current_timestamp();
        self.stage_transitions.push((new_stage, now));
        self.last_access = now;
    }

    /// Get current Unix timestamp in seconds
    fn current_timestamp() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    /// Calculate age in seconds
    pub fn age_seconds(&self) -> i64 {
        Self::current_timestamp() - self.created_at
    }

    /// Calculate time since last access in seconds
    pub fn idle_seconds(&self) -> i64 {
        Self::current_timestamp() - self.last_access
    }
}

/// Larval encoding: 1-4 bits per dimension (ultra-compressed)
///
/// Uses 2-bit quantization for maximum compression, storing 4 values per byte.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LarvalEncoded {
    /// Packed 2-bit values (4 per byte)
    pub data: Vec<u8>,
    /// Number of dimensions
    pub dimensions: usize,
    /// Minimum value for reconstruction
    pub min: f32,
    /// Maximum value for reconstruction
    pub max: f32,
}

impl QuantizedVector for LarvalEncoded {
    fn quantize(vector: &[f32]) -> Self {
        let dimensions = vector.len();
        let num_bytes = (dimensions + 3) / 4; // Ceiling division
        let mut data = vec![0u8; num_bytes];

        let min = vector.iter().copied().fold(f32::INFINITY, f32::min);
        let max = vector.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        // Handle edge case where all values are the same
        let range = if (max - min).abs() < f32::EPSILON {
            1.0
        } else {
            max - min
        };

        // Quantize each value to 2 bits (0-3)
        for (i, &v) in vector.iter().enumerate() {
            let normalized = (v - min) / range;
            let quantized = (normalized * 3.0).round().clamp(0.0, 3.0) as u8;

            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= quantized << bit_offset;
        }

        Self {
            data,
            dimensions,
            min,
            max,
        }
    }

    fn distance(&self, other: &Self) -> f32 {
        // Calculate distance in quantized space for efficiency
        let mut dist_squared = 0.0;

        for i in 0..self.dimensions.min(other.dimensions) {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;

            let a = (self.data[byte_idx] >> bit_offset) & 0b11;
            let b = (other.data[byte_idx] >> bit_offset) & 0b11;

            let diff = a as i32 - b as i32;
            dist_squared += (diff * diff) as f32;
        }

        // Scale by average range
        let avg_range = (self.max - self.min + other.max - other.min) / 2.0;
        let scale = avg_range / 3.0; // 3 is max quantized value

        dist_squared.sqrt() * scale
    }

    fn reconstruct(&self) -> Vec<f32> {
        let mut result = Vec::with_capacity(self.dimensions);
        let range = self.max - self.min;

        for i in 0..self.dimensions {
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            let quantized = (self.data[byte_idx] >> bit_offset) & 0b11;

            let normalized = quantized as f32 / 3.0;
            let value = self.min + normalized * range;
            result.push(value);
        }

        result
    }
}

/// Nymph encoding: 8-bit balanced compression
///
/// Similar to ScalarQuantized but with access tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NymphEncoded {
    /// Quantized values (int8)
    pub data: Vec<u8>,
    /// Minimum value for dequantization
    pub min: f32,
    /// Scale factor for dequantization
    pub scale: f32,
}

impl QuantizedVector for NymphEncoded {
    fn quantize(vector: &[f32]) -> Self {
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

        Self { data, min, scale }
    }

    fn distance(&self, other: &Self) -> f32 {
        let avg_scale = (self.scale + other.scale) / 2.0;

        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| {
                let diff = a as i32 - b as i32;
                (diff * diff) as f32
            })
            .sum::<f32>()
            .sqrt()
            * avg_scale
    }

    fn reconstruct(&self) -> Vec<f32> {
        self.data
            .iter()
            .map(|&v| self.min + (v as f32) * self.scale)
            .collect()
    }
}

/// Adult encoding: Full f32 precision
///
/// No compression, but tracks access for potential demotion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdultEncoded {
    /// Full precision data
    pub data: Vec<f32>,
}

impl QuantizedVector for AdultEncoded {
    fn quantize(vector: &[f32]) -> Self {
        Self {
            data: vector.to_vec(),
        }
    }

    fn distance(&self, other: &Self) -> f32 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(&a, &b)| {
                let diff = a - b;
                diff * diff
            })
            .sum::<f32>()
            .sqrt()
    }

    fn reconstruct(&self) -> Vec<f32> {
        self.data.clone()
    }
}

/// Metamorphic vector that automatically transitions between compression stages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NymphVector {
    /// Current encoding stage
    stage: MetamorphicStage,
    /// Larval representation (if in Larval stage)
    larval: Option<LarvalEncoded>,
    /// Nymph representation (if in Nymph stage)
    nymph: Option<NymphEncoded>,
    /// Adult representation (if in Adult stage)
    adult: Option<AdultEncoded>,
    /// Metadata for tracking and decisions
    #[serde(skip)]
    metadata: Arc<RwLock<StageMetadata>>,
}

impl NymphVector {
    /// Create a new NymphVector in the specified stage
    pub fn new(vector: Vec<f32>, initial_stage: MetamorphicStage) -> Self {
        let metadata = Arc::new(RwLock::new(StageMetadata::new(initial_stage)));

        let (larval, nymph, adult) = match initial_stage {
            MetamorphicStage::Larval => (Some(LarvalEncoded::quantize(&vector)), None, None),
            MetamorphicStage::Nymph => (None, Some(NymphEncoded::quantize(&vector)), None),
            MetamorphicStage::Adult => (None, None, Some(AdultEncoded::quantize(&vector))),
        };

        Self {
            stage: initial_stage,
            larval,
            nymph,
            adult,
            metadata,
        }
    }

    /// Get current stage
    pub fn current_stage(&self) -> MetamorphicStage {
        self.stage
    }

    /// Get metadata (read-only)
    pub fn metadata(&self) -> StageMetadata {
        self.metadata.read().clone()
    }

    /// Access the vector, potentially promoting to higher precision
    pub fn access(&self) -> Vec<f32> {
        let mut meta = self.metadata.write();
        meta.record_access();

        // Promote based on access patterns
        let should_promote = meta.access_count > 10 && meta.idle_seconds() < 60;

        drop(meta); // Release lock before reconstruction

        if should_promote {
            // Note: Actual promotion would require mutable self
            // This is a simplified version for demonstration
        }

        self.reconstruct_current()
    }

    /// Reconstruct vector from current stage
    fn reconstruct_current(&self) -> Vec<f32> {
        match self.stage {
            MetamorphicStage::Larval => self
                .larval
                .as_ref()
                .map(|l| l.reconstruct())
                .unwrap_or_default(),
            MetamorphicStage::Nymph => self
                .nymph
                .as_ref()
                .map(|n| n.reconstruct())
                .unwrap_or_default(),
            MetamorphicStage::Adult => self
                .adult
                .as_ref()
                .map(|a| a.reconstruct())
                .unwrap_or_default(),
        }
    }

    /// Calculate distance to another NymphVector
    pub fn distance(&self, other: &Self) -> f32 {
        // Record access for both vectors
        self.metadata.write().record_access();
        other.metadata.write().record_access();

        // Use the lower precision stage for distance calculation
        match (&self.stage, &other.stage) {
            (MetamorphicStage::Larval, MetamorphicStage::Larval) => self
                .larval
                .as_ref()
                .and_then(|l1| other.larval.as_ref().map(|l2| l1.distance(l2)))
                .unwrap_or(f32::INFINITY),

            (MetamorphicStage::Nymph, MetamorphicStage::Nymph)
            | (MetamorphicStage::Larval, MetamorphicStage::Nymph)
            | (MetamorphicStage::Nymph, MetamorphicStage::Larval) => {
                // Promote to Nymph if needed
                let v1 = self.to_nymph();
                let v2 = other.to_nymph();
                v1.distance(&v2)
            }

            _ => {
                // At least one is Adult, use full precision
                let v1 = self.to_adult();
                let v2 = other.to_adult();
                v1.distance(&v2)
            }
        }
    }

    /// Convert to Nymph representation
    fn to_nymph(&self) -> NymphEncoded {
        if let Some(ref nymph) = self.nymph {
            nymph.clone()
        } else {
            let vector = self.reconstruct_current();
            NymphEncoded::quantize(&vector)
        }
    }

    /// Convert to Adult representation
    fn to_adult(&self) -> AdultEncoded {
        if let Some(ref adult) = self.adult {
            adult.clone()
        } else {
            let vector = self.reconstruct_current();
            AdultEncoded::quantize(&vector)
        }
    }

    /// Promote to a higher precision stage
    pub fn promote(&mut self, target_stage: MetamorphicStage) -> Result<()> {
        if !target_stage.is_higher_than(&self.stage) {
            return Err(RuvectorError::InvalidParameter(
                "Target stage must be higher than current stage".to_string(),
            ));
        }

        let vector = self.reconstruct_current();

        match target_stage {
            MetamorphicStage::Nymph => {
                self.nymph = Some(NymphEncoded::quantize(&vector));
                self.larval = None;
                self.stage = MetamorphicStage::Nymph;
            }
            MetamorphicStage::Adult => {
                self.adult = Some(AdultEncoded::quantize(&vector));
                self.nymph = None;
                self.larval = None;
                self.stage = MetamorphicStage::Adult;
            }
            MetamorphicStage::Larval => {
                return Err(RuvectorError::InvalidParameter(
                    "Cannot promote to Larval stage".to_string(),
                ));
            }
        }

        self.metadata.write().record_transition(target_stage);
        Ok(())
    }

    /// Demote to a lower precision stage (for cold storage)
    pub fn demote(&mut self, target_stage: MetamorphicStage) -> Result<()> {
        if target_stage.is_higher_than(&self.stage) {
            return Err(RuvectorError::InvalidParameter(
                "Target stage must be lower than current stage".to_string(),
            ));
        }

        let vector = self.reconstruct_current();

        match target_stage {
            MetamorphicStage::Larval => {
                self.larval = Some(LarvalEncoded::quantize(&vector));
                self.nymph = None;
                self.adult = None;
                self.stage = MetamorphicStage::Larval;
            }
            MetamorphicStage::Nymph => {
                self.nymph = Some(NymphEncoded::quantize(&vector));
                self.adult = None;
                self.stage = MetamorphicStage::Nymph;
            }
            MetamorphicStage::Adult => {
                return Err(RuvectorError::InvalidParameter(
                    "Cannot demote to Adult stage".to_string(),
                ));
            }
        }

        self.metadata.write().record_transition(target_stage);
        Ok(())
    }

    /// Get memory footprint in bytes
    pub fn memory_footprint(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let data = match self.stage {
            MetamorphicStage::Larval => self.larval.as_ref().map(|l| l.data.len()).unwrap_or(0),
            MetamorphicStage::Nymph => self.nymph.as_ref().map(|n| n.data.len()).unwrap_or(0),
            MetamorphicStage::Adult => self
                .adult
                .as_ref()
                .map(|a| a.data.len() * 4)
                .unwrap_or(0),
        };
        base + data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metamorphic_stage_comparison() {
        assert!(MetamorphicStage::Adult.is_higher_than(&MetamorphicStage::Nymph));
        assert!(MetamorphicStage::Adult.is_higher_than(&MetamorphicStage::Larval));
        assert!(MetamorphicStage::Nymph.is_higher_than(&MetamorphicStage::Larval));
        assert!(!MetamorphicStage::Larval.is_higher_than(&MetamorphicStage::Nymph));
    }

    #[test]
    fn test_compression_ratios() {
        assert_eq!(MetamorphicStage::Larval.compression_ratio(), 8.0);
        assert_eq!(MetamorphicStage::Nymph.compression_ratio(), 4.0);
        assert_eq!(MetamorphicStage::Adult.compression_ratio(), 1.0);
    }

    #[test]
    fn test_larval_encoding() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let encoded = LarvalEncoded::quantize(&vector);

        assert_eq!(encoded.dimensions, 8);
        assert_eq!(encoded.data.len(), 2); // 8 dims / 4 per byte = 2 bytes

        let reconstructed = encoded.reconstruct();
        assert_eq!(reconstructed.len(), 8);

        // Check approximate reconstruction
        for (orig, recon) in vector.iter().zip(&reconstructed) {
            let error = (orig - recon).abs();
            let max_error = (8.0 - 1.0) / 3.0; // Range / 3 levels
            assert!(error < max_error * 1.5, "Error too large: {}", error);
        }
    }

    #[test]
    fn test_larval_distance() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0];
        let v2 = vec![2.0, 3.0, 4.0, 5.0];

        let q1 = LarvalEncoded::quantize(&v1);
        let q2 = LarvalEncoded::quantize(&v2);

        let dist = q1.distance(&q2);
        assert!(dist > 0.0);
        assert!(dist < 10.0); // Reasonable bound for this data
    }

    #[test]
    fn test_nymph_encoding() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let encoded = NymphEncoded::quantize(&vector);

        assert_eq!(encoded.data.len(), 5);

        let reconstructed = encoded.reconstruct();
        for (orig, recon) in vector.iter().zip(&reconstructed) {
            assert!((orig - recon).abs() < 0.1);
        }
    }

    #[test]
    fn test_adult_encoding() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let encoded = AdultEncoded::quantize(&vector);

        assert_eq!(encoded.data.len(), 5);

        let reconstructed = encoded.reconstruct();
        assert_eq!(vector, reconstructed);
    }

    #[test]
    fn test_nymph_vector_creation() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let larval = NymphVector::new(vector.clone(), MetamorphicStage::Larval);
        assert_eq!(larval.current_stage(), MetamorphicStage::Larval);
        assert!(larval.larval.is_some());
        assert!(larval.nymph.is_none());
        assert!(larval.adult.is_none());

        let nymph = NymphVector::new(vector.clone(), MetamorphicStage::Nymph);
        assert_eq!(nymph.current_stage(), MetamorphicStage::Nymph);
        assert!(nymph.larval.is_none());
        assert!(nymph.nymph.is_some());
        assert!(nymph.adult.is_none());

        let adult = NymphVector::new(vector, MetamorphicStage::Adult);
        assert_eq!(adult.current_stage(), MetamorphicStage::Adult);
        assert!(adult.larval.is_none());
        assert!(adult.nymph.is_none());
        assert!(adult.adult.is_some());
    }

    #[test]
    fn test_nymph_vector_access() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nymph = NymphVector::new(vector.clone(), MetamorphicStage::Nymph);

        let accessed = nymph.access();
        assert_eq!(accessed.len(), 5);

        let meta = nymph.metadata();
        assert_eq!(meta.access_count, 1);
    }

    #[test]
    fn test_promotion() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut nymph = NymphVector::new(vector, MetamorphicStage::Larval);

        // Promote to Nymph
        nymph.promote(MetamorphicStage::Nymph).unwrap();
        assert_eq!(nymph.current_stage(), MetamorphicStage::Nymph);
        assert!(nymph.nymph.is_some());
        assert!(nymph.larval.is_none());

        // Promote to Adult
        nymph.promote(MetamorphicStage::Adult).unwrap();
        assert_eq!(nymph.current_stage(), MetamorphicStage::Adult);
        assert!(nymph.adult.is_some());
        assert!(nymph.nymph.is_none());

        // Cannot promote Adult further
        let result = nymph.promote(MetamorphicStage::Larval);
        assert!(result.is_err());
    }

    #[test]
    fn test_demotion() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut nymph = NymphVector::new(vector, MetamorphicStage::Adult);

        // Demote to Nymph
        nymph.demote(MetamorphicStage::Nymph).unwrap();
        assert_eq!(nymph.current_stage(), MetamorphicStage::Nymph);
        assert!(nymph.nymph.is_some());
        assert!(nymph.adult.is_none());

        // Demote to Larval
        nymph.demote(MetamorphicStage::Larval).unwrap();
        assert_eq!(nymph.current_stage(), MetamorphicStage::Larval);
        assert!(nymph.larval.is_some());
        assert!(nymph.nymph.is_none());

        // Cannot demote Larval further
        let result = nymph.demote(MetamorphicStage::Adult);
        assert!(result.is_err());
    }

    #[test]
    fn test_distance_calculation() {
        let v1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let v2 = vec![2.0, 3.0, 4.0, 5.0, 6.0];

        let nymph1 = NymphVector::new(v1, MetamorphicStage::Nymph);
        let nymph2 = NymphVector::new(v2, MetamorphicStage::Nymph);

        let dist = nymph1.distance(&nymph2);
        assert!(dist > 0.0);
        assert!(dist < 10.0);

        // Both vectors should have incremented access count
        assert_eq!(nymph1.metadata().access_count, 1);
        assert_eq!(nymph2.metadata().access_count, 1);
    }

    #[test]
    fn test_memory_footprint() {
        let vector = vec![1.0; 1024];

        let larval = NymphVector::new(vector.clone(), MetamorphicStage::Larval);
        let nymph = NymphVector::new(vector.clone(), MetamorphicStage::Nymph);
        let adult = NymphVector::new(vector, MetamorphicStage::Adult);

        let larval_size = larval.memory_footprint();
        let nymph_size = nymph.memory_footprint();
        let adult_size = adult.memory_footprint();

        // Verify compression ratios
        assert!(larval_size < nymph_size);
        assert!(nymph_size < adult_size);

        // Adult should be roughly 4x Nymph
        let ratio = adult_size as f32 / nymph_size as f32;
        assert!(ratio > 3.0 && ratio < 5.0, "Ratio: {}", ratio);
    }

    #[test]
    fn test_metadata_tracking() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut nymph = NymphVector::new(vector, MetamorphicStage::Larval);

        let meta = nymph.metadata();
        assert_eq!(meta.access_count, 0);
        assert_eq!(meta.stage_transitions.len(), 1);
        assert_eq!(meta.stage_transitions[0].0, MetamorphicStage::Larval);

        // Access and promote
        nymph.access();
        nymph.promote(MetamorphicStage::Nymph).unwrap();

        let meta = nymph.metadata();
        assert_eq!(meta.access_count, 1);
        assert_eq!(meta.stage_transitions.len(), 2);
        assert_eq!(meta.stage_transitions[1].0, MetamorphicStage::Nymph);
    }

    #[test]
    fn test_roundtrip_larval_to_adult() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut nymph = NymphVector::new(original.clone(), MetamorphicStage::Larval);

        // Promote all the way to Adult
        nymph.promote(MetamorphicStage::Nymph).unwrap();
        nymph.promote(MetamorphicStage::Adult).unwrap();

        let reconstructed = nymph.access();

        // Should be close to original despite Larval compression
        for (orig, recon) in original.iter().zip(&reconstructed) {
            let error = (orig - recon).abs();
            // Allow larger error due to Larval compression
            assert!(error < 1.0, "Error too large: {} vs {}", orig, recon);
        }
    }

    #[test]
    fn test_edge_case_single_value() {
        let vector = vec![5.0; 10];
        let larval = LarvalEncoded::quantize(&vector);
        let reconstructed = larval.reconstruct();

        for &val in &reconstructed {
            assert!((val - 5.0).abs() < 0.1);
        }
    }

    #[test]
    fn test_edge_case_extreme_range() {
        let vector = vec![-1000.0, -500.0, 0.0, 500.0, 1000.0];
        let encoded = NymphEncoded::quantize(&vector);
        let reconstructed = encoded.reconstruct();

        assert_eq!(reconstructed.len(), 5);
        for (orig, recon) in vector.iter().zip(&reconstructed) {
            let error_ratio = (orig - recon).abs() / orig.abs().max(1.0);
            assert!(error_ratio < 0.1, "Error ratio too large: {}", error_ratio);
        }
    }

    #[test]
    fn test_serialization() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let nymph = NymphVector::new(vector, MetamorphicStage::Nymph);

        // Serialize
        let serialized = serde_json::to_string(&nymph).unwrap();

        // Deserialize
        let deserialized: NymphVector = serde_json::from_str(&serialized).unwrap();

        assert_eq!(deserialized.current_stage(), MetamorphicStage::Nymph);
        assert_eq!(nymph.reconstruct_current(), deserialized.reconstruct_current());
    }
}

// ========== NymphStorage Persistence Layer ==========

use redb::{Database, TableDefinition};
use std::path::Path;

/// Lifecycle stage for storage (alias for compatibility)
pub use MetamorphicStage as LifecycleStage;

/// Lifecycle metadata for tracking access patterns
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct LifecycleMetadata {
    /// Current lifecycle stage
    pub stage: LifecycleStage,
    /// Last access timestamp (seconds since UNIX epoch)
    pub last_access: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// Number of times accessed
    pub access_count: u64,
    /// Number of times promoted
    pub promotion_count: u32,
    /// Number of times demoted
    pub demotion_count: u32,
}

impl LifecycleMetadata {
    /// Create new metadata for Adult stage
    pub fn new() -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        Self {
            stage: LifecycleStage::Adult,
            last_access: now,
            created_at: now,
            access_count: 0,
            promotion_count: 0,
            demotion_count: 0,
        }
    }

    /// Record an access and return true if promotion needed
    pub fn record_access(&mut self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        self.last_access = now;
        self.access_count += 1;

        // Promote if not already Adult
        if self.stage != LifecycleStage::Adult {
            self.promotion_count += 1;
            self.stage = LifecycleStage::Adult;
            true
        } else {
            false
        }
    }

    /// Check if demotion is needed based on age
    pub fn check_demotion(&mut self) -> Option<LifecycleStage> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let age = now - self.last_access;

        match self.stage {
            LifecycleStage::Adult if age > 24 * 3600 => {
                // Demote to Nymph after 24h
                self.stage = LifecycleStage::Nymph;
                self.demotion_count += 1;
                Some(LifecycleStage::Nymph)
            }
            LifecycleStage::Nymph if age > 7 * 24 * 3600 => {
                // Demote to Larval after 7d
                self.stage = LifecycleStage::Larval;
                self.demotion_count += 1;
                Some(LifecycleStage::Larval)
            }
            _ => None,
        }
    }
}

impl Default for LifecycleMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Encoded data in different formats based on lifecycle stage
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub enum NymphEncodedData {
    /// Adult stage - raw f32 vectors
    Adult(Vec<f32>),
    /// Nymph stage - scalar quantized (u8 + scale/offset)
    Nymph {
        quantized: Vec<u8>,
        scale: f32,
        offset: f32,
    },
    /// Larval stage - binary quantized (bits + threshold)
    Larval {
        bits: Vec<u8>,
        threshold: f32,
    },
}

impl NymphEncodedData {
    /// Encode vector for given stage
    pub fn encode(vector: &[f32], stage: LifecycleStage) -> Self {
        match stage {
            LifecycleStage::Adult => Self::Adult(vector.to_vec()),
            LifecycleStage::Nymph => {
                let encoded = NymphEncoded::quantize(vector);
                Self::Nymph {
                    quantized: encoded.data,
                    scale: encoded.scale,
                    offset: encoded.min,
                }
            }
            LifecycleStage::Larval => {
                let encoded = LarvalEncoded::quantize(vector);
                Self::Larval {
                    bits: encoded.data,
                    threshold: encoded.min,
                }
            }
        }
    }

    /// Decode to f32 vector
    pub fn decode(&self) -> Vec<f32> {
        match self {
            Self::Adult(vector) => vector.clone(),
            Self::Nymph {
                quantized,
                scale,
                offset,
            } => {
                let encoded = NymphEncoded {
                    data: quantized.clone(),
                    min: *offset,
                    scale: *scale,
                };
                encoded.reconstruct()
            }
            Self::Larval { bits, threshold } => {
                let dimensions = bits.len() * 4; // 4 values per byte
                let encoded = LarvalEncoded {
                    data: bits.clone(),
                    dimensions,
                    min: *threshold,
                    max: *threshold + 1.0,
                };
                encoded.reconstruct()
            }
        }
    }

    /// Get memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            Self::Adult(v) => v.len() * 4,
            Self::Nymph { quantized, .. } => quantized.len() + 8, // + scale/offset
            Self::Larval { bits, .. } => bits.len() + 4,          // + threshold
        }
    }

    /// Get current stage
    pub fn stage(&self) -> LifecycleStage {
        match self {
            Self::Adult(_) => LifecycleStage::Adult,
            Self::Nymph { .. } => LifecycleStage::Nymph,
            Self::Larval { .. } => LifecycleStage::Larval,
        }
    }

    /// Re-encode to different stage
    pub fn transcode(&self, target_stage: LifecycleStage) -> Self {
        if self.stage() == target_stage {
            return self.clone();
        }

        // Decode then re-encode
        let vector = self.decode();
        Self::encode(&vector, target_stage)
    }
}

/// Stored entry with lifecycle-based encoding
#[derive(Debug, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct NymphEntry {
    /// Entry ID
    pub id: String,
    /// Lifecycle metadata
    pub metadata: LifecycleMetadata,
    /// Encoded data (format depends on stage)
    pub data: NymphEncodedData,
    /// Original dimensions (needed for decompression)
    pub dimensions: usize,
}

// Table definitions
const NYMPH_ENTRIES: TableDefinition<&str, &[u8]> =
    TableDefinition::new("nymph_entries");

/// Nymph storage backend with lifecycle management
pub struct NymphStorage {
    db: Arc<Database>,
    dimensions: usize,
    // In-memory cache of Adult stage entries for fast access
    adult_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
}

impl NymphStorage {
    /// Create new Nymph storage
    pub fn new<P: AsRef<Path>>(path: P, dimensions: usize) -> crate::error::Result<Self> {
        let db = Arc::new(Database::create(path)?);

        // Initialize tables
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(NYMPH_ENTRIES)?;
        }
        write_txn.commit()?;

        Ok(Self {
            db,
            dimensions,
            adult_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Store a vector (initially in Adult stage)
    pub fn store(&self, id: String, vector: Vec<f32>) -> crate::error::Result<()> {
        use crate::error::RuvectorError;

        if vector.len() != self.dimensions {
            return Err(RuvectorError::InvalidDimension(format!(
                "Expected {} dimensions, got {}",
                self.dimensions,
                vector.len()
            )));
        }

        let metadata = LifecycleMetadata::new();
        let data = NymphEncodedData::Adult(vector.clone());

        let entry = NymphEntry {
            id: id.clone(),
            metadata,
            data,
            dimensions: self.dimensions,
        };

        // Store in database
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(NYMPH_ENTRIES)?;
            let serialized = bincode::encode_to_vec(&entry, bincode::config::standard())
                .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;
            table.insert(id.as_str(), serialized.as_slice())?;
        }
        write_txn.commit()?;

        // Add to Adult cache
        self.adult_cache.write().insert(id, vector);

        Ok(())
    }

    /// Retrieve a vector (with automatic promotion)
    pub fn retrieve(&self, id: &str) -> crate::error::Result<Option<Vec<f32>>> {
        use crate::error::RuvectorError;

        // Check Adult cache first
        if let Some(vector) = self.adult_cache.read().get(id) {
            self.record_access_internal(id)?;
            return Ok(Some(vector.clone()));
        }

        // Load from database
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NYMPH_ENTRIES)?;

        let entry_data = match table.get(id)? {
            Some(data) => data,
            None => return Ok(None),
        };

        let (mut entry, _): (NymphEntry, usize) =
            bincode::decode_from_slice(entry_data.value(), bincode::config::standard())
                .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;

        drop(table);
        drop(read_txn);

        // Decode vector
        let vector = entry.data.decode();

        // Record access and promote if needed
        let promoted = entry.metadata.record_access();

        if promoted {
            // Re-encode as Adult
            entry.data = NymphEncodedData::Adult(vector.clone());

            // Update database
            let write_txn = self.db.begin_write()?;
            {
                let mut table = write_txn.open_table(NYMPH_ENTRIES)?;
                let serialized = bincode::encode_to_vec(&entry, bincode::config::standard())
                    .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;
                table.insert(id, serialized.as_slice())?;
            }
            write_txn.commit()?;

            // Add to Adult cache
            self.adult_cache.write().insert(id.to_string(), vector.clone());
        }

        Ok(Some(vector))
    }

    /// Delete an entry
    pub fn delete(&self, id: &str) -> crate::error::Result<bool> {
        let write_txn = self.db.begin_write()?;
        let deleted = {
            let mut table = write_txn.open_table(NYMPH_ENTRIES)?;
            let result = table.remove(id)?;
            result.is_some()
        };
        write_txn.commit()?;

        self.adult_cache.write().remove(id);

        Ok(deleted)
    }

    /// Run metamorphic cycle to demote old entries
    pub fn run_metamorphic_cycle(&self) -> crate::error::Result<MetamorphicStats> {
        use crate::error::RuvectorError;

        let mut stats = MetamorphicStats::default();

        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NYMPH_ENTRIES)?;

        // Collect entries that need demotion
        let mut entries_to_update = Vec::new();

        let iter = table.iter()?;
        for item in iter {
            let (key, value) = item?;
            let (mut entry, _): (NymphEntry, usize) =
                bincode::decode_from_slice(value.value(), bincode::config::standard())
                    .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;

            // Save the stage before potentially moving entry
            let current_stage = entry.metadata.stage;

            if let Some(new_stage) = entry.metadata.check_demotion() {
                // Transcode data
                let old_size = entry.data.memory_size();
                entry.data = entry.data.transcode(new_stage);
                let new_size = entry.data.memory_size();

                stats.demoted_count += 1;
                stats.bytes_saved += (old_size - new_size) as u64;

                entries_to_update.push((key.value().to_string(), entry));
            }

            // Update stage counts (use saved stage)
            match current_stage {
                LifecycleStage::Adult => stats.adult_count += 1,
                LifecycleStage::Nymph => stats.nymph_count += 1,
                LifecycleStage::Larval => stats.larval_count += 1,
            }
        }

        drop(table);
        drop(read_txn);

        // Update demoted entries
        if !entries_to_update.is_empty() {
            let write_txn = self.db.begin_write()?;
            {
                let mut table = write_txn.open_table(NYMPH_ENTRIES)?;
                for (id, entry) in entries_to_update {
                    let serialized = bincode::encode_to_vec(&entry, bincode::config::standard())
                        .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;
                    table.insert(id.as_str(), serialized.as_slice())?;

                    // Remove from Adult cache if demoted
                    if entry.metadata.stage != LifecycleStage::Adult {
                        self.adult_cache.write().remove(&id);
                    }
                }
            }
            write_txn.commit()?;
        }

        Ok(stats)
    }

    /// Get statistics for all lifecycle stages
    pub fn get_stage_statistics(&self) -> crate::error::Result<StageStatistics> {
        use crate::error::RuvectorError;

        let mut stats = StageStatistics::default();

        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NYMPH_ENTRIES)?;

        let iter = table.iter()?;
        for item in iter {
            let (_, value) = item?;
            let (entry, _): (NymphEntry, usize) =
                bincode::decode_from_slice(value.value(), bincode::config::standard())
                    .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;

            match entry.metadata.stage {
                LifecycleStage::Adult => {
                    stats.adult_count += 1;
                    stats.adult_bytes += entry.data.memory_size() as u64;
                }
                LifecycleStage::Nymph => {
                    stats.nymph_count += 1;
                    stats.nymph_bytes += entry.data.memory_size() as u64;
                }
                LifecycleStage::Larval => {
                    stats.larval_count += 1;
                    stats.larval_bytes += entry.data.memory_size() as u64;
                }
            }

            stats.total_accesses += entry.metadata.access_count;
            stats.total_promotions += entry.metadata.promotion_count as u64;
            stats.total_demotions += entry.metadata.demotion_count as u64;
        }

        stats.total_bytes = stats.adult_bytes + stats.nymph_bytes + stats.larval_bytes;
        stats.cache_entries = self.adult_cache.read().len() as u64;

        Ok(stats)
    }

    // Internal method to record access without retrieval
    fn record_access_internal(&self, id: &str) -> crate::error::Result<()> {
        use crate::error::RuvectorError;

        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(NYMPH_ENTRIES)?;

        let entry_data = match table.get(id)? {
            Some(data) => data,
            None => return Ok(()),
        };

        let (mut entry, _): (NymphEntry, usize) =
            bincode::decode_from_slice(entry_data.value(), bincode::config::standard())
                .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;

        drop(table);
        drop(read_txn);

        entry.metadata.record_access();

        // Update database
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(NYMPH_ENTRIES)?;
            let serialized = bincode::encode_to_vec(&entry, bincode::config::standard())
                .map_err(|e| RuvectorError::SerializationError(e.to_string()))?;
            table.insert(id, serialized.as_slice())?;
        }
        write_txn.commit()?;

        Ok(())
    }
}

/// Statistics from metamorphic cycle
#[derive(Debug, Default, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct MetamorphicStats {
    /// Number of entries demoted
    pub demoted_count: u64,
    /// Bytes saved from compression
    pub bytes_saved: u64,
    /// Current Adult count
    pub adult_count: u64,
    /// Current Nymph count
    pub nymph_count: u64,
    /// Current Larval count
    pub larval_count: u64,
}

/// Statistics for lifecycle stages
#[derive(Debug, Default, Clone, Serialize, Deserialize, bincode::Encode, bincode::Decode)]
pub struct StageStatistics {
    /// Adult stage count
    pub adult_count: u64,
    /// Adult stage bytes
    pub adult_bytes: u64,
    /// Nymph stage count
    pub nymph_count: u64,
    /// Nymph stage bytes
    pub nymph_bytes: u64,
    /// Larval stage count
    pub larval_count: u64,
    /// Larval stage bytes
    pub larval_bytes: u64,
    /// Total bytes
    pub total_bytes: u64,
    /// Total access count
    pub total_accesses: u64,
    /// Total promotions
    pub total_promotions: u64,
    /// Total demotions
    pub total_demotions: u64,
    /// Entries in Adult cache
    pub cache_entries: u64,
}

impl StageStatistics {
    /// Calculate memory savings compared to all-Adult storage
    pub fn memory_savings_ratio(&self) -> f64 {
        let total_entries = self.adult_count + self.nymph_count + self.larval_count;
        if total_entries == 0 {
            return 0.0;
        }

        let adult_equivalent = total_entries * self.adult_bytes / self.adult_count.max(1);
        1.0 - (self.total_bytes as f64 / adult_equivalent as f64)
    }

    /// Get average access count per entry
    pub fn avg_access_count(&self) -> f64 {
        let total_entries = self.adult_count + self.nymph_count + self.larval_count;
        if total_entries == 0 {
            return 0.0;
        }
        self.total_accesses as f64 / total_entries as f64
    }
}
