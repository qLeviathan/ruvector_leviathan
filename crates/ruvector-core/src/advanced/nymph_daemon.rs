//! Nymph Encoding Background Daemon
//!
//! Async metamorphosis daemon for managing stage transitions in the background.
//! Implements priority queues, batch operations, and garbage collection.

use super::nymph_encoding::{
    current_timestamp, AtomicNymphMetrics, EncodedVector, MetamorphicPolicy, StageTransitioner,
    VectorMetadata, VectorStage,
};
use crate::error::Result;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

#[cfg(feature = "nymph-async")]
use tokio::sync::RwLock;
#[cfg(not(feature = "nymph-async"))]
use parking_lot::RwLock;

/// Priority for stage transitions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionPriority {
    /// Urgent: hot data needs promotion
    Urgent,
    /// High: warm data benefits from adjustment
    High,
    /// Normal: routine optimization
    Normal,
    /// Low: cold data can wait
    Low,
}

impl TransitionPriority {
    /// Convert to numeric value for ordering
    fn as_u8(&self) -> u8 {
        match self {
            TransitionPriority::Urgent => 3,
            TransitionPriority::High => 2,
            TransitionPriority::Normal => 1,
            TransitionPriority::Low => 0,
        }
    }
}

/// A transition task in the priority queue
#[derive(Debug, Clone)]
struct TransitionTask {
    vector_id: String,
    current_stage: VectorStage,
    target_stage: VectorStage,
    priority: TransitionPriority,
    timestamp: u64,
}

impl PartialEq for TransitionTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.timestamp == other.timestamp
    }
}

impl Eq for TransitionTask {}

impl PartialOrd for TransitionTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TransitionTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority comes first, then older tasks
        match self.priority.as_u8().cmp(&other.priority.as_u8()) {
            Ordering::Equal => other.timestamp.cmp(&self.timestamp),
            other => other,
        }
    }
}

/// Configuration for the metamorphosis daemon
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Interval between scan cycles (seconds)
    pub scan_interval_secs: u64,
    /// Maximum transitions per batch
    pub max_batch_size: usize,
    /// Enable aggressive garbage collection
    pub aggressive_gc: bool,
    /// Maximum age for transition history (seconds)
    pub history_max_age_secs: u64,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            scan_interval_secs: 60,
            max_batch_size: 100,
            aggressive_gc: false,
            history_max_age_secs: 3600,
        }
    }
}

/// Background daemon for metamorphic transformations
pub struct MetamorphosisDaemon {
    config: DaemonConfig,
    transitioner: Arc<StageTransitioner>,
    metrics: Arc<AtomicNymphMetrics>,
    priority_queue: Arc<RwLock<BinaryHeap<TransitionTask>>>,
    transition_history: Arc<RwLock<HashMap<String, Vec<TransitionRecord>>>>,
}

/// Record of a stage transition
#[derive(Debug, Clone)]
struct TransitionRecord {
    from_stage: VectorStage,
    to_stage: VectorStage,
    timestamp: u64,
    success: bool,
}

impl MetamorphosisDaemon {
    /// Create a new daemon
    pub fn new(
        config: DaemonConfig,
        policy: Arc<dyn MetamorphicPolicy>,
        metrics: Arc<AtomicNymphMetrics>,
    ) -> Self {
        Self {
            config,
            transitioner: Arc::new(StageTransitioner::new(policy)),
            metrics,
            priority_queue: Arc::new(RwLock::new(BinaryHeap::new())),
            transition_history: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Scan vectors and queue transitions
    #[cfg(feature = "nymph-async")]
    pub async fn scan_and_queue(
        &self,
        vectors: &[(String, EncodedVector, VectorMetadata)],
    ) -> Result<usize> {
        let mut queue = self.priority_queue.write().await;
        let mut queued = 0;

        for (id, vector, metadata) in vectors {
            if let Some(target) = self.transitioner.policy.should_transition(metadata) {
                let priority = self.determine_priority(metadata, target);
                let task = TransitionTask {
                    vector_id: id.clone(),
                    current_stage: vector.stage(),
                    target_stage: target,
                    priority,
                    timestamp: metadata.last_access_time,
                };
                queue.push(task);
                queued += 1;
            }
        }

        Ok(queued)
    }

    /// Scan vectors and queue transitions (sync version)
    #[cfg(not(feature = "nymph-async"))]
    pub fn scan_and_queue(
        &self,
        vectors: &[(String, EncodedVector, VectorMetadata)],
    ) -> Result<usize> {
        let mut queue = self.priority_queue.write();
        let mut queued = 0;

        for (id, vector, metadata) in vectors {
            if let Some(target) = self.transitioner.policy.should_transition(metadata) {
                let priority = self.determine_priority(metadata, target);
                let task = TransitionTask {
                    vector_id: id.clone(),
                    current_stage: vector.stage(),
                    target_stage: target,
                    priority,
                    timestamp: metadata.last_access_time,
                };
                queue.push(task);
                queued += 1;
            }
        }

        Ok(queued)
    }

    /// Process a batch of transitions from the priority queue
    #[cfg(feature = "nymph-async")]
    pub async fn process_batch(
        &self,
        vectors: &mut HashMap<String, (EncodedVector, VectorMetadata)>,
    ) -> Result<usize> {
        let mut queue = self.priority_queue.write().await;
        let batch_size = self.config.max_batch_size.min(queue.len());

        if batch_size == 0 {
            return Ok(0);
        }

        // Extract tasks
        let tasks: Vec<TransitionTask> =
            (0..batch_size).filter_map(|_| queue.pop()).collect();

        drop(queue); // Release lock

        // Process transitions
        let mut processed = 0;
        let mut history = self.transition_history.write().await;

        for task in tasks {
            if let Some((vector, metadata)) = vectors.get_mut(&task.vector_id) {
                let from_stage = vector.stage();

                match self
                    .transitioner
                    .transition_to_stage(vector, task.target_stage)
                {
                    Ok(new_vector) => {
                        let to_stage = new_vector.stage();

                        // Calculate memory saved
                        let old_mem = vector.memory_usage() as u64;
                        let new_mem = new_vector.memory_usage() as u64;
                        if old_mem > new_mem {
                            self.metrics.record_memory_saved(old_mem - new_mem);
                        }

                        *vector = new_vector;
                        metadata.stage = to_stage;

                        self.metrics.record_transition();
                        self.metrics.record_stage(to_stage);
                        processed += 1;

                        // Record history
                        let record = TransitionRecord {
                            from_stage,
                            to_stage,
                            timestamp: current_timestamp(),
                            success: true,
                        };
                        history
                            .entry(task.vector_id.clone())
                            .or_insert_with(Vec::new)
                            .push(record);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to transition vector {}: {}", task.vector_id, e);

                        // Record failure
                        let record = TransitionRecord {
                            from_stage,
                            to_stage: task.target_stage,
                            timestamp: current_timestamp(),
                            success: false,
                        };
                        history
                            .entry(task.vector_id.clone())
                            .or_insert_with(Vec::new)
                            .push(record);
                    }
                }
            }
        }

        Ok(processed)
    }

    /// Process a batch of transitions from the priority queue (sync version)
    #[cfg(not(feature = "nymph-async"))]
    pub fn process_batch(
        &self,
        vectors: &mut HashMap<String, (EncodedVector, VectorMetadata)>,
    ) -> Result<usize> {
        let mut queue = self.priority_queue.write();
        let batch_size = self.config.max_batch_size.min(queue.len());

        if batch_size == 0 {
            return Ok(0);
        }

        // Extract tasks
        let tasks: Vec<TransitionTask> =
            (0..batch_size).filter_map(|_| queue.pop()).collect();

        drop(queue); // Release lock

        // Process transitions
        let mut processed = 0;
        let mut history = self.transition_history.write();

        for task in tasks {
            if let Some((vector, metadata)) = vectors.get_mut(&task.vector_id) {
                let from_stage = vector.stage();

                match self
                    .transitioner
                    .transition_to_stage(vector, task.target_stage)
                {
                    Ok(new_vector) => {
                        let to_stage = new_vector.stage();

                        // Calculate memory saved
                        let old_mem = vector.memory_usage() as u64;
                        let new_mem = new_vector.memory_usage() as u64;
                        if old_mem > new_mem {
                            self.metrics.record_memory_saved(old_mem - new_mem);
                        }

                        *vector = new_vector;
                        metadata.stage = to_stage;

                        self.metrics.record_transition();
                        self.metrics.record_stage(to_stage);
                        processed += 1;

                        // Record history
                        let record = TransitionRecord {
                            from_stage,
                            to_stage,
                            timestamp: current_timestamp(),
                            success: true,
                        };
                        history
                            .entry(task.vector_id.clone())
                            .or_insert_with(Vec::new)
                            .push(record);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to transition vector {}: {}", task.vector_id, e);

                        // Record failure
                        let record = TransitionRecord {
                            from_stage,
                            to_stage: task.target_stage,
                            timestamp: current_timestamp(),
                            success: false,
                        };
                        history
                            .entry(task.vector_id.clone())
                            .or_insert_with(Vec::new)
                            .push(record);
                    }
                }
            }
        }

        Ok(processed)
    }

    /// Run garbage collection on transition history
    #[cfg(feature = "nymph-async")]
    pub async fn garbage_collect(&self) -> Result<usize> {
        let mut history = self.transition_history.write().await;
        let now = current_timestamp();
        let cutoff = now.saturating_sub(self.config.history_max_age_secs);

        let mut removed = 0;

        if self.config.aggressive_gc {
            // Remove entire entries older than cutoff
            history.retain(|_, records| {
                if let Some(last) = records.last() {
                    last.timestamp >= cutoff
                } else {
                    false
                }
            });
            removed = history.len();
        } else {
            // Just trim old records from each entry
            for records in history.values_mut() {
                let original_len = records.len();
                records.retain(|r| r.timestamp >= cutoff);
                removed += original_len - records.len();
            }
        }

        Ok(removed)
    }

    /// Run garbage collection on transition history (sync version)
    #[cfg(not(feature = "nymph-async"))]
    pub fn garbage_collect(&self) -> Result<usize> {
        let mut history = self.transition_history.write();
        let now = current_timestamp();
        let cutoff = now.saturating_sub(self.config.history_max_age_secs);

        let mut removed = 0;

        if self.config.aggressive_gc {
            // Remove entire entries older than cutoff
            history.retain(|_, records| {
                if let Some(last) = records.last() {
                    last.timestamp >= cutoff
                } else {
                    false
                }
            });
            removed = history.len();
        } else {
            // Just trim old records from each entry
            for records in history.values_mut() {
                let original_len = records.len();
                records.retain(|r| r.timestamp >= cutoff);
                removed += original_len - records.len();
            }
        }

        Ok(removed)
    }

    /// Pre-warm predicted hot vectors
    #[cfg(feature = "nymph-async")]
    pub async fn prewarm_vectors(
        &self,
        candidates: &[(String, VectorMetadata)],
        vectors: &mut HashMap<String, (EncodedVector, VectorMetadata)>,
    ) -> Result<usize> {
        let mut prewarmed = 0;

        for (id, predicted_metadata) in candidates {
            if let Some((vector, current_metadata)) = vectors.get_mut(id) {
                // Check if prediction suggests promoting to Adult
                if predicted_metadata.recent_access_count > current_metadata.recent_access_count * 2
                    && vector.stage() != VectorStage::Adult
                {
                    match self
                        .transitioner
                        .transition_to_stage(vector, VectorStage::Adult)
                    {
                        Ok(adult_vector) => {
                            *vector = adult_vector;
                            current_metadata.stage = VectorStage::Adult;
                            self.metrics.record_stage(VectorStage::Adult);
                            prewarmed += 1;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to prewarm vector {}: {}", id, e);
                        }
                    }
                }
            }
        }

        Ok(prewarmed)
    }

    /// Pre-warm predicted hot vectors (sync version)
    #[cfg(not(feature = "nymph-async"))]
    pub fn prewarm_vectors(
        &self,
        candidates: &[(String, VectorMetadata)],
        vectors: &mut HashMap<String, (EncodedVector, VectorMetadata)>,
    ) -> Result<usize> {
        let mut prewarmed = 0;

        for (id, predicted_metadata) in candidates {
            if let Some((vector, current_metadata)) = vectors.get_mut(id) {
                // Check if prediction suggests promoting to Adult
                if predicted_metadata.recent_access_count > current_metadata.recent_access_count * 2
                    && vector.stage() != VectorStage::Adult
                {
                    match self
                        .transitioner
                        .transition_to_stage(vector, VectorStage::Adult)
                    {
                        Ok(adult_vector) => {
                            *vector = adult_vector;
                            current_metadata.stage = VectorStage::Adult;
                            self.metrics.record_stage(VectorStage::Adult);
                            prewarmed += 1;
                        }
                        Err(e) => {
                            tracing::warn!("Failed to prewarm vector {}: {}", id, e);
                        }
                    }
                }
            }
        }

        Ok(prewarmed)
    }

    /// Get transition statistics
    #[cfg(feature = "nymph-async")]
    pub async fn get_stats(&self) -> DaemonStats {
        let queue = self.priority_queue.read().await;
        let history = self.transition_history.read().await;

        let queue_size = queue.len();
        let total_history = history.values().map(|v| v.len()).sum();
        let successful_transitions = history
            .values()
            .flat_map(|v| v.iter())
            .filter(|r| r.success)
            .count();
        let failed_transitions = history
            .values()
            .flat_map(|v| v.iter())
            .filter(|r| !r.success)
            .count();

        DaemonStats {
            queue_size,
            total_history,
            successful_transitions,
            failed_transitions,
            metrics: self.metrics.snapshot(),
        }
    }

    /// Get transition statistics (sync version)
    #[cfg(not(feature = "nymph-async"))]
    pub fn get_stats(&self) -> DaemonStats {
        let queue = self.priority_queue.read();
        let history = self.transition_history.read();

        let queue_size = queue.len();
        let total_history = history.values().map(|v| v.len()).sum();
        let successful_transitions = history
            .values()
            .flat_map(|v| v.iter())
            .filter(|r| r.success)
            .count();
        let failed_transitions = history
            .values()
            .flat_map(|v| v.iter())
            .filter(|r| !r.success)
            .count();

        DaemonStats {
            queue_size,
            total_history,
            successful_transitions,
            failed_transitions,
            metrics: self.metrics.snapshot(),
        }
    }

    /// Determine priority for a transition
    fn determine_priority(&self, metadata: &VectorMetadata, target: VectorStage) -> TransitionPriority {
        let current = metadata.stage;

        // Promotions are higher priority than demotions
        match (current, target) {
            (VectorStage::Larval, VectorStage::Adult) => TransitionPriority::Urgent,
            (VectorStage::Nymph, VectorStage::Adult) => TransitionPriority::Urgent,
            (VectorStage::Larval, VectorStage::Nymph) => TransitionPriority::High,
            (VectorStage::Adult, VectorStage::Nymph) => TransitionPriority::Normal,
            (VectorStage::Nymph, VectorStage::Larval) => TransitionPriority::Low,
            (VectorStage::Adult, VectorStage::Larval) => TransitionPriority::Low,
            _ => TransitionPriority::Normal,
        }
    }
}

/// Statistics from the daemon
#[derive(Debug, Clone)]
pub struct DaemonStats {
    pub queue_size: usize,
    pub total_history: usize,
    pub successful_transitions: usize,
    pub failed_transitions: usize,
    pub metrics: crate::advanced::nymph_encoding::NymphMetrics,
}

#[cfg(all(test, feature = "nymph-async"))]
mod async_tests {
    use super::*;
    use crate::advanced::nymph_encoding::TemperaturePolicy;

    #[tokio::test]
    async fn test_daemon_creation() {
        let config = DaemonConfig::default();
        let policy = Arc::new(TemperaturePolicy::balanced());
        let metrics = Arc::new(AtomicNymphMetrics::new());

        let daemon = MetamorphosisDaemon::new(config, policy, metrics);
        let stats = daemon.get_stats().await;

        assert_eq!(stats.queue_size, 0);
        assert_eq!(stats.total_history, 0);
    }

    #[tokio::test]
    async fn test_scan_and_queue() {
        let config = DaemonConfig::default();
        let policy = Arc::new(TemperaturePolicy::aggressive());
        let metrics = Arc::new(AtomicNymphMetrics::new());
        let daemon = MetamorphosisDaemon::new(config, policy, metrics);

        let mut metadata = VectorMetadata::new();
        // Make it cold
        metadata.last_access_time = 0;

        let vector = EncodedVector::Adult {
            data: vec![1.0, 2.0, 3.0],
        };

        let vectors = vec![("test".to_string(), vector, metadata)];

        let queued = daemon.scan_and_queue(&vectors).await.unwrap();
        assert_eq!(queued, 1);

        let stats = daemon.get_stats().await;
        assert_eq!(stats.queue_size, 1);
    }
}

#[cfg(all(test, not(feature = "nymph-async")))]
mod sync_tests {
    use super::*;
    use crate::advanced::nymph_encoding::TemperaturePolicy;

    #[test]
    fn test_daemon_creation() {
        let config = DaemonConfig::default();
        let policy = Arc::new(TemperaturePolicy::balanced());
        let metrics = Arc::new(AtomicNymphMetrics::new());

        let daemon = MetamorphosisDaemon::new(config, policy, metrics);
        let stats = daemon.get_stats();

        assert_eq!(stats.queue_size, 0);
        assert_eq!(stats.total_history, 0);
    }

    #[test]
    fn test_scan_and_queue() {
        let config = DaemonConfig::default();
        let policy = Arc::new(TemperaturePolicy::aggressive());
        let metrics = Arc::new(AtomicNymphMetrics::new());
        let daemon = MetamorphosisDaemon::new(config, policy, metrics);

        let mut metadata = VectorMetadata::new();
        // Make it cold
        metadata.last_access_time = 0;

        let vector = EncodedVector::Adult {
            data: vec![1.0, 2.0, 3.0],
        };

        let vectors = vec![("test".to_string(), vector, metadata)];

        let queued = daemon.scan_and_queue(&vectors).unwrap();
        assert_eq!(queued, 1);

        let stats = daemon.get_stats();
        assert_eq!(stats.queue_size, 1);
    }
}

#[cfg(test)]
mod common_tests {
    use super::*;

    #[test]
    fn test_priority_ordering() {
        let urgent = TransitionTask {
            vector_id: "urgent".to_string(),
            current_stage: VectorStage::Larval,
            target_stage: VectorStage::Adult,
            priority: TransitionPriority::Urgent,
            timestamp: 100,
        };

        let low = TransitionTask {
            vector_id: "low".to_string(),
            current_stage: VectorStage::Adult,
            target_stage: VectorStage::Larval,
            priority: TransitionPriority::Low,
            timestamp: 100,
        };

        assert!(urgent > low);
    }
}
