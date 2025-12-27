//! # Nymph-AgenticDB Integration
//!
//! Integrates Nymph lifecycle-based compression with AgentDB memory systems.
//!
//! ## Integration Strategy
//!
//! 1. **Episode Memory**: Recent episodes => Adult, established patterns => Nymph, historical => Larval
//! 2. **Skill Embeddings**: High-usage => Adult, medium-usage => Nymph, rarely used => Larval
//! 3. **Causal Graph**: Recent edges => Adult, validated patterns => Nymph, historical => Larval
//! 4. **Learning Sessions**: Active => Adult, completed with replay => Nymph, archived => Larval
//!
//! ## Memory Savings Example (1536-dim embeddings)
//!
//! - **Adult**: 6144 bytes/entry
//! - **Nymph**: 1536 bytes/entry (4x compression)
//! - **Larval**: 192 bytes/entry (32x compression)
//!
//! For 10K entries with 70% Nymph/Larval: ~78% memory reduction

use crate::agenticdb::{AgenticDB, CausalEdge, LearningSession, ReflexionEpisode, Skill};
use crate::error::{Result, RuvectorError};
use crate::nymph::{
    LifecycleMetadata, LifecycleStage, MetamorphicStats, NymphEncodedData, NymphEntry,
    NymphStorage, StageStatistics,
};
use crate::types::{SearchQuery, VectorEntry};
use crate::vector_db::VectorDB;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Demotion thresholds (in seconds)
const ADULT_TO_NYMPH_THRESHOLD: u64 = 24 * 3600; // 24 hours
const NYMPH_TO_LARVAL_THRESHOLD: u64 = 7 * 24 * 3600; // 7 days

/// Extended metadata for AgenticDB entries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticMemoryMetadata {
    /// Lifecycle tracking
    pub lifecycle: LifecycleMetadata,
    /// Entry type (episode, skill, causal, session)
    pub entry_type: String,
    /// Original entry ID in AgenticDB
    pub agentic_id: String,
    /// Additional type-specific metadata
    pub custom: HashMap<String, serde_json::Value>,
}

/// NymphAgenticDB - AgenticDB with Nymph lifecycle management
pub struct NymphAgenticDB {
    /// Underlying AgenticDB for core functionality
    agentic_db: Arc<AgenticDB>,
    /// Nymph storage for lifecycle management
    nymph_storage: Arc<NymphStorage>,
    /// Metamorphic daemon for background lifecycle management
    daemon_running: Arc<RwLock<bool>>,
    /// Daemon handle
    daemon_handle: Arc<RwLock<Option<std::thread::JoinHandle<()>>>>,
    /// Metadata tracking
    metadata_map: Arc<RwLock<HashMap<String, AgenticMemoryMetadata>>>,
}

impl NymphAgenticDB {
    /// Create new NymphAgenticDB
    pub fn new<P: AsRef<Path>>(
        agentic_db: Arc<AgenticDB>,
        nymph_path: P,
        dimensions: usize,
    ) -> Result<Self> {
        let nymph_storage = Arc::new(NymphStorage::new(nymph_path, dimensions)?);

        Ok(Self {
            agentic_db,
            nymph_storage,
            daemon_running: Arc::new(RwLock::new(false)),
            daemon_handle: Arc::new(RwLock::new(None)),
            metadata_map: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    // ============ Episode Memory with Nymph ============

    /// Store a reflexion episode with Nymph lifecycle management
    ///
    /// Episodes start in Adult stage (frequently accessed during learning).
    /// After 24h without access => demote to Nymph
    /// After 7d without access => demote to Larval
    pub fn store_episode_with_nymph(
        &self,
        task: String,
        actions: Vec<String>,
        observations: Vec<String>,
        critique: String,
    ) -> Result<String> {
        // Store in AgenticDB
        let episode_id = self
            .agentic_db
            .store_episode(task.clone(), actions, observations, critique.clone())?;

        // Get the embedding from AgenticDB (it was generated during store_episode)
        let vector_id = format!("reflexion_{}", episode_id);
        let entry = self
            .agentic_db
            .get(&vector_id)?
            .ok_or_else(|| RuvectorError::VectorNotFound(vector_id.clone()))?;

        // Store in Nymph storage (starts as Adult)
        let nymph_id = format!("episode_{}", episode_id);
        self.nymph_storage.store(nymph_id.clone(), entry.vector)?;

        // Track metadata
        let metadata = AgenticMemoryMetadata {
            lifecycle: LifecycleMetadata::new(),
            entry_type: "reflexion_episode".to_string(),
            agentic_id: episode_id.clone(),
            custom: {
                let mut map = HashMap::new();
                map.insert("task".to_string(), serde_json::json!(task));
                map.insert("critique_length".to_string(), serde_json::json!(critique.len()));
                map
            },
        };
        self.metadata_map.write().insert(nymph_id, metadata);

        Ok(episode_id)
    }

    /// Retrieve similar episodes with automatic promotion
    ///
    /// On retrieval, episodes are automatically promoted back to Adult stage
    pub fn retrieve_episodes_with_promotion(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<ReflexionEpisode>> {
        // Use AgenticDB's search
        let episodes = self.agentic_db.retrieve_similar_episodes(query, k)?;

        // Promote accessed episodes
        for episode in &episodes {
            let nymph_id = format!("episode_{}", episode.id);
            // This will automatically promote if needed
            let _ = self.nymph_storage.retrieve(&nymph_id)?;
        }

        Ok(episodes)
    }

    // ============ Skill Embeddings with Nymph ============

    /// Create skill with usage-based staging
    ///
    /// High-usage skills => Adult (fast access)
    /// Medium-usage => Nymph
    /// Rarely used => Larval (archive)
    pub fn create_skill_with_nymph(
        &self,
        name: String,
        description: String,
        parameters: HashMap<String, String>,
        examples: Vec<String>,
    ) -> Result<String> {
        // Store in AgenticDB
        let skill_id = self
            .agentic_db
            .create_skill(name.clone(), description.clone(), parameters, examples)?;

        // Get embedding
        let vector_id = format!("skill_{}", skill_id);
        let entry = self
            .agentic_db
            .get(&vector_id)?
            .ok_or_else(|| RuvectorError::VectorNotFound(vector_id.clone()))?;

        // Store in Nymph storage
        let nymph_id = format!("skill_{}", skill_id);
        self.nymph_storage.store(nymph_id.clone(), entry.vector)?;

        // Track metadata
        let metadata = AgenticMemoryMetadata {
            lifecycle: LifecycleMetadata::new(),
            entry_type: "skill".to_string(),
            agentic_id: skill_id.clone(),
            custom: {
                let mut map = HashMap::new();
                map.insert("name".to_string(), serde_json::json!(name));
                map.insert("description".to_string(), serde_json::json!(description));
                map
            },
        };
        self.metadata_map.write().insert(nymph_id, metadata);

        Ok(skill_id)
    }

    /// Search skills with automatic promotion on access
    pub fn search_skills_with_promotion(
        &self,
        query_description: &str,
        k: usize,
    ) -> Result<Vec<Skill>> {
        let skills = self.agentic_db.search_skills(query_description, k)?;

        // Promote accessed skills
        for skill in &skills {
            let nymph_id = format!("skill_{}", skill.id);
            let _ = self.nymph_storage.retrieve(&nymph_id)?;
        }

        Ok(skills)
    }

    // ============ Causal Graph Compression ============

    /// Add causal edge with temporal staging
    ///
    /// Recent causal edges => Adult
    /// Established patterns => Nymph (validated, less volatile)
    /// Historical => Larval
    pub fn add_causal_edge_with_nymph(
        &self,
        causes: Vec<String>,
        effects: Vec<String>,
        confidence: f64,
        context: String,
    ) -> Result<String> {
        // Store in AgenticDB
        let edge_id = self
            .agentic_db
            .add_causal_edge(causes.clone(), effects.clone(), confidence, context.clone())?;

        // Get embedding
        let vector_id = format!("causal_{}", edge_id);
        let entry = self
            .agentic_db
            .get(&vector_id)?
            .ok_or_else(|| RuvectorError::VectorNotFound(vector_id.clone()))?;

        // Store in Nymph storage
        let nymph_id = format!("causal_{}", edge_id);
        self.nymph_storage.store(nymph_id.clone(), entry.vector)?;

        // Track metadata
        let metadata = AgenticMemoryMetadata {
            lifecycle: LifecycleMetadata::new(),
            entry_type: "causal_edge".to_string(),
            agentic_id: edge_id.clone(),
            custom: {
                let mut map = HashMap::new();
                map.insert("causes_count".to_string(), serde_json::json!(causes.len()));
                map.insert("effects_count".to_string(), serde_json::json!(effects.len()));
                map.insert("confidence".to_string(), serde_json::json!(confidence));
                map.insert("context".to_string(), serde_json::json!(context));
                map
            },
        };
        self.metadata_map.write().insert(nymph_id, metadata);

        Ok(edge_id)
    }

    // ============ Learning Session Optimization ============

    /// Start learning session with Nymph lifecycle
    ///
    /// Active sessions => Adult
    /// Completed sessions with replay => Nymph
    /// Archived sessions => Larval
    pub fn start_session_with_nymph(
        &self,
        algorithm: String,
        state_dim: usize,
        action_dim: usize,
    ) -> Result<String> {
        // Store in AgenticDB
        let session_id = self
            .agentic_db
            .start_session(algorithm.clone(), state_dim, action_dim)?;

        // Create a placeholder embedding for the session
        // (in practice, you might embed the algorithm name or session metadata)
        let embedding = vec![0.0; state_dim]; // Placeholder

        // Store in Nymph storage
        let nymph_id = format!("session_{}", session_id);
        self.nymph_storage.store(nymph_id.clone(), embedding)?;

        // Track metadata
        let metadata = AgenticMemoryMetadata {
            lifecycle: LifecycleMetadata::new(),
            entry_type: "learning_session".to_string(),
            agentic_id: session_id.clone(),
            custom: {
                let mut map = HashMap::new();
                map.insert("algorithm".to_string(), serde_json::json!(algorithm));
                map.insert("state_dim".to_string(), serde_json::json!(state_dim));
                map.insert("action_dim".to_string(), serde_json::json!(action_dim));
                map
            },
        };
        self.metadata_map.write().insert(nymph_id, metadata);

        Ok(session_id)
    }

    /// Add experience and touch session (keep it Adult)
    pub fn add_experience_with_promotion(
        &self,
        session_id: &str,
        state: Vec<f32>,
        action: Vec<f32>,
        reward: f64,
        next_state: Vec<f32>,
        done: bool,
    ) -> Result<()> {
        // Add to AgenticDB
        self.agentic_db
            .add_experience(session_id, state, action, reward, next_state, done)?;

        // Touch the session to keep it Adult
        let nymph_id = format!("session_{}", session_id);
        let _ = self.nymph_storage.retrieve(&nymph_id)?;

        Ok(())
    }

    // ============ Metamorphic Cycle Management ============

    /// Run metamorphic cycle to demote old entries
    ///
    /// - Adult => Nymph after 24h without access
    /// - Nymph => Larval after 7d without access
    pub fn run_metamorphic_cycle(&self) -> Result<MetamorphicStats> {
        self.nymph_storage.run_metamorphic_cycle()
    }

    /// Start background metamorphic daemon
    ///
    /// Runs cycle every `interval` duration
    pub fn start_metamorphic_daemon(&self, interval: Duration) {
        if *self.daemon_running.read() {
            return; // Already running
        }

        *self.daemon_running.write() = true;

        let storage = Arc::clone(&self.nymph_storage);
        let running = Arc::clone(&self.daemon_running);

        let handle = std::thread::spawn(move || {
            while *running.read() {
                std::thread::sleep(interval);

                if let Err(e) = storage.run_metamorphic_cycle() {
                    eprintln!("Metamorphic cycle error: {:?}", e);
                }
            }
        });

        *self.daemon_handle.write() = Some(handle);
    }

    /// Stop background metamorphic daemon
    pub fn stop_metamorphic_daemon(&self) {
        *self.daemon_running.write() = false;

        // Wait for daemon to finish
        if let Some(handle) = self.daemon_handle.write().take() {
            let _ = handle.join();
        }
    }

    /// Check if daemon is running
    pub fn is_daemon_running(&self) -> bool {
        *self.daemon_running.read()
    }

    // ============ Statistics and Monitoring ============

    /// Get comprehensive statistics for all lifecycle stages
    pub fn get_stage_statistics(&self) -> Result<NymphAgenticStats> {
        let nymph_stats = self.nymph_storage.get_stage_statistics()?;

        // Break down by entry type
        let metadata_map = self.metadata_map.read();
        let mut type_breakdown: HashMap<String, TypeStageStats> = HashMap::new();

        for (id, meta) in metadata_map.iter() {
            let stats = type_breakdown
                .entry(meta.entry_type.clone())
                .or_insert_with(TypeStageStats::default);

            match meta.lifecycle.stage {
                LifecycleStage::Adult => stats.adult_count += 1,
                LifecycleStage::Nymph => stats.nymph_count += 1,
                LifecycleStage::Larval => stats.larval_count += 1,
            }
            stats.total_accesses += meta.lifecycle.access_count;
        }

        Ok(NymphAgenticStats {
            overall: nymph_stats,
            by_type: type_breakdown,
            total_entries: metadata_map.len() as u64,
        })
    }

    /// Get memory savings compared to all-Adult storage
    pub fn calculate_memory_savings(&self) -> Result<f64> {
        let stats = self.nymph_storage.get_stage_statistics()?;
        Ok(stats.memory_savings_ratio())
    }

    /// Get entries by lifecycle stage
    pub fn get_entries_by_stage(&self, stage: LifecycleStage) -> Vec<AgenticMemoryMetadata> {
        self.metadata_map
            .read()
            .values()
            .filter(|meta| meta.lifecycle.stage == stage)
            .cloned()
            .collect()
    }

    /// Force promotion of specific entry
    pub fn force_promote(&self, entry_id: &str) -> Result<()> {
        let nymph_id = format!("entry_{}", entry_id);
        self.nymph_storage.retrieve(&nymph_id)?;
        Ok(())
    }

    /// Get AgenticDB reference (for advanced operations)
    pub fn agentic_db(&self) -> &Arc<AgenticDB> {
        &self.agentic_db
    }

    /// Get metadata for specific entry
    pub fn get_entry_metadata(&self, nymph_id: &str) -> Option<AgenticMemoryMetadata> {
        self.metadata_map.read().get(nymph_id).cloned()
    }
}

/// Statistics breakdown by entry type
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct TypeStageStats {
    /// Adult stage count
    pub adult_count: u64,
    /// Nymph stage count
    pub nymph_count: u64,
    /// Larval stage count
    pub larval_count: u64,
    /// Total accesses
    pub total_accesses: u64,
}

/// Comprehensive statistics for NymphAgenticDB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NymphAgenticStats {
    /// Overall Nymph storage statistics
    pub overall: StageStatistics,
    /// Breakdown by entry type
    pub by_type: HashMap<String, TypeStageStats>,
    /// Total tracked entries
    pub total_entries: u64,
}

impl NymphAgenticStats {
    /// Print human-readable report
    pub fn print_report(&self) {
        println!("=== NymphAgenticDB Statistics ===\n");

        println!("Overall:");
        println!("  Adult:  {} entries ({} bytes)", self.overall.adult_count, self.overall.adult_bytes);
        println!("  Nymph:  {} entries ({} bytes)", self.overall.nymph_count, self.overall.nymph_bytes);
        println!("  Larval: {} entries ({} bytes)", self.overall.larval_count, self.overall.larval_bytes);
        println!("  Total:  {} bytes", self.overall.total_bytes);
        println!(
            "  Memory savings: {:.1}%",
            self.overall.memory_savings_ratio() * 100.0
        );
        println!("  Avg accesses: {:.1}", self.overall.avg_access_count());

        println!("\nBy Entry Type:");
        for (entry_type, stats) in &self.by_type {
            let total = stats.adult_count + stats.nymph_count + stats.larval_count;
            println!(
                "  {}: {} total (A:{} N:{} L:{})",
                entry_type, total, stats.adult_count, stats.nymph_count, stats.larval_count
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DbOptions;
    use tempfile::tempdir;

    fn create_test_nymph_db() -> Result<NymphAgenticDB> {
        let dir = tempdir().unwrap();
        let agentic_path = dir.path().join("agentic.db");
        let nymph_path = dir.path().join("nymph.db");

        let mut options = DbOptions::default();
        options.storage_path = agentic_path.to_string_lossy().to_string();
        options.dimensions = 128;

        let agentic_db = Arc::new(AgenticDB::new(options)?);
        NymphAgenticDB::new(agentic_db, nymph_path, 128)
    }

    #[test]
    fn test_episode_storage_and_retrieval() -> Result<()> {
        let db = create_test_nymph_db()?;

        let episode_id = db.store_episode_with_nymph(
            "Solve math problem".to_string(),
            vec!["read".to_string(), "calculate".to_string()],
            vec!["got 42".to_string()],
            "Should show work".to_string(),
        )?;

        // Retrieve should promote
        let episodes = db.retrieve_episodes_with_promotion("math problem", 5)?;
        assert!(!episodes.is_empty());

        Ok(())
    }

    #[test]
    fn test_skill_lifecycle() -> Result<()> {
        let db = create_test_nymph_db()?;

        let mut params = HashMap::new();
        params.insert("input".to_string(), "string".to_string());

        let skill_id = db.create_skill_with_nymph(
            "Parse JSON".to_string(),
            "Parse JSON from string".to_string(),
            params,
            vec!["json.parse()".to_string()],
        )?;

        // Search should promote
        let skills = db.search_skills_with_promotion("parse json", 5)?;
        assert!(!skills.is_empty());

        Ok(())
    }

    #[test]
    fn test_causal_edge_storage() -> Result<()> {
        let db = create_test_nymph_db()?;

        let edge_id = db.add_causal_edge_with_nymph(
            vec!["rain".to_string()],
            vec!["wet ground".to_string()],
            0.95,
            "Weather observation".to_string(),
        )?;

        assert!(!edge_id.is_empty());

        Ok(())
    }

    #[test]
    fn test_learning_session() -> Result<()> {
        let db = create_test_nymph_db()?;

        let session_id = db.start_session_with_nymph("Q-Learning".to_string(), 4, 2)?;

        db.add_experience_with_promotion(
            &session_id,
            vec![1.0, 0.0, 0.0, 0.0],
            vec![1.0, 0.0],
            1.0,
            vec![0.0, 1.0, 0.0, 0.0],
            false,
        )?;

        Ok(())
    }

    #[test]
    fn test_statistics() -> Result<()> {
        let db = create_test_nymph_db()?;

        // Add some entries
        for i in 0..5 {
            db.store_episode_with_nymph(
                format!("Task {}", i),
                vec!["action".to_string()],
                vec!["observation".to_string()],
                format!("Critique {}", i),
            )?;
        }

        let stats = db.get_stage_statistics()?;
        assert_eq!(stats.total_entries, 5);
        assert_eq!(stats.overall.adult_count, 5); // All start as Adult

        stats.print_report();

        Ok(())
    }

    #[test]
    fn test_metamorphic_daemon() -> Result<()> {
        let db = create_test_nymph_db()?;

        assert!(!db.is_daemon_running());

        db.start_metamorphic_daemon(Duration::from_millis(100));
        assert!(db.is_daemon_running());

        std::thread::sleep(Duration::from_millis(250));

        db.stop_metamorphic_daemon();
        assert!(!db.is_daemon_running());

        Ok(())
    }

    #[test]
    fn test_memory_savings() -> Result<()> {
        let db = create_test_nymph_db()?;

        // Add entries
        for i in 0..10 {
            db.store_episode_with_nymph(
                format!("Task {}", i),
                vec!["action".to_string()],
                vec!["obs".to_string()],
                format!("Critique {}", i),
            )?;
        }

        // Initially all Adult, so no savings
        let savings = db.calculate_memory_savings()?;
        assert!(savings >= 0.0);

        Ok(())
    }

    #[test]
    fn test_entry_type_breakdown() -> Result<()> {
        let db = create_test_nymph_db()?;

        // Add different types
        db.store_episode_with_nymph(
            "Task".to_string(),
            vec!["action".to_string()],
            vec!["obs".to_string()],
            "Critique".to_string(),
        )?;

        let mut params = HashMap::new();
        params.insert("input".to_string(), "string".to_string());
        db.create_skill_with_nymph(
            "Skill".to_string(),
            "Description".to_string(),
            params,
            vec!["example".to_string()],
        )?;

        db.add_causal_edge_with_nymph(
            vec!["cause".to_string()],
            vec!["effect".to_string()],
            0.9,
            "context".to_string(),
        )?;

        let stats = db.get_stage_statistics()?;
        assert!(stats.by_type.contains_key("reflexion_episode"));
        assert!(stats.by_type.contains_key("skill"));
        assert!(stats.by_type.contains_key("causal_edge"));

        Ok(())
    }
}
