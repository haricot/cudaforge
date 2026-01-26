//! Parallel build configuration

use std::str::FromStr;

/// Parallel build configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Percentage of available threads to use (0.0 - 1.0)
    thread_percentage: f32,
    /// Maximum threads (overrides percentage if set)
    max_threads: Option<usize>,
    /// Minimum threads (floor)
    min_threads: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            thread_percentage: 0.5, // 50% by default
            max_threads: None,
            min_threads: 1,
        }
    }
}

impl ParallelConfig {
    /// Create a new parallel config with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the percentage of available threads to use
    ///
    /// Value should be between 0.0 and 1.0
    pub fn with_percentage(mut self, percentage: f32) -> Self {
        self.thread_percentage = percentage.clamp(0.0, 1.0);
        self
    }

    /// Set the maximum number of threads
    pub fn with_max_threads(mut self, max: usize) -> Self {
        self.max_threads = Some(max.max(1));
        self
    }

    /// Set the minimum number of threads
    pub fn with_min_threads(mut self, min: usize) -> Self {
        self.min_threads = min.max(1);
        self
    }

    /// Calculate the number of threads to use
    pub fn thread_count(&self) -> usize {
        // Check environment variable first
        if let Ok(env_threads) = std::env::var("CUDAFORGE_THREADS") {
            if let Ok(n) = usize::from_str(&env_threads) {
                return n.max(1);
            }
        }

        // Also check RAYON_NUM_THREADS for compatibility
        if let Ok(env_threads) = std::env::var("RAYON_NUM_THREADS") {
            if let Ok(n) = usize::from_str(&env_threads) {
                return n.max(1);
            }
        }

        let available = self.detect_available_threads();

        let calculated = if let Some(max) = self.max_threads {
            max.min(available)
        } else {
            (available as f32 * self.thread_percentage).ceil() as usize
        };

        calculated.max(self.min_threads).min(available)
    }

    /// Initialize the rayon thread pool with configured settings
    pub fn init_thread_pool(&self) -> Result<(), rayon::ThreadPoolBuildError> {
        let thread_count = self.thread_count();

        rayon::ThreadPoolBuilder::new()
            .num_threads(thread_count)
            .build_global()
    }

    /// Get thread count for nvcc --threads argument
    pub fn nvcc_threads(&self) -> Option<usize> {
        // Check NVCC_THREADS environment variable
        if let Ok(val) = std::env::var("NVCC_THREADS") {
            if let Ok(n) = val.parse::<usize>() {
                if n > 0 {
                    return Some(n);
                }
            }
        }

        // Default to 2 threads for nvcc internal parallelism
        Some(2)
    }

    fn detect_available_threads(&self) -> usize {
        // Try std::thread::available_parallelism first
        if let Ok(parallelism) = std::thread::available_parallelism() {
            return parallelism.get();
        }

        // Fallback to num_cpus crate for physical cores
        num_cpus::get_physical()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ParallelConfig::default();
        assert_eq!(config.thread_percentage, 0.5);
        assert!(config.max_threads.is_none());
    }

    #[test]
    fn test_percentage_clamping() {
        let config = ParallelConfig::new().with_percentage(1.5);
        assert_eq!(config.thread_percentage, 1.0);

        let config = ParallelConfig::new().with_percentage(-0.5);
        assert_eq!(config.thread_percentage, 0.0);
    }
}
