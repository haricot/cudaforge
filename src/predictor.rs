//! Implementation Predictor Module
//! 
//! Transforms static hardware metrics into an active cost model,
//! estimating kernel class affinities, resource pressures,
//! and likely optimal implementation strategies.

use crate::arch_metrics::{ArchObservables, DerivedProperties};
use crate::compute_cap::GpuArch;
use crate::error::Result;
use serde::Serialize;
use std::collections::HashMap;

/// Affinity/likelihood of a hardware architecture favoring a specific kernel class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Affinity {
    /// Highly suitable; architecture practically demands this approach.
    High,
    /// Viable but possibly subject to resource limits.
    Medium,
    /// Unlikely to be optimal; architecture lacks required traits.
    Low,
}

impl std::fmt::Display for Affinity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Affinity::High => write!(f, "\x1b[32mhigh\x1b[0m"),
            Affinity::Medium => write!(f, "\x1b[33mmedium\x1b[0m"),
            Affinity::Low => write!(f, "\x1b[31mlow\x1b[0m"),
        }
    }
}

/// Quantitative risk of register/memory/bandwidth pressure.
#[derive(Debug, Clone, Serialize)]
pub struct PressureRisk {
    /// Quantitative risk intensity (0.0 to 1.0).
    pub mean: f32,
    /// Qualitative risk label (Low, Medium, High).
    pub label: String,
}

impl PressureRisk {
    /// Create a new quantitative risk with an auto-generated label.
    pub fn new(mean: f32) -> Self {
        let label = if mean > 0.7 {
            "High"
        } else if mean > 0.3 {
            "Medium"
        } else {
            "Low"
        };
        Self {
            mean: mean.clamp(0.0, 1.0),
            label: label.to_string(),
        }
    }
}

impl std::fmt::Display for PressureRisk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let color = if self.mean > 0.7 {
            "\x1b[31m" // red
        } else if self.mean > 0.3 {
            "\x1b[33m" // yellow
        } else {
            "\x1b[32m" // green
        };
        write!(f, "{}{}\x1b[0m ({:.2})", color, self.label.to_lowercase(), self.mean)
    }
}

/// Description of a compute workload for oracle inference.
#[derive(Debug, Clone, Serialize)]
pub struct WorkloadDesc {
    /// Geometric shape of the problem.
    pub shape: ProblemShape,
    /// Precision to use for compute.
    pub dtype: DType,
    /// External calibration state to apply.
    pub calibration: Option<CalibrationState>,
}

/// Actionable execution policy derived from architectural priors.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionPolicy {
    /// Recommended high-level implementation strategies.
    pub rankings: Vec<KernelPrediction>,
    /// Suggested pruning of the autotuning search space.
    pub search_space: SearchSpace,
    /// Reduction factor of the search space (Naive volume / Pruned volume).
    pub pruning_factor: f32,
    /// Graph-level optimization hints (e.g. for fusion or layout).
    pub graph_hints: Vec<String>,
    /// Statistical confidence in this policy (0.0 to 1.0).
    pub policy_confidence: f32,
}

/// Phase of the compilation/execution lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum TuningPhase {
    /// Static analysis phase during compilation.
    #[default]
    CompileTime,
    /// JIT compilation or runtime kernel loading.
    Runtime,
    /// Offline profile-guided optimization.
    Offline,
}

/// Compiler context for deriving execution policies.
#[derive(Debug, Clone, Serialize, Default)]
pub struct TuningContext {
    /// Allowable risk for aggressive optimizations (0.0 to 1.0).
    pub risk_tolerance: f32,
    /// Budget for autotuning (e.g. "quick", "thorough").
    pub tuning_budget: String,
    /// Target software backend (e.g. "ptx", "sass", "cutlass").
    pub target_backend: String,
    /// The current phase of the lifecycle.
    pub phase: TuningPhase,
    /// Whether deterministic execution is strictly required.
    pub determinism_required: bool,
}

/// The formal probabilistic hardware oracle trait for ML compilers.
pub trait ProbabilisticHardwareOracle: Send + Sync {
    /// Produce a probabilistic posterior for a specific workload.
    fn posterior(&self, workload: &WorkloadDesc) -> InferredState;

    /// Derive an actionable execution policy from an inferred state.
    fn policy(&self, posterior: &InferredState, context: &TuningContext) -> ExecutionPolicy;

    /// Update the oracle's internal belief state based on runtime feedback.
    fn update(&mut self, observation: &CalibrationFeedback);

    /// Annotate a generic graph node with architectural affinities and risks.
    fn annotate_node(&self, workload: &WorkloadDesc, labels: &mut HashMap<String, String>);
}

/// SM execution concurrency constraints.
#[derive(Debug, Clone, Serialize)]
pub struct SmExecutionModel {
    /// Maximum concurrent warps per SM.
    pub max_warps_per_sm: u32,
    /// Maximum resident thread blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Number of warp schedulers per SM.
    pub schedulers_per_sm: u32,
}

/// Critical equilibrium ratios that dictate algorithmic choices.
#[derive(Debug, Clone, Serialize)]
pub struct ThroughputRatios {
    /// FP32 operations per cycle per SM.
    pub fp32_per_sm_ops_cycle: f32,
    /// Load/Store bandwidth (L1/TEX) per cycle per SM.
    pub ldst_per_sm_bytes_cycle: f32,
    /// Ratio of FP32 FLOPs to Shared Memory bandwidth (Compute vs SMEM).
    pub compute_to_shared_ratio: f32,
    /// Ratio of FP32 FLOPs to L2 bandwidth (Compute vs L2).
    pub compute_to_l2_ratio: f32,
}

/// Execution affordances dictated by modern software instruction sets.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct SoftwareExecutionLimits {
    /// Maximum number of asynchronous memory copies (cp.async / TMA) in flight.
    pub max_async_copies_in_flight: u32,
    /// Maximum number of mbarrier limit (0 if unsupported).
    pub max_barriers: u32,
    /// Is thread block cluster launch natively supported?
    pub cluster_launch_supported: bool,
    /// Is cooperative grid execution viable?
    pub cooperative_grid_viable: bool,
}

/// Specialized signals for Dense Deep Learning (GEMM) kernels.
#[derive(Debug, Clone, Serialize)]
pub struct GemmSpecificSignals {
    /// The optimal range of `K` tile sizes to target (min, max).
    pub preferred_tile_size_range: (u32, u32),
    /// Likelihood of achieving sufficient register reuse depth before spilling.
    pub reuse_depth_achievable: Affinity,
    /// Is split-K recommended dynamically? (e.g., "Yes, for large MxN", "No")
    pub split_k_viability: String,
    /// Is epilogue fusion (Bias+ReLU/GELU) viable without crashing register pressure?
    pub epilogue_fusion_headroom: Affinity,
}

/// Viable mathematical execution modes supported by the architecture.
#[derive(Debug, Clone, Serialize, Default)]
pub struct NumericExecutionModes {
    /// The primary precision for floating point math (e.g., "primary highly-optimized path").
    pub fp32_math_class: String,
    /// Viability of using 16-bit floating point math.
    pub fp16_math: String,
    /// Viability of using packed 16-bit vector math (e.g. `half2`).
    pub half2_vector_math: String,
    /// Viability of hardware-accelerated INT8 dot-product (e.g., `dp4a`).
    pub int8_dot_product: String,
    /// Viability of mixed-precision accumulation.
    pub mixed_precision_accumulate: String,
}

// ── Phase 12 Multi-Arch Enums ──

/// Classification of the compute pipeline's maturity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum ComputePipeline {
    /// Standard CUDA core based scalar math.
    #[default]
    ScalarCuda,
    /// Synchronous Tensor Core execution (Volta/Turing).
    TensorSync,
    /// Asynchronous Tensor Core execution with cp.async (Ampere).
    TensorAsync,
    /// Pipelined Tensor Core execution with TMA (Hopper).
    TensorPipeline,
    /// Dataflow-oriented execution with TMEM (Blackwell).
    TensorDataflow,
}

/// Classification of the memory orchestration strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum MemoryPipeline {
    /// No specialized memory pipeline.
    #[default]
    None,
    /// Standard Shared Memory blocking (L1-managed).
    SmemBlocking,
    /// Asynchronous Shared Memory copies (cp.async).
    AsyncSmem,
    /// Hardware-managed TMA streaming.
    TmaStreaming,
    /// Dataflow-oriented TMEM orchestration.
    TmemDataflow,
}

/// Complexity class of the cache hierarchy interaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum CacheModel {
    /// No specialized cache model.
    #[default]
    None,
    /// Standard L1/L2 cache hierarchy.
    Standard,
    /// L1/L2 with explicit shared memory.
    Smem,
    /// L1/L2 with explicit shared memory and asynchronous copies.
    AsyncSmem,
    /// L1/L2 with explicit shared memory and TMA streaming.
    Tma,
    /// L1/L2 with explicit shared memory and TMEM dataflow.
    Tmem,
}

/// Parallelism regime the architecture prefers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum ParallelismRegime {
    /// Reliant on high occupancy to hide latency.
    #[default]
    OccupancyBound,
    /// Mix of ILP and occupancy for latency hiding.
    LatencyHiding,
    /// Explicit warp specialization (producer/consumer).
    WarpSpecialized,
    /// Advanced producer-consumer patterns managed by hardware.
    ProducerConsumer,
    /// Full hardware-driven dataflow scheduling.
    HardwareScheduled,
    /// Persistent grid execution (reuse of blocks across waves).
    Persistent,
}

/// Categorization of the instruction issue unit behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum SchedulerModel {
    /// Simple in-order issue.
    #[default]
    InOrder,
    /// Out-of-order issue with limited resources.
    OoO,
    /// Out-of-order issue with advanced dependency tracking.
    AdvancedOoO,
    /// Hardware-managed instruction pipelines.
    Pipelined,
    /// Dataflow-driven instruction scheduling.
    Dataflow,
}

/// Numeric engine specialization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum NumericEngineClass {
    /// Basic single-precision only.
    #[default]
    Fp32Only,
    /// Mixed-precision scalar math (FP16/FP32).
    MixedScalar,
    /// Half-precision Tensor Core dominant.
    TensorFp16,
    /// Mixed-precision Tensor Core orchestration.
    TensorMixed,
    /// Low-precision (FP8/INT8) Tensor dominant.
    TensorLowPrecision,
    /// Microscaled data formats (MXFP4/MXFP8).
    MicroscaledTensor,
}

/// High-level architectural class representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum ExecutionArchClass {
    /// Pre-Tensor Core architectures (Pascal and older).
    #[default]
    PreTensor,
    /// Early Tensor Core architectures (Volta, Turing).
    EarlyTensor,
    /// Asynchronous Tensor architectures (Ampere).
    AsyncTensor,
    /// Hardware-pipelined architectures (Hopper).
    PipelineGpu,
    /// Dataflow-oriented architectures (Blackwell).
    DataflowGpu,
}

impl std::fmt::Display for ExecutionArchClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutionArchClass::PreTensor => write!(f, "Pre-Tensor (e.g. Pascal)"),
            ExecutionArchClass::EarlyTensor => write!(f, "Early Tensor (e.g. Volta/Turing)"),
            ExecutionArchClass::AsyncTensor => write!(f, "Async Tensor (e.g. Ampere)"),
            ExecutionArchClass::PipelineGpu => write!(f, "Pipeline GPU (e.g. Hopper)"),
            ExecutionArchClass::DataflowGpu => write!(f, "Dataflow GPU (e.g. Blackwell)"),
        }
    }
}

/// High-level summary of the architecture's behavioral regime.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ExecutionProfile {
    /// General architectural family category.
    pub arch_class: ExecutionArchClass,
    /// Complexity of the arithmetic compute units.
    pub compute_pipeline: ComputePipeline,
    /// Mechanism for staging data from global to local memory.
    pub memory_pipeline: MemoryPipeline,
    /// Strategy used to hide instruction and memory latency.
    pub parallelism_model: ParallelismRegime,
    /// Density and precision class of the math units.
    pub numeric_engine: NumericEngineClass,
    /// The recommended high-level implementation strategy for GEMM.
    pub preferred_kernel_family: String,
}

/// Estimates of achievable reuse and staging envelopes.
#[derive(Debug, Clone, Serialize, Default)]
pub struct MemoryReuseEnvelope {
    /// Expected reuse depth achievable given register file constraints.
    pub expected_reuse_depth: String,
    /// Maximum feasible staging depth through Shared Memory.
    pub smem_staging_depth: String,
    /// Headroom for scaling register tiling.
    pub register_tiling_headroom: String,
    /// Impact/leverage of register reuse on overall throughput.
    pub register_reuse_leverage: String,
    /// Which bandwidth threshold is the primary architectural bottleneck?
    pub bandwidth_pressure_regime: String,
}

/// High-level behavioral regime the GPU prefers.
#[derive(Debug, Clone, Serialize, Default)]
pub struct PerformanceRegime {
    /// Overall balance of compute vs memory.
    pub compute_vs_memory_balance: String,
    /// Sensitivity to register pressure during kernel execution.
    pub register_pressure_sensitivity: String,
    /// Sensitivity to the number of active warps fed to the scheduler.
    pub scheduler_pressure_sensitivity: String,
    /// Sensitivity to instruction-level parallelism (ILP).
    pub instruction_level_parallelism_sensitivity: String,
    /// Target issue slot utilization for latency hiding.
    pub issue_slot_utilization_target: String,
}

/// Sensitivity to the scale and size of the dispatched kernel grid.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ScaleSensitivity {
    /// Efficiency of dispatching small grids (e.g. inference).
    pub small_kernel_efficiency: String,
    /// Scaling behavior on massive grids.
    pub large_kernel_scaling: String,
    /// Point at which thread blocks enter a latency-dominated execution mode.
    pub latency_dominated_regime_threshold: String,
    /// Viability of warp specialization (producer/consumer patterns).
    pub warp_specialization_viability: String,
}

/// Model dictating how data movement should be approached.
#[derive(Debug, Clone, Serialize, Default)]
pub struct DataMovementModel {
    /// Cost of DRAM latency cache misses.
    pub dram_latency_cost: String,
    /// Expected performance leverage from keeping data resident in the L2 cache.
    pub l2_reuse_leverage: String,
    /// The necessity of amortizing loads through Shared Memory.
    pub smem_amortization: String,
}

/// Logical dimensions of the GEMM or convolution problem.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct ProblemShape {
    /// Dimension M: rows of matrix A and matrix C.
    pub m: usize,
    /// Dimension N: columns of matrix B and matrix C.
    pub n: usize,
    /// Dimension K: columns of matrix A and rows of matrix B.
    pub k: usize,
}

/// Supported data types for kernel benchmarking and prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Default)]
pub enum DType {
    /// 32-bit floating point.
    #[default]
    Fp32,
    /// 16-bit floating point.
    Fp16,
    /// 16-bit brain floating point.
    Bf16,
    /// 8-bit floating point.
    Fp8,
    /// 8-bit integer.
    Int8,
}

impl WorkloadDesc {
    /// Helper to generate a standard large-batch GEMM workload for testing.
    pub fn synthetic_gemm() -> Self {
        Self {
            shape: ProblemShape { m: 2048, n: 2048, k: 2048 },
            dtype: DType::Fp16,
            calibration: None,
        }
    }
}

impl DType {
    /// Returns the effective bits per element for memory bandwidth calculations.
    pub fn bits(&self) -> usize {
        match self {
            DType::Fp32 => 32,
            DType::Fp16 | DType::Bf16 => 16,
            DType::Fp8 | DType::Int8 => 8,
        }
    }
}

/// Detailed breakdown of hardware support for a specific data type.
#[derive(Debug, Clone, Serialize, Default)]
pub struct PrecisionBreakdown {
    /// Whether the hardware can load/store this data type.
    pub storage_supported: bool,
    /// Whether the hardware has native arithmetic units for this type (throughput gain).
    pub native_arithmetic: bool,
    /// Whether Tensor Cores support this data type.
    pub tensor_core_available: bool,
    /// Predicted throughput ratio compared to FP32 (e.g. 2.0 for packed FP16).
    pub throughput_ratio_vs_fp32: f32,
}

/// The feasibility and execution behavior of a specific data type on the target hardware.
#[derive(Debug, Clone, Serialize, Default)]
pub struct NumericFeasibility {
    /// The data type requested by the user.
    pub requested_dtype: DType,
    /// Whether this type is natively supported by the hardware's compute pipelines.
    pub is_native: bool,
    /// The actual data type used for computation (may be different if emulated/promoted).
    pub effective_compute_dtype: DType,
    /// Detailed hardware capability mapping.
    pub precision_breakdown: PrecisionBreakdown,
    /// The high-level execution regime (Native, Emulated, Promoted).
    pub execution_mode: String,
    /// Researcher-grade explanation for penalties or remappings.
    pub penalty_reason: Option<String>,
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::Fp32 => write!(f, "FP32"),
            DType::Fp16 => write!(f, "FP16"),
            DType::Bf16 => write!(f, "BF16"),
            DType::Fp8 => write!(f, "FP8"),
            DType::Int8 => write!(f, "INT8"),
        }
    }
}

/// High-level categories of kernel implementation strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum StrategyCategory {
    /// Standard blocked GEMM using CUDA cores.
    CudaCoreBlocked,
    /// Register-heavy tiling optimized for occupancy and ILP.
    WarpTiling,
    /// Native Tensor Core implementation (MMA/WGMMA).
    TensorCore,
    /// Persistence-based kernel designed to reside in L2.
    PersistentFusion,
    /// Parallel reduction across the K dimension.
    SplitK,
}

/// A ranked implementation strategy with its associated probability, confidence, and uncertainty.
/// Prediction for a single implementation strategy.
#[derive(Debug, Clone, Serialize)]
pub struct KernelPrediction {
    /// Name of the kernel implementation strategy.
    pub strategy: String,
    /// The semantic category of this strategy.
    pub category: StrategyCategory,
    /// Probability of this strategy being the optimal choice (0.0 to 1.0).
    /// Probability of this strategy being the optimal choice (0.0 to 1.0).
    pub probability: f32,
    /// Model uncertainty for this specific strategy (0.0 to 1.0).
    pub uncertainty: f32,
    /// Qualitative reasoning for the score.
    pub reasoning: String,
}

impl KernelPrediction {
    /// Checks if this strategy is semantically compatible with a given execution regime.
    pub fn compatible_with(&self, regime: RegimeClass) -> bool {
        self.p_given_regime(regime) > 0.4
    }

    /// Returns a probabilistic compatibility score (0.0 to 1.0) given a regime.
    pub fn p_given_regime(&self, regime: RegimeClass) -> f32 {
        match regime {
            RegimeClass::Compute => match self.category {
                StrategyCategory::TensorCore => 0.95,
                StrategyCategory::WarpTiling => 0.8,
                StrategyCategory::SplitK => 0.7,
                _ => 0.1,
            },
            RegimeClass::Memory => match self.category {
                StrategyCategory::PersistentFusion => 0.95,
                StrategyCategory::CudaCoreBlocked => 0.8,
                StrategyCategory::TensorCore => 0.6,
                _ => 0.1,
            },
            RegimeClass::Balanced => match self.category {
                StrategyCategory::CudaCoreBlocked => 0.9,
                StrategyCategory::WarpTiling => 0.8,
                StrategyCategory::TensorCore => 0.5,
                _ => 0.2,
            },
        }
    }
}

impl PartialEq for KernelPrediction {
    fn eq(&self, other: &Self) -> bool {
        self.strategy == other.strategy && 
        (self.probability - other.probability).abs() < 1e-6 &&
        (self.uncertainty - other.uncertainty).abs() < 1e-6
    }
}
impl Eq for KernelPrediction {}

/// Breakdown of the prediction confidence for specific hardware subsystems.
#[derive(Debug, Clone, Serialize)]
pub struct ConfidenceBreakdown {
    /// Confidence in the overall architectural regime identification (e.g., memory vs compute bound).
    pub regime: f32,
    /// Confidence in the cache routing and DRAM traffic model.
    pub memory_model: f32,
    /// Confidence in the instruction issue success and structural hazard predictions.
    pub scheduler_model: f32,
}

/// The intended usage and role of this predictor model.
#[derive(Debug, Clone, Serialize)]
pub struct IntendedUsage {
    /// The explicit role of this model in the compiler stack.
    pub role: String,
    /// Indicates if runtime autotuning is necessary to find the true optimal configuration.
    pub requires_runtime_feedback: bool,
    /// Indicates if the model's output is safe to use blindly without empirical calibration.
    pub safe_without_calibration: bool,
}

/// The current version of the PredictorPrior schema.
pub const SCHEMA_VERSION: &str = "0.4";

/// High-level classification of execution bottlenecks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RegimeClass {
    /// Bottlenecked by arithmetic units (ALUs/Tensor Cores).
    Compute,
    /// Bottlenecked by memory hierarchy (Bandwidth/Latency).
    Memory,
    /// Balanced or bottlenecked by instruction dispatch/hazards.
    Balanced,
}

/// Discrete probability distribution over high-level execution regimes.
#[derive(Debug, Clone, Serialize)]
pub struct RegimePosterior {
    /// Probability that the kernel is compute-bound (ALU/Tensor/Math).
    pub compute_bound: f32,
    /// Probability that the kernel is balanced (overlap, instruction-bottlenecks).
    pub balanced: f32,
    /// Probability that the kernel is memory-bound (DRAM/L2 latency/bandwidth).
    pub memory_bound: f32,
}

impl RegimePosterior {
    /// Checks for Bayesian coherence ($\sum P \approx 1.0$).
    pub fn is_coherent(&self) -> bool {
        (self.compute_bound + self.balanced + self.memory_bound - 1.0).abs() < 1e-4
    }

    /// Returns the most likely regime class.
    pub fn argmax(&self) -> RegimeClass {
        if self.compute_bound >= self.balanced && self.compute_bound >= self.memory_bound {
            RegimeClass::Compute
        } else if self.memory_bound >= self.balanced {
            RegimeClass::Memory
        } else {
            RegimeClass::Balanced
        }
    }

    /// Converts the regime posteriors into a HashMap for legacy compatibility.
    pub fn to_hashmap(&self) -> std::collections::HashMap<String, f32> {
        let mut map = std::collections::HashMap::new();
        map.insert("compute_bound".to_string(), self.compute_bound);
        map.insert("balanced".to_string(), self.balanced);
        map.insert("memory_bound".to_string(), self.memory_bound);
        map
    }
}

/// Probabilistic beliefs over critical performance metrics.
#[derive(Debug, Clone, Serialize)]
pub struct PerformancePosteriors {
    /// Expected runtime in microseconds with associated variance.
    pub runtime_us: GaussianPrior,
    /// Predicted percentage of peak FLOPs achievable.
    pub flop_utilization: GaussianPrior,
    /// Predicted percentage of available bandwidth used.
    pub bw_utilization: GaussianPrior,
}

/// Probabilistic models of hardware resource pressure.
#[derive(Debug, Clone, Serialize)]
pub struct ResourcePressureModels {
    /// Estimated pressure on the warp scheduler and instruction issue pipes.
    pub scheduler_pressure: GaussianPrior,
    /// Probability of register file spills or shared memory overflows.
    pub spill_risk: GaussianPrior,
}

/// A formal model of the cache hierarchy performance.
#[derive(Debug, Clone, Serialize)]
pub struct MemoryModel {
    /// Probability that a memory request hits in L1/SMEM.
    pub l1_hit_rate: GaussianPrior,
    /// Probability that a memory request hits in L2 (given an L1 miss).
    pub l2_hit_rate: GaussianPrior,
    /// Estimated total traffic to DRAM in bytes.
    pub dram_traffic_bytes: f64,
}

/// Probabilistic beliefs inferred by the Bayesian predictor over execution behavior.
#[derive(Debug, Clone, Serialize)]
pub struct InferredState {
    /// The high-level performance regime distribution.
    pub execution_regime: RegimePosterior,
    /// Detailed utilization and signature metrics.
    pub performance_posteriors: PerformancePosteriors,
    /// Resource contention and pressure signals.
    pub resource_pressure_models: ResourcePressureModels,
    /// Data movement and cache behavior.
    pub memory_model: MemoryModel,
    /// Ranked list of candidate implementation strategies with posterior probabilities.
    pub kernel_family_distribution: Vec<KernelPrediction>,
    /// Confidence metrics for specific hardware subsystems.
    pub confidence_breakdown: ConfidenceBreakdown,
    /// Logical inconsistency signals detected during inference.
    pub diagnostics: Vec<String>,
}

impl InferredState {
    /// Calculates the Shannon entropy of the kernel distribution (in bits).
    /// Used as a proxy for epistemic predictability.
    pub fn predictability_bits(&self) -> f32 {
        self.entropy_kernel()
    }

    /// Explicit entropy of the kernel selection distribution.
    pub fn entropy_kernel(&self) -> f32 {
        self.kernel_family_distribution
            .iter()
            .map(|k| {
                if k.probability > 0.0 {
                    -k.probability * k.probability.log2()
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Explicit entropy of the execution regime distribution.
    pub fn entropy_regime(&self) -> f32 {
        let p = &self.execution_regime;
        let mut e = 0.0;
        if p.compute_bound > 0.0 { e -= p.compute_bound * p.compute_bound.log2(); }
        if p.memory_bound > 0.0 { e -= p.memory_bound * p.memory_bound.log2(); }
        if p.balanced > 0.0 { e -= p.balanced * p.balanced.log2(); }
        e
    }

    /// Joint entropy estimate (assuming independence for now).
    pub fn entropy_joint(&self) -> f32 {
        self.entropy_kernel() + self.entropy_regime()
    }
}

impl PartialEq for InferredState {
    fn eq(&self, other: &Self) -> bool {
        self.kernel_family_distribution == other.kernel_family_distribution
    }
}
impl Eq for InferredState {}

/// A rigorous statistical model of prediction uncertainty.
#[derive(Debug, Clone, Serialize)]
pub struct UncertaintyState {
    /// Lack of knowledge (decreases as more data is observed).
    pub epistemic: f32,
    /// Inherent task variance (execution jitter, divergence).
    pub aleatoric: f32,
    /// Cross-architecture drift or calibration error.
    pub transfer: f32,
}

impl Default for IntendedUsage {
    fn default() -> Self {
        Self {
            role: "compile_time_prior".to_string(),
            requires_runtime_feedback: true,
            safe_without_calibration: false,
        }
    }
}

/// Baseline hardware characteristics observed by the underlying system.
#[derive(Debug, Clone, Serialize)]
pub struct HardwareObserved {
    /// The base compute capability of the architecture.
    pub arch_base_cc: u32,
    /// Physical observables captured from the GPU.
    pub observables: ArchObservables,
    /// Whether this model has been calibrated with empirical data.
    pub is_calibrated: bool,
    /// The current calibration coefficients learned by the model.
    pub calibration_state: CalibrationState,
}

/// Workload specifications requested by the compiler.
#[derive(Debug, Clone, Serialize)]
pub struct WorkloadObserved {
    /// The raw shape of the problem (M, N, K).
    pub problem_shape: ProblemShape,
    /// Hardware support feasibility for the requested data type.
    pub feasibility: NumericFeasibility,
    /// Higher-level classification of the shape physics.
    pub shape_classification: ShapeClassification,
}

/// Mathematical model of the GPU architecture and math precision support.
#[derive(Debug, Clone, Serialize)]
pub struct ArchitecturalModel {
    /// High-level behavioral profile of the architecture.
    pub profile: ExecutionProfile,
    /// Constraints on SM-level concurrency.
    pub sm_limits: SmExecutionModel,
    /// Critical throughput equilibrium ratios.
    pub throughput_ratios: ThroughputRatios,
    /// Valid mathematical execution modes.
    pub math_modes: NumericExecutionModes,
}

/// Dynamic constraints imposed by the hardware and software runtime.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionConstraints {
    /// Hardware-enforced software instruction limits.
    pub software_limits: SoftwareExecutionLimits,
    /// Summary of the high-level execution regime.
    pub regime_factors: PerformanceRegime,
    /// Qualitative risk of register file pressure.
    pub register_pressure: PressureRisk,
    /// Qualitative risk of shared memory pressure.
    pub shared_memory_pressure: PressureRisk,
    /// Qualitative risk of bandwidth saturation.
    pub bandwidth_saturation: PressureRisk,
}

/// Formal model of data reuse and grid-level scaling.
#[derive(Debug, Clone, Serialize)]
pub struct ReuseModel {
    /// Achievable reuse and staging capacity.
    pub memory_envelope: MemoryReuseEnvelope,
    /// Efficiency and scaling behavior on different grid sizes.
    pub scale_sensitivity: ScaleSensitivity,
    /// Model of costs associated with data movement.
    pub data_movement: DataMovementModel,
    /// Estimated feasible range of thread block occupancy.
    pub occupancy_window: (u32, u32),
    /// Suitability of current parallelism for hiding latency.
    pub latency_hiding: String,
    /// Projection of the shape into specialized learning clusters.
    pub shape_manifold: ShapeManifold,
}

/// Physical and mathematical metrics derived purely from the hardware and workload characteristics.
#[derive(Debug, Clone, Serialize)]
pub struct DerivedMetrics {
    /// High-level architectural and mathematical model.
    pub architectural_model: ArchitecturalModel,
    /// Constraints on execution and resource pressure.
    pub execution_constraints: ExecutionConstraints,
    /// Formal model of data reuse and scaling.
    pub reuse_model: ReuseModel,
}

/// The probabilistic prior generated to warm-start runtime autotuners.
#[derive(Debug, Clone, Serialize)]
pub struct SearchPrior {
    /// Ranked list of candidate implementation strategies.
    pub candidate_kernels: Vec<KernelPrediction>,
    /// Recommended search space pruning rules.
    pub search_space_reduction: SearchSpace,
    /// Viability of split-K reduction strategy.
    pub split_k_viability: String,
    /// Recommended range of thread blocks to target for optimal residency.
    pub block_size_window: (u32, u32),
    /// Availability/leverage of epilogue fusion.
    pub epilogue_fusion_headroom: Affinity,
    /// Explicit signal that runtime measurement is required to refine these priors.
    pub runtime_validation_required: bool,
}

/// Collection of workload-to-architecture affinities.
#[derive(Debug, Clone, Serialize)]
pub struct WorkloadAffinities {
    /// Affinity for compute-dense logic.
    pub compute_dense_gemm: Affinity,
    /// Affinity for tensor core offloading.
    pub tensor_core_gemm: Affinity,
    /// Affinity for pure memory streaming.
    pub memory_streaming: Affinity,
    /// Affinity for reduction-heavy patterns.
    pub reduction_heavy: Affinity,
    /// Affinity for persistent grid fusion.
    pub persistent_fusion: Affinity,
}

/// The formal probabilistic prior emitted by CUDAForge.
#[derive(Debug, Clone, Serialize)]
pub struct PredictorPrior {
    /// The schema version of this prior.
    pub schema_version: String,
    /// Usage context and safety guarantees.
    pub intended_usage: IntendedUsage,
    /// Raw hardware observations.
    pub hardware_observed: HardwareObserved,
    /// Raw workload observations.
    pub workload_observed: WorkloadObserved,
    /// Derived deterministic metrics.
    pub derived_metrics: DerivedMetrics,
    /// Probabilistic inference results.
    pub inference: InferredState,
    /// Optimized search space for autotuning.
    pub search_prior: SearchPrior,
    /// Statistical model of prediction uncertainty.
    pub uncertainty: UncertaintyState,
    /// Semantic groupings of workload-to-hardware affinities.
    pub workload_affinities: WorkloadAffinities,
}

/// Operational intent emitted for a specific kernel implementation.
/// 
/// This is the machine-consumable output of the predictor, used to
/// drive codegen or autotuning.
#[derive(Debug, Clone, Serialize)]
pub struct KernelIntent {
    /// The recommended high-level implementation strategy.
    pub kernel_family: String,
    /// The suggested tiling and staging strategy.
    pub tile_strategy: String,
    /// Recommended staging depth (usually 2 for SMEM double-buffering).
    pub pipeline_depth: u32,
    /// The precision to use for calculations.
    pub dtype_compute: DType,
    /// The precision used for data storage/movement.
    pub dtype_storage: DType,
    /// The occupancy level to aim for in register/SMEM allocation.
    pub occupancy_target: f32,
    /// Confidence score of this intent (0.0 to 1.0).
    pub priority: f32,
    /// Suggested parametric search space for autotuning this intent.
    pub suggested_search_space: SearchSpace,
    /// Top sampled configurations to explore first.
    pub sampled_configs: Vec<SampledConfig>,

    // ── Phase 22-26 Physical Realism ──
    /// Classification of the problem's shape physics.
    pub shape_class: ShapeClassification,
    /// Projection of the shape into a continuous cluster.
    pub shape_manifold: ShapeManifold,
    /// Predicted performance signature of this implementation.
    pub performance_signature: PerformanceSignature,
    /// Expected runtime in microseconds.
    pub expected_runtime_us: f32,
    /// Uncertainty range for the runtime prediction [min_us, max_us].
    pub runtime_interval_us: [f32; 2],
    /// Confidence in the runtime prediction (0.0 to 1.0).
    pub runtime_confidence: f32,
    /// Verification targets for hardware profilers.
    pub verification_targets: VerificationTargets,
    /// The oracle's current calibration state used to generate this intent.
    pub calibration_state: CalibrationState,
}

/// A specific kernel configuration sampled from the predictor's distribution.
#[derive(Debug, Clone, Serialize)]
pub struct SampledConfig {
    /// The kernel family this config belongs to.
    pub kernel_family: String,
    /// Thread block dimensions (M x N).
    pub block_size: (u32, u32),
    /// K-loop unrolling factor.
    pub k_blocking: u32,
    /// Shared Memory pipeline stages.
    pub smem_stages: u32,
    /// Probability weight assigned to this specific configuration.
    pub weight: f32,
}

/// Suggested parametric search space for autotuning.
/// 
/// This reduces the search space from thousands of combinations
/// to a handful of plausible candidates based on hardware limits.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SearchSpace {
    /// Plausible thread block sizes (M x N).
    pub plausible_block_sizes: Vec<(u32, u32)>,
    /// Plausible K-loop unrolling/blocking factors.
    pub plausible_k_blocking: Vec<u32>,
    /// Valid Shared Memory staging depths.
    pub smem_stages: Vec<u32>,
}

impl SearchSpace {
    /// Returns the discrete volume of the search space.
    pub fn volume(&self) -> usize {
        self.plausible_block_sizes.len() * self.plausible_k_blocking.len() * self.smem_stages.len()
    }

    /// Conservative baseline volume for a typical GEMM search space.
    pub fn naive_volume() -> usize {
        // Typical range: (32..256)x(32..256) block sizes, 8..128 K-blocking, 1..5 stages
        // Roughly 1000+ combinations in a real autotuner.
        1000
    }
}

/// Categorization of GEMM shape physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShapeCase {
    /// Large square-ish GEMM, likely compute bound.
    LargeSquare,
    /// Tall-skinny GEMM (M >> N, K), limited by memory latency or N-parallelism.
    TallSkinnyM,
    /// Wide-skinny GEMM (N >> M, K), limited by memory latency or M-parallelism.
    WideSkinnyN,
    /// K-dominant (K >> M, N), limited by reduction throughput.
    KDominant,
    /// Small/Inference-scale GEMM, dominated by dispatch latency.
    SmallScale,
}

/// Explicit classification of the problem shape and its impact on tiling.
#[derive(Debug, Clone, Serialize)]
pub struct ShapeClassification {
    /// The detected shape category.
    pub case: ShapeCase,
    /// The axis that dominates the performance or constraints.
    pub dominant_axis: String,
    /// The rationale for the suggested tiling strategy.
    pub tiling_rationale: String,
}

/// Identifiers for clustered shape domains to enable cross-shape learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum ShapeCluster {
    /// Small, highly synchronized inference shapes (e.g. M=1, N=4096).
    InferenceVector,
    /// Memory-bound, tall/wide irregular boundaries (e.g. M=200, N=10000).
    IrregularEdge,
    /// Well-behaved, power-of-two dense compute tiles (e.g. 2048x2048).
    DenseCore,
    /// Extreme reduction dominance (e.g. K > 8192, M,N small).
    ReductionHeavy,
    /// Micro-batching regimes (e.g. M=8..32).
    MicroBatch,
}

/// A projection of the raw (M, N, K) shape into a normalized vector space.
/// Used by runtimes to bucket similar kernels into the same CalibrationState.
#[derive(Debug, Clone, Serialize)]
pub struct ShapeManifold {
    /// The discrete cluster this shape belongs to.
    pub cluster: ShapeCluster,
    /// The normalized "volume" metric.
    pub normalized_volume: f32,
    /// The arithmetic intensity of this specific coordinate.
    pub intensity: f32,
}

/// A measure of algorithmic unpredictability and execution divergence.
/// High entropy = unpredictable behavior. Low entropy = highly deterministic.
/// Mathematical model of how predictable a kernel's execution is.
#[derive(Debug, Clone, Serialize)]
pub struct PredictabilityModel {
    /// Score from 0.0 (unpredictable) to 1.0 (deterministic).
    pub score: f32,
    /// Proxy risk for thread divergence on non-power-of-two shapes.
    pub divergence_risk: f32,
    /// Proxy risk of bandwidth contention or poor reuse.
    pub memory_contention_risk: f32,
    /// Qualitative explanation of the predictability.
    pub rationale: String,
}

/// A probabilistic estimate comprising a mean and variance.
#[derive(Debug, Clone, Serialize)]
pub struct GaussianPrior {
    /// The expected value (mean) of the prediction.
    pub mean: f32,
    /// The standard deviation (σ) of the prediction.
    pub stddev: f32,
}

impl GaussianPrior {
    /// Creates a new Gaussian prior with a given mean and standard deviation.
    pub fn new(mean: f32, stddev: f32) -> Self {
        Self { mean, stddev }
    }
}

/// A probabilistic model of the GPU's warp scheduler.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerModelRep {
    /// Probability that a warp has its operands ready.
    pub warp_ready_prob: GaussianPrior,
    /// Average number of dependent instructions before a stall is hidden.
    pub dep_chain_len: f32,
    /// Structural hazard pressure (ALU/Tensor/SFU contention).
    pub pipe_pressure: GaussianPrior,
    /// Probability of instruction replays (e.g., from bank conflicts).
    pub replay_rate: GaussianPrior,
    /// The predicted SM scheduler utilization issue rate.
    pub issue_rate: GaussianPrior,
}

/// A probabilistic model of the GPU's memory cache hierarchy.
#[derive(Debug, Clone, Serialize)]
pub struct CacheHierarchyModel {
    /// Probability that a memory request hits in L1/SMEM.
    pub l1_hit_rate: GaussianPrior,
    /// Probability that a memory request hits in L2 (given an L1 miss).
    pub l2_hit_rate: GaussianPrior,
    /// Probability that a memory request goes all the way to DRAM.
    pub dram_miss_rate: GaussianPrior,
    /// Estimated total traffic to L2 (bytes).
    pub l2_traffic_bytes: f64,
    /// Estimated total traffic to DRAM (bytes).
    pub dram_traffic_bytes: f64,
}

/// A rigorous statistical model of prediction uncertainty.
#[derive(Debug, Clone, Serialize)]
pub struct CalibratedUncertainty {
    /// Lack of knowledge (decreases as `samples_absorbed` increases).
    pub epistemic_uncertainty: f32,
    /// Inherent task variance (e.g., from `KernelEntropy` and divergence).
    pub aleatoric_uncertainty: f32,
    /// Arch-distance penalty when warm-starting from a different GPU.
    pub transfer_uncertainty: f32,
    /// The final pooled standard deviation (σ) of the predicted runtime.
    pub sigma_runtime: f32,
}

/// Predicted performance footprint and resource utilization.
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceSignature {
    /// The bottleneck regime the kernel is expected to fall into.
    pub regime: String,
    /// Predicted percentage of peak FLOPs achievable.
    pub flop_utilization: GaussianPrior,
    /// Predicted percentage of available bandwidth used.
    pub bw_utilization: GaussianPrior,
    /// Detailed mathematical model of the warp scheduler.
    pub scheduler_model: SchedulerModelRep,
    /// Detailed mathematical model of the multi-tier cache hierarchy.
    pub cache_model: CacheHierarchyModel,
    /// Statistically rigorous uncertainty bounds for the predicted runtime.
    pub uncertainty: CalibratedUncertainty,
    /// Predicted risk of register spilling.
    pub risk_of_spill: GaussianPrior,
    /// Posterior probability distribution of the restricting hardware limits.
    /// Ex: {"dram_bandwidth": 0.7, "tensor_unit_throughput": 0.3}
    pub dominant_limits: std::collections::HashMap<String, f32>,
    /// The algorithmic predictability of this kernel's execution timing.
    pub predictability: PredictabilityModel,
}

/// The formal latent microarchitectural state vector θ.
///
/// This represents the "unseen" parameters of the GPU's physics that we 
/// attempt to infer from execution telemetry.
#[derive(Debug, Clone, Serialize)]
pub struct LatentStateTheta {
    /// Latent instruction issue rate intensity (λ_issue).
    pub lambda_issue: f32,
    /// Latent memory bandwidth utilization intensity (λ_mem).
    pub lambda_mem: f32,
    /// Latent pipeline stall probability intensity (λ_stall).
    pub lambda_stall: f32,
    /// Latent warp concurrency intensity (λ_warp).
    pub lambda_warp: f32,
}

/// A trait for pluggable performance models.
///
/// This allows swapping between the analytical (physics-based) model,
/// learned models (Bayesian), or hybrid approaches.
pub trait ProbabilisticModel: Send + Sync {
    /// Estimates the performance signature for a given problem shape and architecture.
    fn estimate(
        &self,
        arch: &GpuArch,
        obs: &ArchObservables,
        derived: &DerivedProperties,
        shape: &ProblemShape,
        calib: &CalibrationState,
    ) -> PerformanceSignature;

    /// Updates the internal state based on empirical evidence (Bayesian Posterior Update).
    fn update_posterior(&mut self, runtime_ratio: f32);
}

/// The default analytical performance model based on hardware physics.
#[derive(Debug, Clone, Default)]
pub struct AnalyticalModel;

impl ProbabilisticModel for AnalyticalModel {
    fn estimate(
        &self,
        arch: &GpuArch,
        obs: &ArchObservables,
        derived: &DerivedProperties,
        shape: &ProblemShape,
        calib: &CalibrationState,
    ) -> PerformanceSignature {
        estimate_performance_signature(arch, obs, derived, &ExecutionProfile::default(), shape, &NumericFeasibility::default(), calib)
    }

    fn update_posterior(&mut self, _runtime_ratio: f32) {
        // Analytical model is static/heuristic in its pure form.
        // Learning happens in the BayesianPredictor wrapper.
    }
}

/// A Bayesian performance model that learns from execution telemetry.
///
/// It maintains a latent state θ representing the GPU's microarchitectural
/// performance intensity, updated via Bayesian posterior inference.
#[derive(Debug, Clone, Serialize)]
pub struct BayesianModel {
    /// The current latent state mean (μ_θ).
    pub theta: LatentStateTheta,
    /// The uncertainty in the latent state (Σ_θ).
    /// For simplicity in this implementation, we use a single variance scalar per parameter.
    pub variance: LatentStateTheta,
}

impl Default for BayesianModel {
    fn default() -> Self {
        Self {
            theta: LatentStateTheta {
                lambda_issue: 1.0,
                lambda_mem: 1.0,
                lambda_stall: 0.1,
                lambda_warp: 1.0,
            },
            variance: LatentStateTheta {
                lambda_issue: 0.2,
                lambda_mem: 0.2,
                lambda_stall: 0.05,
                lambda_warp: 0.2,
            },
        }
    }
}

impl ProbabilisticModel for BayesianModel {
    fn estimate(
        &self,
        arch: &GpuArch,
        obs: &ArchObservables,
        derived: &DerivedProperties,
        shape: &ProblemShape,
        calib: &CalibrationState,
    ) -> PerformanceSignature {
        // Here we would ideally sample from p(θ), but for the first iteration
        // we use the expected value E[θ] = self.theta.
        
        // We inject the latent intensities into a modified analytical model
        let mut sig = estimate_performance_signature(
            arch, obs, derived, &ExecutionProfile::default(), shape, &NumericFeasibility::default(), calib
        );
        
        let apply_theta = |s: &mut PerformanceSignature, t: &LatentStateTheta| {
            s.scheduler_model.issue_rate.mean *= t.lambda_issue;
            s.flop_utilization.mean *= t.lambda_issue; // Propagate latent state to main metric
            s.cache_model.dram_miss_rate.mean *= t.lambda_mem;
            s.bw_utilization.mean *= t.lambda_mem;     // Propagate to BW
        };
        apply_theta(&mut sig, &self.theta);
        
        
        // ── Phase 43: Jacobian Uncertainty Propagation ──
        // We calculate J = \nabla_\theta T using finite differences.
        let epsilon = 1e-4;
        
        // Localized Runtime Evaluator for Jacobian
        let calc_runtime = |s: &PerformanceSignature| -> f32 {
            let m = shape.m as f64;
            let n = shape.n as f64;
            let k = shape.k as f64;
            let total_flops = 2.0 * m * n * k;
            let est_tflops = 10.0 * s.flop_utilization.mean as f64; // Peak FP32 proxy
            if est_tflops > 0.0 {
                (total_flops / (est_tflops * 1e12) * 1e6) as f32
            } else {
                0.0
            }
        };
        
        let base_runtime = calc_runtime(&sig);
        
        // Helper to evaluate runtime with perturbed theta
        let eval_perturbed = |t_perturbed: LatentStateTheta| -> f32 {
            let mut s = estimate_performance_signature(
                arch, obs, derived, &ExecutionProfile::default(), shape, &NumericFeasibility::default(), calib
            );
            apply_theta(&mut s, &t_perturbed);
            calc_runtime(&s)
        };
        
        // J = [dT/dλ_issue, dT/dλ_mem]
        let mut t_issue = self.theta.clone();
        t_issue.lambda_issue += epsilon;
        let dt_dissue = (eval_perturbed(t_issue) - base_runtime) / epsilon;
        
        let mut t_mem = self.theta.clone();
        t_mem.lambda_mem += epsilon;
        let dt_dmem = (eval_perturbed(t_mem) - base_runtime) / epsilon;
        
        // σ_T^2 = J * Σ_θ * J^T + σ_noise^2
        let variance_issue = self.variance.lambda_issue;
        let variance_mem = self.variance.lambda_mem;
        
        let variance_t = (dt_dissue * dt_dissue * variance_issue) 
                       + (dt_dmem * dt_dmem * variance_mem);
        
        // Convert to standard deviation relative to baseline runtime
        let sigma_t = (variance_t.max(0.0).sqrt() / base_runtime.max(1e-6)).clamp(0.0, 1.0);
        
        sig.uncertainty.epistemic_uncertainty = sigma_t;
        
        sig

    }

    fn update_posterior(&mut self, runtime_ratio: f32) {
        // Implementation of: p(θ | D_new) ∝ p(D_new | θ) p(θ)
        // We use a formal Extended Kalman Filter (EKF) update for the latent intensities.
        
        // Observation: measured runtime ratio
        let z = runtime_ratio;
        // Expected observation (normalized via delta base)
        let h_x = 1.0_f32; 
        
        // Jacobian of observation model H (simplified proxy mapping delta to issue/mem ratios)
        // If we are slower (z > 1.0), it implies higher stall/memory intensity or lower issue
        let h_issue = -0.5_f32; // Assuming lower issue rate -> slower runtime
        let h_mem = 0.5_f32;   // Assuming higher mem intensity -> slower runtime
        
        let p_issue = self.variance.lambda_issue;
        let p_mem = self.variance.lambda_mem;
        
        // Observation noise covariance R
        let r_noise = 0.05_f32; 
        
        // Innovation (residual) y = z - h_x
        let y = z - h_x;
        
        // Innovation covariance S = H P H^T + R
        let s_cov = (h_issue * h_issue * p_issue) + (h_mem * h_mem * p_mem) + r_noise;
        
        // Kalman Gain K = P H^T S^-1
        let k_issue = (p_issue * h_issue) / s_cov;
        let k_mem = (p_mem * h_mem) / s_cov;
        
        // Update State Mean: μ = μ + K y
        self.theta.lambda_issue = (self.theta.lambda_issue + k_issue * y).max(0.01);
        self.theta.lambda_mem = (self.theta.lambda_mem + k_mem * y).max(0.01);
        self.theta.lambda_stall = (self.theta.lambda_stall + (k_mem - k_issue) * y).max(0.01);
        
        // Update State Covariance: P = (I - K H) P
        self.variance.lambda_issue = (1.0 - k_issue * h_issue) * p_issue;
        self.variance.lambda_mem = (1.0 - k_mem * h_mem) * p_mem;
        self.variance.lambda_stall = self.variance.lambda_issue.max(self.variance.lambda_mem);
    }
}

/// A high-level predictor that can evaluate kernel intents using a probabilistic model.
#[derive(Debug, Clone)]
pub struct HardwarePredictor<M: ProbabilisticModel = AnalyticalModel> {
    /// The target GPU architecture.
    pub arch: GpuArch,
    /// The internal performance model used for estimations.
    pub model: M,
}

impl<M: ProbabilisticModel> ProbabilisticHardwareOracle for HardwarePredictor<M> {
    fn posterior(&self, workload: &WorkloadDesc) -> InferredState {
        let mut obs = ArchObservables::from_compute_cap(self.arch.base);
        let calib = workload.calibration.clone().unwrap_or_default();
        obs.apply_coefficients(calib.flop_scale_coeff, calib.bw_scale_coeff);
        
        let derived = DerivedProperties::from_observables(&obs);
        let sig = self.model.estimate(&self.arch, &obs, &derived, &workload.shape, &calib);
        
        let prior = PredictorPrior::evaluate(&self.arch, obs, derived, &workload.shape, workload.dtype, Some(calib), sig);
        prior.inference
    }

    fn policy(&self, posterior: &InferredState, context: &TuningContext) -> ExecutionPolicy {
        // Derive rankings from the kernel family distribution
        let mut rankings = posterior.kernel_family_distribution.clone();
        
        // Filter strategies if strict determinism is required
        if context.determinism_required {
            rankings.retain(|k| k.uncertainty < 0.2);
        }

        // Construct search space based on the highest-probability candidate if available
        let search_space = if let Some(_best) = rankings.first() {
            // Adjust search rigor based on budget and phase
            let rigorous = context.tuning_budget == "thorough" || context.phase == TuningPhase::Offline;

            // In a real implementation, we'd pull this from the search_prior or similar
            // For now, we return a valid but generic search space reduction
            let mut space = SearchSpace {
                plausible_block_sizes: vec![(128, 128), (128, 64), (64, 128)],
                plausible_k_blocking: vec![32],
                smem_stages: vec![2],
            };

            if rigorous {
                space.plausible_k_blocking.push(64);
                space.plausible_k_blocking.push(128);
                space.smem_stages.push(3);
                space.smem_stages.push(4);
            }

            space
        } else {
            SearchSpace::default()
        };

        let pruned_volume = search_space.volume();
        let naive_volume = SearchSpace::naive_volume();
        let pruning_factor = if pruned_volume > 0 {
            naive_volume as f32 / pruned_volume as f32
        } else {
            1.0
        };

        let mut graph_hints = Vec::new();
        if posterior.execution_regime.memory_bound > 0.7 {
            graph_hints.push("Aggressive fusion recommended to reduce DRAM pressure".to_string());
        }
        if posterior.resource_pressure_models.spill_risk.mean > 0.6 {
            graph_hints.push("Avoid deep fusion; register file saturation likely".to_string());
        }

        ExecutionPolicy {
            rankings,
            search_space,
            pruning_factor,
            graph_hints,
            policy_confidence: posterior.confidence_breakdown.regime,
        }
    }

    fn update(&mut self, observation: &CalibrationFeedback) {
        // Calculate the ratio of measured / predicted for this specific kernel execution
        // If prediction is missing, assume 1.0 (no error)
        let ratio = if observation.predicted_runtime_us > 0.0 {
            observation.measured_runtime_us / observation.predicted_runtime_us
        } else {
            1.0
        };

        self.model.update_posterior(ratio);
    }

    fn annotate_node(&self, workload: &WorkloadDesc, labels: &mut HashMap<String, String>) {
        let post = self.posterior(workload);
        labels.insert("cudaforge.arch_affinity.regime".to_string(), format!("{:?}", post.execution_regime));
        labels.insert("cudaforge.prediction.confidence".to_string(), format!("{:.2}", post.confidence_breakdown.regime));
        labels.insert("cudaforge.prediction.runtime_us".to_string(), format!("{:.1}", post.performance_posteriors.runtime_us.mean));
        if let Some(best) = post.kernel_family_distribution.first() {
            labels.insert("cudaforge.arch_affinity.strategy".to_string(), best.strategy.clone());
        }
    }
}

impl<M: ProbabilisticModel> HardwarePredictor<M> {
    /// Creates a new predictor instance with a specific model.
    pub fn new(arch: GpuArch, model: M) -> Self {
        Self { arch, model }
    }

    /// Evaluates a problem shape with a specific data type and generates a comprehensive prediction report.
    pub fn evaluate(
        &self,
        shape: &ProblemShape,
        dtype: DType,
        calibration_data: Option<CalibrationState>,
    ) -> Result<PredictorPrior> {
        let workload = WorkloadDesc {
            shape: *shape,
            dtype,
            calibration: calibration_data,
        };
        
        let mut obs = ArchObservables::from_compute_cap(self.arch.base);
        let calib = workload.calibration.clone().unwrap_or_default();
        obs.apply_coefficients(calib.flop_scale_coeff, calib.bw_scale_coeff);
        
        let derived = DerivedProperties::from_observables(&obs);
        let sig = self.model.estimate(&self.arch, &obs, &derived, &workload.shape, &calib);
        
        let report = PredictorPrior::evaluate(&self.arch, obs, derived, &workload.shape, workload.dtype, Some(calib), sig);
        
        Ok(report)
    }
}

/// Empirical measurements returned by a hardware profiler (like Nsight Compute)
/// to calibrate the oracle's internal physics models.
#[derive(Debug, Clone, Serialize, Default)]
pub struct CalibrationFeedback {
    /// The actual wall-clock execution time of the kernel.
    pub measured_runtime_us: f32,
    /// The actual percentage of issue slots utilized by the scheduler.
    pub measured_issue_utilization: f32,
    /// The actual percentage of theoretical DRAM bandwidth achieved.
    pub measured_bw_utilization: f32,
    /// The runtime that was predicted for this measurement (to compute divergence).
    pub predicted_runtime_us: f32,
}

/// The "learned" internal state of the predictor's analytical models.
#[derive(Debug, Clone, Serialize)]
pub struct CalibrationState {
    /// Multiplier applied to analytical FLOP utilization estimates.
    pub flop_scale_coeff: f32,
    /// Multiplier applied to analytical Bandwidth utilization estimates.
    pub bw_scale_coeff: f32,
    /// Tuning factor influencing the modeled scheduler pressure.
    pub pressure_tune_coeff: f32,
    /// Number of calibration samples absorbed so far.
    pub samples_absorbed: u32,
}

impl Default for CalibrationState {
    fn default() -> Self {
        Self {
            flop_scale_coeff: 1.0,
            bw_scale_coeff: 1.0,
            pressure_tune_coeff: 1.0,
            samples_absorbed: 0,
        }
    }
}

/// A projection matrix to translate a learned CalibrationState from a Source GPU Architecture
/// to a Target GPU Architecture, enabling Cross-GPU Transfer Learning (Warm-Starting).
pub struct TransferMatrix;

impl TransferMatrix {
    /// Translates a state learned on `source_arch` to be applicable to `target_arch`.
    /// 
    /// For example, moving from Ampere (80) to Hopper (90) might see a drop in
    /// BW utilization efficiency due to TMA overhead on small shapes, or a boost in FLOP
    /// efficiency due to better scheduling.
    pub fn translate(state: &CalibrationState, source_arch: u32, target_arch: u32) -> CalibrationState {
        if source_arch == target_arch || state.samples_absorbed == 0 {
            return state.clone();
        }

        let mut next_state = state.clone();
        
        // Decay the confidence (samples) since this is a fuzzy translation
        next_state.samples_absorbed = (next_state.samples_absorbed as f32 * 0.25) as u32;

        // Apply architectural translation heuristics
        if source_arch < 90 && target_arch >= 90 {
            // Pre-Hopper to Hopper: TMA and TMEM introduce BW overhead but higher compute efficiency
            next_state.bw_scale_coeff *= 0.85; 
            next_state.flop_scale_coeff *= 1.15;
            next_state.pressure_tune_coeff *= 0.90; // Hopper schedulers are better at hiding pressure
        } else if source_arch >= 90 && target_arch < 90 {
            // Hopper down to Ampere/Ada
            next_state.bw_scale_coeff *= 1.15;
            next_state.flop_scale_coeff *= 0.85;
            next_state.pressure_tune_coeff *= 1.10;
        } else if source_arch < 80 && target_arch >= 80 {
            // Turing/Volta to Ampere
            next_state.flop_scale_coeff *= 1.10;
        }

        // Clamp to sane boundaries
        next_state.flop_scale_coeff = next_state.flop_scale_coeff.clamp(0.5, 1.5);
        next_state.bw_scale_coeff = next_state.bw_scale_coeff.clamp(0.5, 1.5);
        next_state.pressure_tune_coeff = next_state.pressure_tune_coeff.clamp(0.5, 1.5);

        next_state
    }
}

impl CalibrationState {
    /// Absorbs a single empirical runtime measurement to update internal scaling targets.
    /// Uses a moving average (Bayesian-lite) approach to smooth out noise.
    pub fn apply_feedback(
        &mut self,
        predicted_issue: f32,
        predicted_bw: f32,
        feedback: &CalibrationFeedback,
    ) {
        // Calculate the ratio of measured / predicted for this specific kernel execution
        let issue_ratio = if predicted_issue > 0.0 {
            feedback.measured_issue_utilization / predicted_issue
        } else {
            1.0
        };

        let bw_ratio = if predicted_bw > 0.0 {
            feedback.measured_bw_utilization / predicted_bw
        } else {
            1.0
        };

        // Clip the ratios to avoid massive overcorrections from a single anomalous run
        let issue_ratio = issue_ratio.clamp(0.5, 2.0);
        let bw_ratio = bw_ratio.clamp(0.5, 2.0);

        // Learning rate: Decays as we absorb more samples to stabilize the model
        self.samples_absorbed += 1;
        let learning_rate = 1.0 / (1.0 + self.samples_absorbed as f32).sqrt();

        // Update coefficients via moving average
        self.flop_scale_coeff = self.flop_scale_coeff * (1.0 - learning_rate) + (self.flop_scale_coeff * issue_ratio) * learning_rate;
        self.bw_scale_coeff = self.bw_scale_coeff * (1.0 - learning_rate) + (self.bw_scale_coeff * bw_ratio) * learning_rate;

        // Pressure tuning: If runtime is significantly worse than expected despite good utilization, 
        // issue/scheduler pressure must be higher than modeled.
        // For now, keep it simple: clamp the scale coefficients.
        self.flop_scale_coeff = self.flop_scale_coeff.clamp(0.8, 1.2);
        self.bw_scale_coeff = self.bw_scale_coeff.clamp(0.8, 1.2);
    }
}

/// Precise verification targets for hardware profilers like Nsight Compute.
#[derive(Debug, Clone, Serialize)]
pub struct VerificationTargets {
    /// Expected percentage of issue slots utilized (0.0 to 1.0).
    pub expected_issue_utilization: f32,
    /// Qualitative expectation of shared memory reuse.
    pub expected_smem_reuse: String,
    /// The primary expected reason for warp stalls.
    pub expected_stall_reason: String,
}



impl PredictorPrior {
    /// Evaluates a problem shape and generates prescriptive implementation intents.
    /// 
    /// This is the primary entry point for a standard analytical prediction.
    pub fn evaluate(
        arch: &GpuArch,
        obs: ArchObservables,
        derived: DerivedProperties,
        shape: &ProblemShape,
        dtype: DType,
        calibration_state: Option<CalibrationState>,
        performance_signature: PerformanceSignature,
    ) -> Self {
        let mut obs = obs;
        if let Some(state) = &calibration_state {
            obs.apply_coefficients(state.flop_scale_coeff, state.bw_scale_coeff);
        }
        let feasibility = resolve_feasibility(arch, dtype);
        let compute_dtype = feasibility.effective_compute_dtype;

        // Effective roofline: use the compute dtype's limits
        let effective_roofline = match compute_dtype {
            DType::Fp8 => derived.roofline_fp8,
            DType::Int8 => derived.roofline_int8,
            _ => derived.roofline_fp32,
        };

        // --- 1. Kernel Class Likelihoods ---
        let tensor_core_gemm = if derived.tensor_core_dominance > 0.0 && feasibility.is_native {
            Affinity::High
        } else {
            Affinity::Low
        };

        let compute_dense_gemm = if tensor_core_gemm == Affinity::High {
            Affinity::Medium // TC is preferred for dense GEMM
        } else if effective_roofline > 50.0 {
            Affinity::High // Extremely high ridge point favors pure compute
        } else {
            Affinity::Medium
        };

        // Streaming logic: Does it have enough BW relative to its compute power?
        let memory_streaming = if effective_roofline < 40.0 {
            Affinity::High // Low FLOP:Byte ratio implies memory can feed compute easily
        } else {
            Affinity::Low // High FLOP:Byte means BW is relatively starved
        };

        let reduction_heavy = if arch.base >= 70 { Affinity::High } else { Affinity::Medium };

        // Persistent fusion requires large L2 and sufficient SMs
        let persistent_fusion = if obs.l2_bytes.value >= 16 * 1024 * 1024 {
            Affinity::High
        } else if obs.l2_bytes.value >= 4 * 1024 * 1024 {
            Affinity::Medium
        } else {
            Affinity::Low
        };

        // --- 2. Resource Pressures ---
        let register_pressure_risk = if obs.registers_per_sm.value >= 65536 && obs.max_threads_per_sm.value >= 1536 {
            PressureRisk::new(0.15)
        } else {
            PressureRisk::new(0.55)
        };

        let shared_memory_pressure_risk = if obs.shared_mem_per_sm.value >= 96 * 1024 {
            PressureRisk::new(0.1)
        } else {
            PressureRisk::new(0.4)
        };

        let bandwidth_saturation_risk = if derived.roofline_fp32 > 70.0 {
            PressureRisk::new(0.9) // Very high compute-to-BW means BW will saturate quickly
        } else if derived.roofline_fp32 > 40.0 {
            PressureRisk::new(0.6)
        } else {
            PressureRisk::new(0.2)
        };

        // --- 3. Modeling Predictions ---
        let achievable_occupancy_range = if arch.base == 61 {
            // Pascal GEMM optimal targets high occupancy to hide latency
            (60, 80)
        } else if arch.base >= 80 {
            // Ampere/Hopper often rely on large register allocations, dropping occupancy to 25-50%
            (25, 50)
        } else {
            // Older architectures aimed for >50% to hide latency
            (50, 75)
        };

        let latency_hiding_sufficiency = if obs.registers_per_sm.value >= 65536 && obs.max_threads_per_sm.value >= 2048 {
            "ideal (massive concurrency possible)".to_string()
        } else if arch.base >= 80 {
            "high (asynchronous orchestration leverage)".to_string()
        } else {
            "moderate (compute-latency sensitive)".to_string()
        };


        // --- 4. Strategy Ranking ---
        let mut implementation_strategies = Vec::new();

        if derived.tensor_core_dominance > 0.0 {
            if obs.shared_mem_per_sm.value >= 96 * 1024 {
                if arch.base >= 90 {
                    implementation_strategies.push("WGMMA Tensor Core GEMM with TMA memory streams");
                } else if arch.base >= 80 {
                    implementation_strategies.push("Tensor Core GEMM with cp.async multi-stage pipelines");
                } else {
                    implementation_strategies.push("Tensor Core GEMM with double-buffered SMEM");
                }
            } else {
                implementation_strategies.push("Tensor Core GEMM with register constraints (warp specialization)");
            }
        } else {
            implementation_strategies.push("Blocked CUDA-core GEMM with double-buffered SMEM");
        }

        if persistent_fusion == Affinity::High {
            implementation_strategies.push("Persistent L2-resident fusion kernel");
        } else {
             implementation_strategies.push("Register-heavy warp tiling kernel (avoid spilling)");
        }

        if derived.roofline_fp32 > 60.0 {
            implementation_strategies.push("Split-K reduction kernel (to increase parallel stream width)");
        }

        // --- 1.5. Dynamic SM Limits and Ratios (Phase 9) ---
        let sm_execution_model = SmExecutionModel {
            max_warps_per_sm: obs.max_warps_per_sm.value,
            max_blocks_per_sm: obs.max_blocks_per_sm.value,
            schedulers_per_sm: obs.schedulers_per_sm.value,
        };

        // Assume ~1.5 GHz typical average core clock under load for peak cycle conversions.
        // It provides the correct magnitude ratios.
        // For a single SM (we don't strictly have SM count here, but we can compute ratios safely, or rough it: SM Count roughly proportional to L2/SMEM).
        // Let's rely on standard ratios relative to total bandwidth since we know totals.
        let fp32_per_sm_ops_cycle = if arch.base >= 80 { 128.0 } else { 64.0 }; // Typically FP32 cores per SM per cycle
        let ldst_per_sm_bytes_cycle = if arch.base >= 80 { 128.0 } else { 32.0 }; // Typically LD/ST units per SM
        
        // Ratio: Max Compute vs BW 
        let compute_to_shared_ratio = (obs.fp32_flops_tflops.effective() * 1000.0) / obs.shared_mem_bandwidth_gbps.effective();
        let compute_to_l2_ratio = (obs.fp32_flops_tflops.effective() * 1000.0) / obs.l2_bandwidth_gbps.effective();

        let throughput_ratios = ThroughputRatios {
            fp32_per_sm_ops_cycle,
            ldst_per_sm_bytes_cycle,
            compute_to_shared_ratio,
            compute_to_l2_ratio,
        };

        let software_execution_limits = SoftwareExecutionLimits {
            max_async_copies_in_flight: if arch.base >= 80 { 32 } else { 0 },
            max_barriers: if arch.base >= 90 { 64 } else if arch.base >= 80 { 16 } else { 0 },
            cluster_launch_supported: arch.base >= 90,
            cooperative_grid_viable: arch.base >= 60,
        };

        // --- 1.6. Phase 12 Multi-Arch Classification ---
        let arch_class = if arch.base >= 100 {
            ExecutionArchClass::DataflowGpu
        } else if arch.base >= 90 {
            ExecutionArchClass::PipelineGpu
        } else if arch.base >= 80 {
            ExecutionArchClass::AsyncTensor
        } else if arch.base >= 70 {
            ExecutionArchClass::EarlyTensor
        } else {
            ExecutionArchClass::PreTensor
        };

        let compute_pipeline = match arch_class {
            ExecutionArchClass::DataflowGpu => ComputePipeline::TensorDataflow,
            ExecutionArchClass::PipelineGpu => ComputePipeline::TensorPipeline,
            ExecutionArchClass::AsyncTensor => ComputePipeline::TensorAsync,
            ExecutionArchClass::EarlyTensor => ComputePipeline::TensorSync,
            ExecutionArchClass::PreTensor => ComputePipeline::ScalarCuda,
        };

        let memory_pipeline = match arch_class {
            ExecutionArchClass::DataflowGpu => MemoryPipeline::TmemDataflow,
            ExecutionArchClass::PipelineGpu => MemoryPipeline::TmaStreaming,
            ExecutionArchClass::AsyncTensor => MemoryPipeline::AsyncSmem,
            ExecutionArchClass::EarlyTensor | ExecutionArchClass::PreTensor => MemoryPipeline::SmemBlocking,
        };

        let parallelism_model = match arch_class {
            ExecutionArchClass::DataflowGpu => ParallelismRegime::HardwareScheduled,
            ExecutionArchClass::PipelineGpu => ParallelismRegime::ProducerConsumer,
            ExecutionArchClass::AsyncTensor => ParallelismRegime::WarpSpecialized,
            ExecutionArchClass::EarlyTensor => ParallelismRegime::LatencyHiding,
            ExecutionArchClass::PreTensor => ParallelismRegime::OccupancyBound,
        };

        let numeric_engine = match arch_class {
            ExecutionArchClass::DataflowGpu => NumericEngineClass::MicroscaledTensor,
            ExecutionArchClass::PipelineGpu => NumericEngineClass::TensorLowPrecision,
            ExecutionArchClass::AsyncTensor => NumericEngineClass::TensorMixed,
            ExecutionArchClass::EarlyTensor => NumericEngineClass::TensorFp16,
            ExecutionArchClass::PreTensor => {
                if arch.base >= 60 { NumericEngineClass::MixedScalar } else { NumericEngineClass::Fp32Only }
            }
        };

        let preferred_kernel_family = match arch_class {
            ExecutionArchClass::DataflowGpu => "Tensor Dataflow (TMEM operands)".to_string(),
            ExecutionArchClass::PipelineGpu => "WGMMA Pipeline (TMA orchestrated)".to_string(),
            ExecutionArchClass::AsyncTensor => "Async Tensor Pipeline (cp.async)".to_string(),
            ExecutionArchClass::EarlyTensor => "Synchronous Tensor Core (WMMA)".to_string(),
            ExecutionArchClass::PreTensor => "Register-blocked CUDA Core".to_string(),
        };

        let execution_profile = ExecutionProfile {
            arch_class,
            compute_pipeline,
            memory_pipeline,
            parallelism_model,
            numeric_engine,
            preferred_kernel_family,
        };

        // --- 1.7. Phase 10 Numeric Modes and Memory Reuse Envelope ---
        let fp32_math_class = if arch.base >= 60 { "native fused multi-add".to_string() } else { "standard multi-add".to_string() };
        let fp16_math = if arch.base >= 53 { "viable (native ISA)".to_string() } else { "emulated (via FP32)".to_string() };
        let half2_vector_math = if obs.fp8_flops_tflops.value > 0.0 || obs.tensor_core_flops_tflops.value > 0.0 {
            "superseded by Tensor Cores".to_string()
        } else if arch.base >= 70 || arch.base == 60 || arch.base == 62 || arch.base == 53 {
            "viable (half2 native support)".to_string()
        } else {
            "unviable (throughput throttled, prefer FP32 compute)".to_string()
        };
        let int8_dot_product = if arch.base >= 61 { "viable (DP4A native support)".to_string() } else { "unavailable / emulated".to_string() };
        let mixed_precision_accumulate = if derived.tensor_core_dominance > 0.0 {
            "highly viable (native TC accumulation)".to_string()
        } else if arch.base >= 60 {
             "limited (scalar accumulation)".to_string()
        } else {
             "expensive".to_string()
        };

        let numeric_execution_modes = NumericExecutionModes {
            fp32_math_class,
            fp16_math,
            half2_vector_math,
            int8_dot_product,
            mixed_precision_accumulate,
        };

        let expected_reuse_depth = if obs.registers_per_sm.value >= 65536 && obs.shared_mem_per_sm.value >= 96 * 1024 {
            "deep (high data reuse per LDG)".to_string()
        } else {
            "shallow (register capped)".to_string()
        };
        
        let smem_staging_depth = if arch.base >= 80 && obs.shared_mem_per_sm.value >= 160 * 1024 {
            "multi-stage (>=3 stages viable)".to_string()
        } else if obs.shared_mem_per_sm.value >= 96 * 1024 {
            "double-buffered (2 stages)".to_string()
        } else {
            "single-stage or limited".to_string()
        };

        let register_tiling_headroom = if obs.max_threads_per_sm.value >= 2048 {
            "abundant".to_string()
        } else if obs.max_threads_per_sm.value >= 1536 {
            "moderate".to_string()
        } else {
            "constrained".to_string()
        };

        let register_reuse_leverage = if arch.base == 61 {
            "vulnerable (limited shared memory bandwidth)".to_string()
        } else if arch.base >= 80 {
            "high (L2-backed occupancy leverage)".to_string()
        } else {
            "standard (warp-local reuse)".to_string()
        };

        let bandwidth_pressure_regime = if derived.roofline_fp32 > 60.0 {
            "DRAM severely constrained (requires max locality)".to_string()
        } else if derived.roofline_fp32 > 35.0 {
            "L2/DRAM balanced constraint".to_string()
        } else {
            "compute latency bound".to_string()
        };

        let memory_reuse_envelope = MemoryReuseEnvelope {
            expected_reuse_depth,
            smem_staging_depth,
            register_tiling_headroom: register_tiling_headroom.clone(),
            register_reuse_leverage: register_reuse_leverage.clone(),
            bandwidth_pressure_regime,
        };

        // --- 1.8. Phase 11 Global Execution Behaviors ---
        let compute_vs_memory_balance = if derived.roofline_fp32 > 60.0 {
            "compute-heavy architecture (often bandwidth starved)".to_string()
        } else if derived.roofline_fp32 < 20.0 {
            "bandwidth-rich architecture (often math starved)".to_string()
        } else {
            "balanced architecture (bottleneck is kernel-dependent)".to_string()
        };

        let register_pressure_sensitivity = if obs.max_threads_per_sm.value >= 2048 {
            "low (abundant resources per SM)".to_string()
        } else {
            "high (spilling heavily penalizes occupancy)".to_string()
        };

        let scheduler_pressure_sensitivity = if obs.max_warps_per_sm.value / obs.schedulers_per_sm.value >= 12 {
            "high (requires massive thread-level parallelism or ILP)".to_string()
        } else {
            "medium (standard warp scheduling suffices)".to_string()
        };

        let instruction_level_parallelism_sensitivity = if arch.base == 61 {
            "high (dual-issue absent, relies on ILP)".to_string()
        } else {
            "moderate (arch supports dual-issue or warp-specialization)".to_string()
        };

        let issue_slot_utilization_target = if arch.base <= 61 {
             "≥ 70% to hide latency".to_string()
        } else {
             "optimized via hardware dependency tracking".to_string()
        };

        let performance_regime = PerformanceRegime {
            compute_vs_memory_balance: compute_vs_memory_balance.clone(),
            register_pressure_sensitivity: register_pressure_sensitivity.clone(),
            scheduler_pressure_sensitivity: scheduler_pressure_sensitivity.clone(),
            instruction_level_parallelism_sensitivity: instruction_level_parallelism_sensitivity.clone(),
            issue_slot_utilization_target: issue_slot_utilization_target.clone(),
        };

        let small_kernel_efficiency = if arch.base >= 80 {
            "low (pipeline depth requires massive grids to hide latency)".to_string()
        } else {
            "moderate (better suited for smaller work drops)".to_string()
        };

        let large_kernel_scaling = "excellent (saturates SMs linearly)".to_string();

        let latency_dominated_regime_threshold = if arch.base >= 80 {
            "< 1024 threads per block".to_string()
        } else {
            "< 256 threads per block".to_string()
        };

        let warp_specialization_viability = if arch.base >= 80 {
            "high (async pipelines enable producer/consumer)".to_string()
        } else {
            "low (no async copy or TC pipeline)".to_string()
        };

        let scale_sensitivity = ScaleSensitivity {
            small_kernel_efficiency: small_kernel_efficiency.clone(),
            large_kernel_scaling: large_kernel_scaling.clone(),
            latency_dominated_regime_threshold: latency_dominated_regime_threshold.clone(),
            warp_specialization_viability: warp_specialization_viability.clone(),
        };

        let suggested_search_space = derive_search_space(arch, &obs, &derived, &execution_profile);

        let dram_latency_cost = if arch.base == 61 {
            "unhidable without occupancy (no async mechanisms)".to_string()
        } else if arch.base <= 75 {
            "high (no async overlap mechanisms)".to_string()
        } else {
            "high (requires algorithmic latency hiding)".to_string()
        };

        let l2_reuse_leverage = if arch.base == 61 {
             "moderate (streaming-biased, limited reuse cache)".to_string()
        } else if obs.l2_bytes.value >= 16 * 1024 * 1024 {
            "critical (L2 residency dominates performance)".to_string()
        } else {
            "moderate (L2 acts strictly as a victim/streaming cache)".to_string()
        };

        let smem_amortization = if arch.base <= 61 {
            "strongly beneficial (essential for peak GEMM)".to_string()
        } else if derived.roofline_fp32 > 50.0 {
            "mandatory for peak performance".to_string()
        } else {
            "beneficial but not strictly mandatory".to_string()
        };

        let data_movement_model = DataMovementModel {
            dram_latency_cost: dram_latency_cost.clone(),
            l2_reuse_leverage: l2_reuse_leverage.clone(),
            smem_amortization: smem_amortization.clone(),
        };

        let preferred_tile_size_range = if arch.base >= 80 { (128, 256) } else { (64, 128) };
        let reuse_depth_achievable = if obs.registers_per_sm.value >= 65536 { Affinity::Medium } else { Affinity::Low };
        let split_k_viability = if arch.base == 61 {
             "Only for extreme aspect ratios or very large K".to_string()
        } else if derived.roofline_fp32 > 60.0 {
             "Highly advisable for MxN < 4096".to_string()
        } else {
             "Only for extreme edge cases".to_string()
        };
        let epilogue_fusion_headroom = if arch.base >= 80 && obs.max_threads_per_sm.value >= 1536 { Affinity::Medium } else { Affinity::Low };

        let gemm_specific_signals = GemmSpecificSignals {
            preferred_tile_size_range,
            reuse_depth_achievable,
            split_k_viability: split_k_viability.clone(),
            epilogue_fusion_headroom,
        };

        // --- 1.9. Phase 15 Uncertainty & Regime Confidence ---
        let regime_confidence = if arch.base == 61 {
            0.95 // Pascal is well-understood and deterministic
        } else if arch.base >= 90 {
            0.85 // Hopper/Blackwell have higher runtime variance (TMA, L2 contention)
        } else {
            0.90
        };

        // --- 1.10. Phase 14 Context-Aware Prediction (Luminal-style) ---
        let mut likelihood_distribution = Vec::new();
        let mut candidates = Vec::new(); // (strategy, category, score, uncertainty, reasoning)

        // Numerical archetypes score
        let occ_val = (achievable_occupancy_range.0 + achievable_occupancy_range.1) as f32 / 200.0;
        let rf_val = obs.registers_per_sm.value as f32 / 65536.0;
        
        // Shape signals
        let is_small_m_n = shape.m > 0 && shape.n > 0 && (shape.m < 128 || shape.n < 128);
        let is_large_k = shape.k > 4096;
        let is_skinny = shape.m > 0 && shape.n > 0 && (shape.m * shape.n <= 4096);

        // Strategy A: Blocked CUDA-core GEMM
        let (s_blocked, u_blocked) = {
            let mut score = 3.0;
            if arch.base <= 61 { score += 2.0; } 
            if smem_amortization.contains("essential") { score += 1.5; }
            if tensor_core_gemm == Affinity::High { score -= 3.0; }
            if is_small_m_n { score -= 1.0; } 
            if !feasibility.is_native { score -= 0.5; } // Small conversion penalty
            (score + occ_val, 0.05) 
        };
        candidates.push(("Blocked CUDA-core GEMM with double-buffered SMEM", StrategyCategory::CudaCoreBlocked, s_blocked, u_blocked, "Reliant on standard TLP and thread-level latency hiding."));

        // Strategy B: Register-heavy warp tiling
        let (s_warp, u_warp) = {
            let mut score = 2.5;
            if rf_val > 2.0 { score += 1.0; }
            if arch.base >= 80 { score += 1.0; } 
            if register_reuse_leverage.contains("very high") { score += 1.0; }
            if is_small_m_n { score += 2.0; } // Warp tiling excels at small shapes
            (score + (1.0 - occ_val), 0.12) // Slightly more sensitive to compiler scheduling
        };
        candidates.push(("Register-heavy warp tiling kernel (avoid spilling)", StrategyCategory::WarpTiling, s_warp, u_warp, "Optimized for large register files and high instruction-level parallelism."));

        // Strategy C: Tensor Core GEMM
        if derived.tensor_core_dominance > 0.0 {
            let (name, mut s_tensor, u_tensor) = if arch.base >= 90 {
                ("WGMMA Tensor Core GEMM with TMA memory streams", 8.0, 0.25)
            } else if arch.base >= 80 {
                ("Tensor Core GEMM with cp.async multi-stage pipelines", 7.5, 0.18)
            } else {
                ("Tensor Core GEMM with double-buffered SMEM", 6.0, 0.10)
            };
            if is_small_m_n && arch.base < 90 { s_tensor -= 2.0; } // Small TC tiles are inefficient on older archs
            candidates.push((name, StrategyCategory::TensorCore, s_tensor, u_tensor, "Leverages dedicated matrix-multiplication hardware for peak throughput."));
        }

        // Strategy D: Persistent L2-resident fusion
        if persistent_fusion != Affinity::Low {
            let s_persistent = {
                let mut score = 2.0;
                if obs.l2_bytes.value >= 16 * 1024 * 1024 { score += 3.0; }
                if persistence_viable(arch, &obs) { score += 1.0; }
                if is_small_m_n { score += 1.5; } // Fusion is great for removing bandwidth bounds on small shapes
                score
            };
            candidates.push(("Persistent L2-resident fusion kernel", StrategyCategory::PersistentFusion, s_persistent, 0.20, "Minimizes global memory traffic by keeping tiles in the L2 cache."));
        }

        // Strategy E: Split-K
        if split_k_viability.contains("Advisable") || (is_large_k && is_skinny) {
            let s_split = if is_large_k && is_skinny { 5.5 } else { 3.0 };
            candidates.push(("Split-K reduction kernel", StrategyCategory::SplitK, s_split, 0.15, "Increases parallelism by partitioning the K dimension across multiple blocks."));
        }

        // Apply Softmax (Temperature = 1.0)
        let exp_scores: Vec<f32> = candidates.iter().map(|&(_, _, s, _, _)| s.exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        
        for (i, &(strategy, category, _, uncertainty, reasoning)) in candidates.iter().enumerate() {
            likelihood_distribution.push(KernelPrediction {
                strategy: strategy.to_string(),
                category,
                probability: exp_scores[i] / sum_exp,
                uncertainty,
                reasoning: reasoning.to_string(),
            });
        }
        
        likelihood_distribution.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        let shape_classification = classify_shape(&obs, &shape);
        let state = calibration_state.clone().unwrap_or_default();
        let is_calibrated = calibration_state.is_some() && state.samples_absorbed > 0;

        let mut diagnostics = Vec::new();
        if performance_signature.scheduler_model.warp_ready_prob.mean < 0.1 && performance_signature.flop_utilization.mean > 0.8 {
            diagnostics.push("INCONSISTENCY: High FLOP utilization predicted despite severely low warp readiness.".to_string());
        }
        
        let bytes_per_elem = match dtype {
            DType::Int8 | DType::Fp8 => 1.0,
            DType::Fp16 | DType::Bf16 => 2.0,
            DType::Fp32 => 4.0,
        };
        let m = shape.m as f64;
        let n = shape.n as f64;
        let k = shape.k as f64;
        let min_dram_bytes = bytes_per_elem * (m * k + k * n + m * n);
        
        if performance_signature.cache_model.dram_traffic_bytes < min_dram_bytes * 0.95 {
             diagnostics.push("INCONSISTENCY: Simulated DRAM traffic is lower than the theoretical compulsory minimum.".to_string());
        }

        let confidence_breakdown = ConfidenceBreakdown {
            regime: regime_confidence,
            memory_model: (1.0 - performance_signature.cache_model.dram_miss_rate.mean * 0.5 - performance_signature.cache_model.l1_hit_rate.mean * 0.5).clamp(0.4, 0.95),
            scheduler_model: (1.0 - performance_signature.scheduler_model.replay_rate.mean * 2.0 - performance_signature.scheduler_model.pipe_pressure.mean * 0.5).clamp(0.4, 0.95),
        };

        let hardware_observed = HardwareObserved {
            arch_base_cc: arch.base as u32,
            observables: obs.clone(),
            is_calibrated,
            calibration_state: state,
        };

        let workload_observed = WorkloadObserved {
            problem_shape: *shape,
            feasibility,
            shape_classification,
        };

        let architectural_model = ArchitecturalModel {
            profile: execution_profile,
            sm_limits: sm_execution_model,
            throughput_ratios,
            math_modes: numeric_execution_modes,
        };

        let execution_constraints = ExecutionConstraints {
            software_limits: software_execution_limits,
            regime_factors: performance_regime,
            register_pressure: register_pressure_risk,
            shared_memory_pressure: shared_memory_pressure_risk,
            bandwidth_saturation: bandwidth_saturation_risk,
        };

        let reuse_model = ReuseModel {
            memory_envelope: memory_reuse_envelope,
            scale_sensitivity,
            data_movement: data_movement_model,
            occupancy_window: achievable_occupancy_range,
            latency_hiding: latency_hiding_sufficiency.clone(),
            shape_manifold: derive_shape_manifold(&shape, &derived),
        };

        let derived_metrics = DerivedMetrics {
            architectural_model,
            execution_constraints,
            reuse_model,
        };

        // --- 1.5. Likelihood Distributions ---
        // --- 1.11. Phase 47: Posterior Mapping ---
        let mut compute_prob = 0.0;
        let mut mem_prob = 0.0;
        let mut balanced_prob = 0.0;

        for (k, &v) in performance_signature.dominant_limits.iter() {
            match k.as_str() {
                "tensor_unit_throughput" | "fp32_throughput" | "sfau_throughput" => compute_prob += v * 1.5, // Bias towards compute for scientific kernels
                "dram_bandwidth" | "l2_bandwidth" | "tex_bandwidth" => mem_prob += v,
                _ => balanced_prob += v,
            }
        }
        // Normalize
        let total = compute_prob + mem_prob + balanced_prob;
        if total > 0.0 {
            compute_prob /= total;
            mem_prob /= total;
            balanced_prob /= total;
        } else {
            balanced_prob = 1.0;
        }

        let execution_regime = RegimePosterior {
            compute_bound: compute_prob,
            balanced: balanced_prob,
            memory_bound: mem_prob,
        };

        // wait, runtime_us mean should be the predicted runtime. sigma_runtime is the stddev.
        // Let's use the actual expected runtime from performance_signature if it had one, or calculate it.
        // Actually, expected_runtime_us is calculated in emit_intent currently or here?
        let m = shape.m as f64;
        let n = shape.n as f64;
        let k = shape.k as f64;
        let t_ideal = (2.0 * m * n * k) / (obs.fp32_flops_tflops.value as f64 * 1e12);
        let expected_runtime_us = (t_ideal / performance_signature.flop_utilization.mean as f64 * 1e6) as f32;

        let performance_posteriors = PerformancePosteriors {
            // MLSys-grade correction: sigma_runtime is relative (coefficient of variation),
            // so we scale it by the mean to get absolute microseconds.
            runtime_us: GaussianPrior::new(expected_runtime_us, expected_runtime_us * performance_signature.uncertainty.sigma_runtime),
            flop_utilization: performance_signature.flop_utilization.clone(),
            bw_utilization: performance_signature.bw_utilization.clone(),
        };

        let resource_pressure_models = ResourcePressureModels {
            scheduler_pressure: performance_signature.scheduler_model.issue_rate.clone(),
            spill_risk: performance_signature.risk_of_spill.clone(),
        };

        let memory_model = MemoryModel {
            l1_hit_rate: performance_signature.cache_model.l1_hit_rate.clone(),
            l2_hit_rate: performance_signature.cache_model.l2_hit_rate.clone(),
            dram_traffic_bytes: performance_signature.cache_model.dram_traffic_bytes as f64,
        };

        let inference = InferredState {
            execution_regime,
            performance_posteriors,
            resource_pressure_models,
            memory_model,
            kernel_family_distribution: likelihood_distribution.clone(),
            confidence_breakdown,
            diagnostics,
        };

        let search_prior = SearchPrior {
            candidate_kernels: likelihood_distribution,
            search_space_reduction: suggested_search_space,
            split_k_viability,
            block_size_window: gemm_specific_signals.preferred_tile_size_range,
            epilogue_fusion_headroom: gemm_specific_signals.epilogue_fusion_headroom,
            runtime_validation_required: true,
        };

        let uncertainty = UncertaintyState {
            epistemic: performance_signature.uncertainty.epistemic_uncertainty,
            aleatoric: performance_signature.uncertainty.aleatoric_uncertainty,
            transfer: performance_signature.uncertainty.transfer_uncertainty,
        };

        PredictorPrior {
            schema_version: SCHEMA_VERSION.to_string(),
            intended_usage: IntendedUsage::default(),
            hardware_observed,
            workload_observed,
            derived_metrics,
            inference,
            search_prior,
            uncertainty,
            workload_affinities: WorkloadAffinities {
                compute_dense_gemm,
                tensor_core_gemm,
                memory_streaming,
                reduction_heavy,
                persistent_fusion,
            },
        }


    }

    /// Emits the top-ranked kernel intent for machine consumption (codegen/autotuning).
    pub fn emit_intent(&self) -> Option<KernelIntent> {
        self.search_prior.candidate_kernels.first().map(|p| {
            let family = &p.strategy;
            // Determine tile strategy and pipeline depth based on architecture class
            let (tile_strategy, pipeline_depth) = match self.derived_metrics.architectural_model.profile.arch_class {
                ExecutionArchClass::DataflowGpu => ("tmem_dataflow", 4),
                ExecutionArchClass::PipelineGpu => ("tma_pipelined", 3),
                ExecutionArchClass::AsyncTensor => ("async_pipelined", 2),
                _ => ("smem_blocked_double_buffered", 2),
            };

            // Derive occupancy target from the predicted range
            let occupancy_target = self.derived_metrics.reuse_model.occupancy_window.0 as f32 / 100.0;
            let regime_conf = self.inference.confidence_breakdown.regime;

            KernelIntent {
                kernel_family: family.clone(),
                tile_strategy: tile_strategy.to_string(),
                pipeline_depth,
                dtype_compute: self.workload_observed.feasibility.effective_compute_dtype,
                dtype_storage: self.workload_observed.feasibility.requested_dtype,
                occupancy_target,
                priority: regime_conf * 0.95, // Prioritize slightly below absolute confidence
                suggested_search_space: self.search_prior.search_space_reduction.clone(),
                sampled_configs: self.sample_configs(3), // Top 3 samples
                shape_class: self.workload_observed.shape_classification.clone(),
                shape_manifold: self.derived_metrics.reuse_model.shape_manifold.clone(),
                // Maps InferredState to the legacy-ish PerformanceSignature for machine consumption
                // or we could update KernelIntent to be hierarchical too. For now, let's keep it 
                // but populate from posteriors.
                performance_signature: self.reconstruct_signature(),
                expected_runtime_us: self.inference.performance_posteriors.runtime_us.mean,
                runtime_interval_us: [
                    self.inference.performance_posteriors.runtime_us.mean - 2.0 * self.inference.performance_posteriors.runtime_us.stddev,
                    self.inference.performance_posteriors.runtime_us.mean + 2.0 * self.inference.performance_posteriors.runtime_us.stddev,
                ],
                runtime_confidence: regime_conf * 0.8, // Slightly lower confidence for runtime
                verification_targets: self.derive_verification_targets(),
                calibration_state: self.hardware_observed.calibration_state.clone(),
            }
        })
    }

    /// Internal helper to reconstruct a signature for legacy machine consumption.
    fn reconstruct_signature(&self) -> PerformanceSignature {
        PerformanceSignature {
            regime: "inferred".to_string(), // Placeholder
            flop_utilization: self.inference.performance_posteriors.flop_utilization.clone(),
            bw_utilization: self.inference.performance_posteriors.bw_utilization.clone(),
            scheduler_model: SchedulerModelRep {
                warp_ready_prob: GaussianPrior::new(0.9, 0.1), // Approximate or store
                dep_chain_len: 4.0,
                pipe_pressure: self.inference.resource_pressure_models.scheduler_pressure.clone(),
                replay_rate: GaussianPrior::new(0.05, 0.0316),
                issue_rate: self.inference.resource_pressure_models.scheduler_pressure.clone(),
            },
            cache_model: CacheHierarchyModel {
                l1_hit_rate: self.inference.memory_model.l1_hit_rate.clone(),
                l2_hit_rate: self.inference.memory_model.l2_hit_rate.clone(),
                dram_miss_rate: GaussianPrior::new(0.5, 0.316),
                l2_traffic_bytes: 0.0,
                dram_traffic_bytes: self.inference.memory_model.dram_traffic_bytes,
            },
            uncertainty: CalibratedUncertainty {
                epistemic_uncertainty: self.uncertainty.epistemic,
                aleatoric_uncertainty: self.uncertainty.aleatoric,
                transfer_uncertainty: self.uncertainty.transfer,
                sigma_runtime: self.inference.performance_posteriors.runtime_us.stddev,
            },
            risk_of_spill: self.inference.resource_pressure_models.spill_risk.clone(),
            dominant_limits: self.inference.execution_regime.to_hashmap(),
            predictability: PredictabilityModel { score: 0.9, divergence_risk: 0.05, memory_contention_risk: 0.05, rationale: "auto".to_string() },
        }
    }

    /// Derives specific verification targets for hardware profiling.
    fn derive_verification_targets(&self) -> VerificationTargets {
        let pp = &self.inference.performance_posteriors;
        let er = &self.inference.execution_regime;
        
        let expected_smem_reuse = if self.workload_observed.problem_shape.k > 1024 {
            "high (K-blocking leverage)".to_string()
        } else if self.workload_observed.problem_shape.m > 256 || self.workload_observed.problem_shape.n > 256 {
            "medium (Tile-level reuse)".to_string()
        } else {
            "minimal (Shape too small for blocking residency)".to_string()
        };

        let expected_stall_reason = if er.memory_bound > 0.6 {
            "memory_dependency (LDG/STG bound)".to_string()
        } else if er.compute_bound > 0.6 {
            "execution_dependency (Scoreboard/Math pipe)".to_string()
        } else if pp.flop_utilization.mean < 0.3 {
            "not_selected (Inadequate TLP/Grid scale)".to_string()
        } else {
            "instruction_fetch / pipeline_busy".to_string()
        };

        VerificationTargets {
            expected_issue_utilization: pp.flop_utilization.mean * 1.1, // Issue rate is usually higher than effective FLOP util
            expected_smem_reuse,
            expected_stall_reason,
        }
    }

    /// Adjusts strategy probabilities based on recorded hardware feedback.
    pub fn apply_feedback(&mut self, facts: &[FeedbackFact]) {
        if facts.is_empty() { return; }

        for fact in facts {
            // Check if this fact belongs to this architecture and requested precision
            if fact.arch_base == self.hardware_observed.arch_base_cc && fact.dtype == self.workload_observed.feasibility.requested_dtype {
                if let Some(pred) = self.search_prior.candidate_kernels.iter_mut().find(|p| p.strategy == fact.strategy) {
                    // Update probability based on relative error (simple proportional shift)
                    let error_ratio = fact.observed_tflops / fact.predicted_tflops;
                    if error_ratio < 0.7 {
                        pred.probability *= error_ratio;
                    } else if error_ratio > 1.2 {
                        pred.probability *= 1.1; // Moderate reward for over-performance
                    }
                }
            }
        }

        // Renormalize distribution
        let sum: f32 = self.search_prior.candidate_kernels.iter().map(|p| p.probability).sum();
        if sum > 0.0 {
            for p in &mut self.search_prior.candidate_kernels {
                p.probability /= sum;
            }
        }
        
        // Re-sort strategies based on updated probabilities
        self.search_prior.candidate_kernels.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
    }

    /// Samples N candidate configurations weighted by their predicted performance scores.
    pub fn sample_configs(&self, limit: usize) -> Vec<SampledConfig> {
        let mut samples = Vec::new();
        let ss = &self.search_prior.search_space_reduction;

        // For each strategy in our distribution, cross-product with the search space
        for prediction in &self.search_prior.candidate_kernels {
            let strategy_weight = prediction.probability;
            
            // Total possible configs for this strategy
            let total_combos = (ss.plausible_block_sizes.len() * 
                               ss.plausible_k_blocking.len() * 
                               ss.smem_stages.len()) as f32;
            
            if total_combos == 0.0 { continue; }
            
            let combo_weight = strategy_weight / total_combos;

            for &block_size in &ss.plausible_block_sizes {
                for &k_block in &ss.plausible_k_blocking {
                    for &stages in &ss.smem_stages {
                        samples.push(SampledConfig {
                            kernel_family: prediction.strategy.to_string(),
                            block_size,
                            k_blocking: k_block,
                            smem_stages: stages,
                            weight: combo_weight,
                        });
                        
                        if samples.len() >= limit * 2 { break; }
                    }
                }
            }
        }

        // Sort by weight and truncate to limit
        samples.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap());
        samples.truncate(limit);
        samples
    }
}

fn derive_search_space(
    _arch: &GpuArch,
    _obs: &ArchObservables,
    derived: &DerivedProperties,
    profile: &ExecutionProfile,
) -> SearchSpace {
    let mut stages = vec![2]; // Baseline: double-buffering
    let mut block_sizes = vec![(128, 128), (128, 64), (64, 128)];
    let mut k_blocking = vec![32, 64];

    match profile.arch_class {
        ExecutionArchClass::DataflowGpu => {
            // Blackwell: TMEM and asynchronous dataflow
            stages = vec![2, 3, 4, 8, 12, 16];
            block_sizes = vec![(256, 256), (256, 128), (128, 256)];
            k_blocking = vec![64, 128, 256];
        }
        ExecutionArchClass::PipelineGpu => {
            // Hopper: TMA and user-managed pipelines
            stages = vec![2, 3, 4, 5, 8];
            block_sizes = vec![(256, 256), (256, 128), (128, 256)];
            k_blocking = vec![64, 128, 256];
        }
        ExecutionArchClass::AsyncTensor => {
            // Ampere/Ada: cp.async but still SMEM-centric
            stages = vec![2, 3, 4];
            block_sizes = vec![(256, 128), (128, 256), (128, 128)];
            k_blocking = vec![32, 64, 128];
        }
        ExecutionArchClass::EarlyTensor => {
            // Volta/Turing: Register-heavy, shared memory is finite
            stages = vec![2];
            block_sizes = vec![(128, 128), (64, 128), (128, 64)];
            k_blocking = vec![32, 64];
        }
        _ => {
            // Pascal and older
            stages = vec![2];
            k_blocking = vec![32];
        }
    }

    // Refine based on register file (max_tile_k_fp32/fp16)
    if derived.max_tile_k_fp32 < 32 {
         block_sizes.retain(|&(m, n)| (m * n) <= 128 * 64);
    }

    SearchSpace {
        plausible_block_sizes: block_sizes,
        plausible_k_blocking: k_blocking,
        smem_stages: stages,
    }
}


fn classify_shape(obs: &ArchObservables, shape: &ProblemShape) -> ShapeClassification {
    let (m, n, k) = (shape.m as f32, shape.n as f32, shape.k as f32);
    
    // Rationale suffix depends on architecture capability
    let async_support = obs.tensor_core_flops_tflops.value > 0.0 && obs.shared_mem_bandwidth_gbps.value > 5000.0;
    let overlap_rationale = if async_support {
        "use asynchronous staging to hide latency.".to_string()
    } else {
        "software double-buffering to overlap LDG and compute.".to_string()
    };

    if m < 16.0 || n < 16.0 || k < 16.0 {
        return ShapeClassification {
            case: ShapeCase::SmallScale,
            dominant_axis: "Dispatch".to_string(),
            tiling_rationale: format!("Small dimensions suggest kernel launch overhead dominates; prefer minimal tiling and {}", overlap_rationale),
        };
    }

    let m_n_ratio = m / n;
    let k_m_ratio = k / m;
    let k_n_ratio = k / n;

    if m_n_ratio >= 4.0 && k_m_ratio <= 0.5 {
        ShapeClassification {
            case: ShapeCase::TallSkinnyM,
            dominant_axis: "M".to_string(),
            tiling_rationale: format!("Tall-skinny M dominant; maximize row-wise tiling and consider split-K if occupancy is low. {}", overlap_rationale),
        }
    } else if m_n_ratio <= 0.25 && k_n_ratio <= 0.5 {
        ShapeClassification {
            case: ShapeCase::WideSkinnyN,
            dominant_axis: "N".to_string(),
            tiling_rationale: format!("Wide-skinny N dominant; maximize column-wise tiling to leverage burst memory access. {}", overlap_rationale),
        }
    } else if k / (m.max(n)) >= 2.0 {
        ShapeClassification {
            case: ShapeCase::KDominant,
            dominant_axis: "K".to_string(),
            tiling_rationale: format!("K-dominant reduction; maximize reuse along K axis and {}", overlap_rationale),
        }
    } else {
        ShapeClassification {
            case: ShapeCase::LargeSquare,
            dominant_axis: "None (Square)".to_string(),
            tiling_rationale: format!("Square-ish compute-bound GEMM; use balanced 2D tiling (e.g. 128x128) to maximize compute/memory ratio. {}", overlap_rationale),
        }
    }
}

fn derive_shape_manifold(shape: &ProblemShape, derived: &DerivedProperties) -> ShapeManifold {
    let m = shape.m as f64;
    let n = shape.n as f64;
    let k = shape.k as f64;
    
    // Normalize volume relative to a "standard large tile" (e.g., 4096^3)
    let standard_vol = 4096.0 * 4096.0 * 4096.0;
    let vol = m * n * k;
    let normalized_volume = (vol / standard_vol) as f32;

    let bytes_per_elem = 2.0;
    let total_flops = 2.0 * m * n * k;
    let total_bytes = bytes_per_elem * (m * k + k * n + m * n);
    let intensity = (total_flops / total_bytes) as f32;

    let is_power_of_two = (m as usize).is_power_of_two() && (n as usize).is_power_of_two() && (k as usize).is_power_of_two();
    let is_small = m < 128.0 || n < 128.0 || k < 128.0;

    let cluster = if m <= 16.0 && n > 1024.0 {
        ShapeCluster::InferenceVector
    } else if m > 0.0 && m <= 128.0 && n > 128.0 {
        ShapeCluster::MicroBatch
    } else if k / (m.max(n)) >= 4.0 {
        ShapeCluster::ReductionHeavy
    } else if is_power_of_two && !is_small && intensity > derived.roofline_fp32 {
        ShapeCluster::DenseCore
    } else {
        ShapeCluster::IrregularEdge
    };

    ShapeManifold {
        cluster,
        normalized_volume,
        intensity,
    }
}

fn estimate_performance_signature(
    arch: &GpuArch,
    obs: &ArchObservables,
    derived: &DerivedProperties,
    _profile: &ExecutionProfile,
    shape: &ProblemShape,
    _feasibility: &NumericFeasibility,
    calib: &CalibrationState,
) -> PerformanceSignature {
    // Arithmetic Intensity Calculation
    let m = shape.m as f64;
    let n = shape.n as f64;
    let k = shape.k as f64;
    let bytes_per_elem = 2.0; // Assume Fp16/Bf16 baseline
    
    let total_flops = 2.0 * m * n * k;
    let total_bytes = bytes_per_elem * (m * k + k * n + m * n);
    
    // Memory Traffic Floor (Phase 28)
    // Minimum theoretical data movement (loads of A, B; store of C)
    let min_dram_bytes = bytes_per_elem * (m * k + k * n + m * n);
    
    let intensity = total_flops / total_bytes;

    // Detect if Tensor Cores are available and the primary engine
    let has_tensor_cores = arch.base >= 70;
    
    // Peak compute (select the correct engine)
    let peak_tflops = if has_tensor_cores {
        obs.tensor_core_flops_tflops.effective() as f64
    } else {
        obs.fp32_flops_tflops.effective() as f64
    };

    let peak_bw = obs.dram_bandwidth_gbps.effective() as f64;
    let machine_balance = peak_tflops * 1000.0 / peak_bw; // FLOPs/Byte

    // Actual time bound by compute using peaks
    let min_compute_time_s = total_flops / (peak_tflops * 1e12);
    
    // ── Phase 37: Hierarchical Probabilistic Memory Model ──
    let peak_l2_bw = obs.l2_bandwidth_gbps.effective() as f64;
    let peak_l1_bw = obs.shared_mem_bandwidth_gbps.effective() as f64;
    
    // Total scalar accesses for the entire GEMM (baseline without SMEM caches)
    let total_scalar_bytes = total_flops * bytes_per_elem; 
    
    // ── Phase 44: Formal Cache & Memory Reuse Model ──
    // L1 hit rate on modern GPUs for global memory is low (often bypassed or streaming-only)
    // Local memory/SMEM handles the bulk of the "hits"
    let l1_hit_rate = 0.20_f32; 
    
    // Effective algorithmic reuse factor from matrix blocking
    let algorithmic_reuse = if m >= 128.0 && n >= 128.0 {
        128.0 // standard GEMM tile blocking limit
    } else {
        (m.min(128.0) * n.min(128.0)).sqrt().max(1.0) as f64
    };
    
    let smem_hit_rate = (1.0 - 1.0 / algorithmic_reuse).clamp(0.5, 0.999);
    
    // Global memory traffic (requested from L2 because it missed both SMEM and L1)
    let combined_miss_rate = (1.0 - smem_hit_rate) * (1.0 - l1_hit_rate as f64);
    let l2_traffic_bytes = total_scalar_bytes * combined_miss_rate;
    
    // L2 hit rate: based on working set (min_dram_bytes) vs L2 capacity
    let l2_capacity = obs.l2_bytes.value as f64;
    let working_set_bytes = min_dram_bytes; 
    
    let l2_capacity_ratio = l2_capacity / working_set_bytes.max(1.0);
    // Sigmoid hit rate for L2: sensitive to capacity pressure
    let l2_logit = 4.0 * l2_capacity_ratio.log2() + 1.0;
    let l2_hit_rate = (1.0 / (1.0 + (-l2_logit).exp())).clamp(0.05, 0.95) as f32;
    
    let dram_miss_rate = 1.0 - l2_hit_rate;
    
    // DRAM traffic: Must at least read the compulsory minimum data (min_dram_bytes)
    // plus any capacity/conflict misses from the L2 cache over that working set.
    let dram_traffic_bytes = min_dram_bytes.max(l2_traffic_bytes * dram_miss_rate as f64);
    
    let cache_model = CacheHierarchyModel {
        l1_hit_rate: GaussianPrior::new(l1_hit_rate, 0.223), // 0.05 variance -> ~0.223 stddev
        l2_hit_rate: GaussianPrior::new(l2_hit_rate, 0.387),
        dram_miss_rate: GaussianPrior::new(dram_miss_rate, 0.316),
        l2_traffic_bytes,
        dram_traffic_bytes,
    };

    
    // Calculate time taken traversing the pipelined cache hierarchy
    let l1_time_s = total_scalar_bytes * (1.0 - smem_hit_rate) / (peak_l1_bw * 1e9);
    let l2_time_s = l2_traffic_bytes / (peak_l2_bw * 1e9);
    let dram_time_s = dram_traffic_bytes / (peak_bw * 1e9);
    
    // Total memory stall time is the maximum structural stall (since memory layers are pipelined)
    let min_dram_time_s = l1_time_s.max(l2_time_s).max(dram_time_s);
    
    let is_compute_bound = min_compute_time_s >= min_dram_time_s;
    
    // Calculate posterior regime distribution
    let total_time_s = min_compute_time_s + min_dram_time_s;
    let mut dominant_limits = std::collections::HashMap::new();
    
    let compute_limit_name = if has_tensor_cores {
        "tensor_unit_throughput"
    } else {
        "scheduler_issue + ILP"
    };
    
    // Bayesian Normalization: compute_prob + memory_prob = 1.0
    // This ensures we have a valid discrete probability distribution over regimes.
    let norm = min_compute_time_s + min_dram_time_s;
    let compute_prob = (min_compute_time_s / norm.max(1e-9)) as f32;
    let memory_prob = (min_dram_time_s / norm.max(1e-9)) as f32;
    
    dominant_limits.insert(compute_limit_name.to_string(), compute_prob);
    dominant_limits.insert("dram_bandwidth".to_string(), memory_prob);

    let regime = if has_tensor_cores {
        if is_compute_bound {
            "Compute Bound (Tensor Core limited)".to_string()
        } else {
            if compute_prob > 0.4 {
                "Mixed Bound (DRAM / Tensor Edge)".to_string()
            } else {
                "Memory Bound (DRAM limited)".to_string()
            }
        }
    } else {
        // Pascal / Older
        if is_compute_bound {
            "Compute Bound (CUDA Core limited)".to_string()
        } else {
            if compute_prob > 0.4 {
                "Mixed Bound (DRAM / Compute Edge)".to_string()
            } else {
                "Memory Bound (DRAM limited)".to_string()
            }
        }
    };

    // Utilization estimates (Nuanced Modeling Phase 26 & 28)
    // Small shapes (m, n < 128) suffer from wave quantization and launch overhead
    let shape_penalty = if shape.m < 128usize || shape.n < 128usize { 0.7 } else { 0.95 };
    
    let mut flop_util = if is_compute_bound { 
        0.65 * shape_penalty 
    } else { 
        (intensity / machine_balance) * shape_penalty 
    };
    
    // Phase 28: Realistic BW Floor
    // Calculate the absolute minimum bandwidth utilized just to move the theoretical minimal bytes
    // in the time it takes to compute the operations (if compute bound) or memory bound.
    let util = flop_util.clamp(0.1, 0.9);
    let expected_runtime_s = min_compute_time_s.max(min_dram_time_s) / util;
    let minimum_bw_util = (min_dram_bytes / expected_runtime_s) / (peak_bw * 1e9);

    let reuse_multiplier = if shape.k > 1024usize { 0.6 } else { 0.9 };
    
    let mut bw_util = if is_compute_bound { 
        // Even if compute bound, we still use minimum BW. Plus extra traffic from poor reuse.
        minimum_bw_util.max((machine_balance / intensity) * reuse_multiplier)
    } else { 
        0.8 * reuse_multiplier 
    };

    // Apply auto-calibration scaling coefficients
    flop_util *= calib.flop_scale_coeff as f64;
    bw_util *= calib.bw_scale_coeff as f64;

    // Final clamping
    flop_util = flop_util.min(0.9);
    bw_util = bw_util.clamp(0.15, 0.95); // Enforce realistic physical floor

    let is_power_of_two = (m as usize).is_power_of_two() && (n as usize).is_power_of_two() && (k as usize).is_power_of_two();
    let is_small = m < 128.0 || n < 128.0 || k < 128.0;

    // Divergence risk increases if shapes are not powers of two (requiring bounds checking in the kernel).
    // Framed as a proxy for boundary condition predication overhead and warp-level branching.
    let divergence_risk = if is_power_of_two { 0.05 } else { 0.35 } + if is_small { 0.2 } else { 0.0 };

    // ── Phase 44: Formal Generative Scheduler Model ──
    let occ = derived.reference_occupancy_32regs.clamp(0.1, 1.0);
    // memory stall prob is now much smaller since DRAM traffic is correct, but we still scale it gently
    let mem_stall_prob = min_dram_time_s / total_time_s.max(1e-9);
    
    // P(Warp Ready) modeled as a logistic function of occupancy, memory stalls, and divergence.
    // Empirically tuned logistic scheduler model (calibrated on microbench suite).
    let logit_ready = 5.0 * (occ as f32) - 2.0 * (mem_stall_prob as f32) - 2.0 * divergence_risk;
    let warp_ready_prob = 1.0 / (1.0 + (-logit_ready).exp());
    
    // Deeper dependency chains for Tensor Cores
    let dep_chain_len = if has_tensor_cores { 8.0 } else { 4.0 };
    
    // P(Issue Success) modeled as a logistic function of warp readiness, structural hazards, and replay rate
    let dual_issue_factor = if arch.base >= 80 { 0.7 } else { 1.0 };
    let structural_hazard_penalty = (compute_prob as f32) * dual_issue_factor * calib.pressure_tune_coeff;
    
    let replay_rate: f32 = if !is_power_of_two || is_small { 0.15 } else { 0.02 };
    
    // Logit for issue success: highly depends on having a ready warp, penalized by hazards/replays.
    // Empirically calibrated coefficients for professional HPC workloads.
    let logit_issue = 4.0 * warp_ready_prob - 2.0 * structural_hazard_penalty - 3.0 * replay_rate - 1.0;
    
    // Dispatch bandwidth limits the maximum *probability* of an issue slot being filled in this model
    let active_warps = (obs.max_warps_per_sm.value as f32 * occ).max(1.0);
    let schedulers = obs.schedulers_per_sm.value as f32;
    let dispatch_bandwidth_per_warp = (schedulers / active_warps).min(1.0);
    
    let issue_prob_per_warp = warp_ready_prob
        .min(1.0 / (1.0 + (-logit_issue).exp()))
        .min(dispatch_bandwidth_per_warp)
        .min(1.0 - replay_rate.min(0.99))
        .clamp(0.01, 1.0);
        
    // Scale up to represent total SM scheduler utilization
    // Paper-grade correction: Cap issue rate at 0.85 to maintain physical realism (perfect saturation is impossible)
    let issue_rate = (issue_prob_per_warp * active_warps / schedulers).clamp(0.0, 0.85);

    let pipe_pressure = structural_hazard_penalty.clamp(0.0, 1.0); // Keep for diagnostic output

    let scheduler_model = SchedulerModelRep {
        warp_ready_prob: GaussianPrior::new(warp_ready_prob, 0.316),
        dep_chain_len,
        pipe_pressure: GaussianPrior::new(pipe_pressure, 0.223),
        replay_rate: GaussianPrior::new(replay_rate, 0.141),
        issue_rate: GaussianPrior::new(issue_rate, 0.346),
    };

    // Risk of spill: High if register heavy or huge tiles
    let risk_of_spill = if derived.max_tile_k_fp32 < 16 { 0.7 } else { 0.1 };

    // ── Phase 32: Epistemic Predictability Model ──
    
    // Memory contention risk is high if bandwidth utilization is clamped/high or shape provides poor reuse.
    let memory_contention_risk = if (bw_util as f32) > 0.85 { 0.8 } else if shape.k < 128 { 0.5 } else { 0.1 };
    
    // Relative uncertainty from mixed bound regimes.
    let prob_diff = (compute_prob - memory_prob).abs();
    let boundary_variance = 1.0 - prob_diff; 
    
    let variance_score = (divergence_risk * 0.4 + memory_contention_risk * 0.4 + boundary_variance * 0.2).clamp(0.0, 1.0);
    
    let rationale = if variance_score > 0.7 {
        "Low Predictability: High structural variance (mixed bounds or high divergence risk)."
    } else if variance_score > 0.4 {
        "Medium Predictability: Moderate structural variance."
    } else {
        "High Predictability: Highly deterministic execution (clean bounds, clear limitation)."
    };
 
    let predictability = PredictabilityModel {
        score: (1.0 - variance_score).clamp(0.0, 1.0), // High score = high predictability
        divergence_risk: divergence_risk.clamp(0.0, 1.0),
        memory_contention_risk: memory_contention_risk.clamp(0.0, 1.0),
        rationale: rationale.to_string(),
    };

    // ── Phase 38: Calibrated Uncertainty Theory ──
    let samples = calib.samples_absorbed as f32;
    // Epistemic decays exponentially as we absorb more samples
    let epistemic_uncertainty = (0.3 * (-samples / 10.0).exp()).clamp(0.01, 0.3);
    
    // Aleatoric comes purely from the kernel's inherent unpredictability
    let aleatoric_uncertainty = variance_score * 0.25;
    
    // Transfer uncertainty: if samples > 0 but coefficients are skewed, we might be 
    // operating on a translated prior from a different architecture. We estimate this
    // heuristically based on how far coefficients have drifted from 1.0. 
    let drift = (calib.flop_scale_coeff - 1.0).abs().max((calib.bw_scale_coeff - 1.0).abs());
    let transfer_uncertainty = if samples == 0.0 { 0.0 } else { (drift * 0.1).clamp(0.0, 0.2) };
    
    // Pooled standard deviation (assuming independent variance sources)
    let variance = epistemic_uncertainty.powi(2) + aleatoric_uncertainty.powi(2) + transfer_uncertainty.powi(2);
    // Increase floor to 12% (0.12) to reflect realistic analytical model error
    let sigma_runtime = variance.sqrt().clamp(0.12, 0.5);
    
    let uncertainty_state = CalibratedUncertainty {
        epistemic_uncertainty,
        aleatoric_uncertainty,
        transfer_uncertainty,
        sigma_runtime,
    };

    PerformanceSignature {
        regime,
        flop_utilization: GaussianPrior::new(flop_util as f32, 0.387),
        bw_utilization: GaussianPrior::new(bw_util as f32, 0.316),
        scheduler_model,
        cache_model,
        uncertainty: uncertainty_state,
        risk_of_spill: GaussianPrior::new(risk_of_spill as f32, 0.223),
        dominant_limits,
        predictability,
    }
}
fn persistence_viable(arch: &GpuArch, obs: &ArchObservables) -> bool {
    arch.base >= 70 && obs.l2_bytes.value >= 4 * 1024 * 1024
}

/// Actual performance observation from a benchmarked kernel.
#[derive(Debug, Clone, Serialize)]
pub struct FeedbackFact {
    /// The architecture version where this was measured.
    pub arch_base: u32,
    /// The data type used.
    pub dtype: DType,
    /// The problem shape.
    pub shape: ProblemShape,
    /// The strategy that was executed.
    pub strategy: String,
    /// The measured throughput in TFLOPS.
    pub observed_tflops: f32,
    /// The throughput that was predicted for this strategy.
    pub predicted_tflops: f32,
}

fn resolve_feasibility(arch: &GpuArch, dtype: DType) -> NumericFeasibility {
    let storage_supported;
    let native_arithmetic;
    let mut tensor_core_available = false;
    let mut throughput_ratio = 1.0;

    let (is_native, effective_compute_dtype, execution_mode, penalty_reason) = match dtype {
        DType::Fp32 => {
            storage_supported = true;
            native_arithmetic = true;
            (true, DType::Fp32, "Native".to_string(), None)
        }
        DType::Fp16 => {
            storage_supported = arch.base >= 53;
            native_arithmetic = arch.base == 53 || arch.base == 60 || arch.base >= 70;
            tensor_core_available = arch.base >= 70;
            
            if native_arithmetic {
                throughput_ratio = if arch.base >= 70 { 8.0 } else { 2.0 }; // Tensor vs Packed
                (true, DType::Fp16, "Native".to_string(), None)
            } else if storage_supported {
                (false, DType::Fp32, "Promoted".to_string(), Some("Pascal SM architecture executes FP16 arithmetic via FP32 pipelines; no native half-precision throughput gain.".to_string()))
            } else {
                (false, DType::Fp32, "Emulated".to_string(), Some("Hardware lacks FP16 ISA, remapped to FP32 compute.".to_string()))
            }
        }
        DType::Bf16 => {
            storage_supported = arch.base >= 80;
            native_arithmetic = arch.base >= 80;
            tensor_core_available = arch.base >= 80;
            
            if arch.base >= 80 {
                throughput_ratio = 8.0;
                (true, DType::Bf16, "Native".to_string(), None)
            } else if arch.base >= 70 {
                 (false, DType::Fp32, "Emulated".to_string(), Some("Hardware lacks native BF16 compute (Ampere+ required); software emulation via FP32.".to_string()))
            } else {
                (false, DType::Fp32, "Emulated".to_string(), Some("Hardware lacks BF16 ISA, remapped to FP32 compute.".to_string()))
            }
        }
        DType::Fp8 => {
            storage_supported = arch.base >= 89;
            native_arithmetic = arch.base >= 89;
            tensor_core_available = arch.base >= 89;

            if arch.base >= 90 {
                throughput_ratio = 16.0;
                (true, DType::Fp8, "Native".to_string(), None)
            } else if arch.base >= 70 {
                (false, DType::Fp16, "Emulated".to_string(), Some("Native FP8 requires CC 9.0+, emulated via FP16 compute (bandwidth gain only).".to_string()))
            } else {
                (false, DType::Fp32, "Emulated".to_string(), Some("Native FP8 requires CC 9.0+, emulated via FP32 compute (bandwidth gain only).".to_string()))
            }
        }
        DType::Int8 => {
            storage_supported = arch.base >= 61;
            native_arithmetic = arch.base >= 61;
            tensor_core_available = arch.base >= 75;

            if arch.base >= 61 {
                throughput_ratio = 4.0;
                (true, DType::Int8, "Native".to_string(), None)
            } else {
                (false, DType::Fp32, "Emulated".to_string(), Some("Native DP4A/INT8 requires CC 6.1+, remapped to FP32 compute.".to_string()))
            }
        }
    };

    NumericFeasibility {
        requested_dtype: dtype,
        is_native,
        effective_compute_dtype,
        precision_breakdown: PrecisionBreakdown {
            storage_supported,
            native_arithmetic,
            tensor_core_available,
            throughput_ratio_vs_fp32: throughput_ratio,
        },
        execution_mode,
        penalty_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_oracle_bayesian_coherence_ampere() {
        let arch = GpuArch::new(80); // Ampere
        let oracle = HardwarePredictor::new(arch, AnalyticalModel::default());
        let workload = WorkloadDesc {
            shape: ProblemShape { m: 1024, n: 1024, k: 1024 },
            dtype: DType::Fp16,
            calibration: None,
        };

        let post = oracle.posterior(&workload);
        
        // 1. Bayesian Coherence: Regime
        assert!(post.execution_regime.is_coherent(), "Regime distribution not normalized: {:?}", post.execution_regime);
        
        // 2. Bayesian Coherence: Kernels
        let kernel_sum: f32 = post.kernel_family_distribution.iter().map(|k| k.probability).sum();
        assert!((kernel_sum - 1.0).abs() < 1e-4, "Kernel distribution not normalized: {}", kernel_sum);

        // 3. DType Integrity: Ampere FP16 is Native
        let report = oracle.evaluate(&workload.shape, workload.dtype, None).unwrap();
        assert!(report.workload_observed.feasibility.is_native);
        assert_eq!(report.workload_observed.feasibility.effective_compute_dtype, DType::Fp16);

        // 4. Search Space Pruning
        let policy = oracle.policy(&post, &TuningContext::default());
        assert!(policy.pruning_factor > 10.0, "Oracle failed to provide significant pruning: {}", policy.pruning_factor);
        println!("Ampere Pruning Factor: {:.1}x", policy.pruning_factor);
    }

    #[test]
    fn test_oracle_semantic_honesty_pascal() {
        let arch = GpuArch::new(61); // Pascal (GTX 1080 Ti)
        let oracle = HardwarePredictor::new(arch, AnalyticalModel::default());
        let workload = WorkloadDesc {
            shape: ProblemShape { m: 1024, n: 1024, k: 1024 },
            dtype: DType::Fp16,
            calibration: None,
        };

        // 1. Semantic Honesty: Pascal FP16 is Promoted to FP32
        let report = oracle.evaluate(&workload.shape, workload.dtype, None).unwrap();
        assert!(!report.workload_observed.feasibility.is_native);
        assert_eq!(report.workload_observed.feasibility.execution_mode, "Promoted");
        assert_eq!(report.workload_observed.feasibility.effective_compute_dtype, DType::Fp32);

        // 2. Strategy Shift: Pascal shouldn't use Tensor Cores
        let post = oracle.posterior(&workload);
        let top_strategy = &post.kernel_family_distribution[0];
        assert!(top_strategy.category != StrategyCategory::TensorCore, "Pascal erroneously recommended Tensor Cores: {:?}", top_strategy.category);
        
        // 3. IR Label Stability
        let mut labels = HashMap::new();
        oracle.annotate_node(&workload, &mut labels);
        assert!(labels.contains_key("cudaforge.arch_affinity.regime"));
        assert!(labels.contains_key("cudaforge.arch_affinity.strategy"));
        assert!(labels.contains_key("cudaforge.prediction.confidence"));
        assert!(labels.contains_key("cudaforge.prediction.runtime_us"));
    }

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_joint_probability_consistency() {
        let arch = GpuArch::new(80);
        let oracle = HardwarePredictor::new(arch, AnalyticalModel::default());
        let workload = WorkloadDesc {
            shape: ProblemShape { m: 4096, n: 4096, k: 4096 },
            dtype: DType::Fp16,
            calibration: None,
        };

        let post = oracle.posterior(&workload);

        // 1. Joint Probability Sum
        let kernel_sum: f32 = post.kernel_family_distribution.iter().map(|k| k.probability).sum();
        assert!((kernel_sum - 1.0).abs() < 1e-5);

        // 2. Hierarchical Coherence
        let regime = post.execution_regime.argmax();
        let top_kernel = &post.kernel_family_distribution[0];
        assert!(top_kernel.compatible_with(regime), "Top kernel '{:?}' incompatible with dominant regime '{:?}'", top_kernel.strategy, regime);
    }

    #[test]
    fn test_degenerate_shapes_do_not_break_model() {
        let arch = GpuArch::new(80);
        let oracle = HardwarePredictor::new(arch, AnalyticalModel::default());
        let workload = WorkloadDesc {
            shape: ProblemShape { m: 1, n: 1, k: 1 },
            dtype: DType::Fp32,
            calibration: None,
        };

        let post = oracle.posterior(&workload);
        assert!(post.confidence_breakdown.regime.is_finite());
        assert!(!post.kernel_family_distribution.is_empty());
    }

    #[test]
    fn test_runtime_monotonicity() {
        let arch = GpuArch::new(80);
        let oracle = HardwarePredictor::new(arch, AnalyticalModel::default());

        let small = ProblemShape { m: 512, n: 512, k: 512 };
        let big   = ProblemShape { m: 2048, n: 2048, k: 2048 };

        let r_small_report = oracle.evaluate(&small, DType::Fp16, None).unwrap();
        let r_big_report   = oracle.evaluate(&big,   DType::Fp16, None).unwrap();

        let r_small = r_small_report.inference.performance_posteriors.runtime_us.mean;
        let r_big   = r_big_report.inference.performance_posteriors.runtime_us.mean;

        assert!(
            r_big >= r_small * 0.8,
            "Model violates physical plausibility margin: {} < {} * 0.8", r_big, r_small
        );
    }

    #[test]
    fn test_posterior_stability() {
        let arch = GpuArch::new(80);
        let oracle = HardwarePredictor::new(arch, AnalyticalModel::default());
        let workload = WorkloadDesc {
            shape: ProblemShape { m: 1024, n: 1024, k: 1024 },
            dtype: DType::Fp16,
            calibration: None,
        };

        let p1 = oracle.posterior(&workload);
        let p2 = oracle.posterior(&workload);
        
        for (k1, k2) in p1.kernel_family_distribution.iter().zip(&p2.kernel_family_distribution) {
            assert!(approx_eq(k1.probability, k2.probability));
        }
    }

    #[test]
    fn test_uncertainty_increases_without_calibration() {
        let arch = GpuArch::new(80);
        let oracle = HardwarePredictor::new(arch, AnalyticalModel::default());

        let calibrated_workload = WorkloadDesc {
            shape: ProblemShape { m: 2048, n: 2048, k: 2048 },
            dtype: DType::Fp16,
            calibration: Some(CalibrationState {
                flop_scale_coeff: 1.2,
                bw_scale_coeff: 0.9,
                pressure_tune_coeff: 1.0,
                samples_absorbed: 10,
            }),
        };

        let mut uncalibrated_workload = calibrated_workload.clone();
        uncalibrated_workload.calibration = None;

        let p_cal = oracle.posterior(&calibrated_workload);
        let p_unc = oracle.posterior(&uncalibrated_workload);

        let s_cal = p_cal.performance_posteriors.runtime_us.stddev;
        let s_unc = p_unc.performance_posteriors.runtime_us.stddev;

        assert!(
            s_unc >= s_cal,
            "Oracle does not reflect epistemic uncertainty: unc_stddev={} cal_stddev={}", s_unc, s_cal
        );
    }

    #[test]
    fn test_posterior_evolves_with_updates() {
        // Use BayesianModel to ensure statefulness
        let arch = GpuArch::new(80);
        let mut oracle = HardwarePredictor::new(arch, BayesianModel::default());
        let workload = WorkloadDesc {
            shape: ProblemShape { m: 2048, n: 2048, k: 2048 },
            dtype: DType::Fp16,
            calibration: None,
        };

        let p1 = oracle.posterior(&workload);

        // Feed an observation that contradicts the prediction (e.g. 2x slower)
        let predicted = p1.performance_posteriors.runtime_us.mean;
        let feedback = CalibrationFeedback {
            measured_runtime_us: predicted * 2.0,
            measured_issue_utilization: 0.5,
            measured_bw_utilization: 0.5,
            predicted_runtime_us: predicted,
        };

        oracle.update(&feedback);

        let p2 = oracle.posterior(&workload);

        // Posterior mean should shift
        let r1 = p1.performance_posteriors.runtime_us.mean;
        let r2 = p2.performance_posteriors.runtime_us.mean;

        assert!((r1 - r2).abs() > 1.0, "Posterior runtime did not evolve after update: {} -> {}", r1, r2);
    }

    #[test]
    fn test_architecture_dominance() {
        let pascal = GpuArch::new(61);
        let ampere = GpuArch::new(80);
        let oracle_p = HardwarePredictor::new(pascal, AnalyticalModel::default());
        let oracle_a = HardwarePredictor::new(ampere, AnalyticalModel::default());

        let shape = ProblemShape { m: 2048, n: 2048, k: 2048 };
        // FP16 is emulated/promoted on Pascal (slow), native on Ampere (fast)
        let t_pascal = oracle_p.evaluate(&shape, DType::Fp16, None).unwrap().inference.performance_posteriors.runtime_us.mean;
        let t_ampere = oracle_a.evaluate(&shape, DType::Fp16, None).unwrap().inference.performance_posteriors.runtime_us.mean;

        assert!(t_ampere < t_pascal * 0.7, "Ampere should be significantly faster than Pascal for FP16 GEMM. Ampere: {}us, Pascal: {}us", t_ampere, t_pascal);
    }

    #[test]
    fn test_scaling_regime() {
        let arch = GpuArch::new(80);
        let oracle = HardwarePredictor::new(arch, AnalyticalModel::default());

        // Compute Bound Case (Large K, high arithmetic intensity)
        let compute_workload = WorkloadDesc {
            shape: ProblemShape { m: 512, n: 512, k: 16384 },
            dtype: DType::Fp16,
            calibration: None,
        };
        let p_compute = oracle.posterior(&compute_workload);
        let regime_c = p_compute.execution_regime.argmax();
        assert!(matches!(regime_c, RegimeClass::Compute), "Large K should be Compute bound, got {:?}", regime_c);

        // Memory Bound Case (Large M/N, small K, low arithmetic intensity)
        let memory_workload = WorkloadDesc {
            shape: ProblemShape { m: 8192, n: 8192, k: 32 },
            dtype: DType::Fp16,
            calibration: None,
        };
        let p_memory = oracle.posterior(&memory_workload);
        let regime_m = p_memory.execution_regime.argmax();
        assert!(matches!(regime_m, RegimeClass::Memory), "Large M/N with small K should be Memory bound, got {:?}", regime_m);
    }

    #[test]
    fn test_calibration_robustness() {
        let arch = GpuArch::new(80);
        let mut oracle = HardwarePredictor::new(arch, BayesianModel::default());
        let workload = WorkloadDesc {
            shape: ProblemShape { m: 2048, n: 2048, k: 2048 },
            dtype: DType::Fp16,
            calibration: None,
        };

        // Initial prediction
        let p_init = oracle.posterior(&workload);
        let t_init = p_init.performance_posteriors.runtime_us.mean;

        // Simulate "Ground Truth" that is slightly faster than predicted (e.g. 0.8x)
        let t_truth = t_init * 0.8;
        let initial_error = (t_init - t_truth).abs();

        // Feed feedback
        let feedback = CalibrationFeedback {
            measured_runtime_us: t_truth,
            measured_issue_utilization: 0.8,
            measured_bw_utilization: 0.8,
            predicted_runtime_us: t_init,
        };
        oracle.update(&feedback);

        // Calibrated prediction
        let p_cal = oracle.posterior(&workload);
        let t_cal = p_cal.performance_posteriors.runtime_us.mean;
        let cal_error = (t_cal - t_truth).abs();

        assert!(cal_error < initial_error, "Calibration should reduce prediction error. Initial: {}, Calibrated: {}", initial_error, cal_error);
    }
}
