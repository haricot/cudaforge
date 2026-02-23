//! Implementation Predictor Module
//! 
//! Transforms static hardware metrics into an active cost model,
//! estimating kernel class affinities, resource pressures,
//! and likely optimal implementation strategies.

use crate::arch_metrics::{ArchObservables, DerivedProperties};
use crate::compute_cap::GpuArch;
use serde::{Serialize, Deserialize};

/// Affinity/likelihood of a hardware architecture favoring a specific kernel class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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

/// Estimates of resource pressure limits for the architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PressureRisk {
    /// Severe bottleneck likely.
    High,
    /// Requires careful tuning or tiling.
    Medium,
    /// Plenty of headroom available.
    Low,
}

impl std::fmt::Display for PressureRisk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PressureRisk::High => write!(f, "\x1b[31mhigh\x1b[0m"),
            PressureRisk::Medium => write!(f, "\x1b[33mmedium\x1b[0m"),
            PressureRisk::Low => write!(f, "\x1b[32mlow\x1b[0m"),
        }
    }
}

/// SM execution concurrency constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmExecutionModel {
    /// Maximum concurrent warps per SM.
    pub max_warps_per_sm: u32,
    /// Maximum resident thread blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Number of warp schedulers per SM.
    pub schedulers_per_sm: u32,
}

/// Critical equilibrium ratios that dictate algorithmic choices.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GemmSpecificSignals {
    /// The optimal range of `K` tile sizes to target (min, max).
    pub preferred_tile_size_range: (u32, u32),
    /// Likelihood of achieving sufficient register reuse depth before spilling.
    pub reuse_depth_achievable: Affinity,
    /// Is split-K recommended dynamically? (e.g., "Yes, for large MxN", "No")
    pub split_k_viability: &'static str,
    /// Is epilogue fusion (Bias+ReLU/GELU) viable without crashing register pressure?
    pub epilogue_fusion_headroom: Affinity,
}

/// Viable mathematical execution modes supported by the architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericExecutionModes {
    /// The primary precision for floating point math (e.g., "primary highly-optimized path").
    pub fp32_math_class: &'static str,
    /// Viability of using 16-bit floating point math.
    pub fp16_math: &'static str,
    /// Viability of using packed 16-bit vector math (e.g. `half2`).
    pub half2_vector_math: &'static str,
    /// Viability of hardware-accelerated INT8 dot-product (e.g., `dp4a`).
    pub int8_dot_product: &'static str,
    /// Viability of mixed-precision accumulation.
    pub mixed_precision_accumulate: &'static str,
}

// ── Phase 12 Multi-Arch Enums ──

/// Classification of the compute pipeline's maturity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputePipeline {
    /// Standard CUDA core based scalar math.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPipeline {
    /// No specialized memory pipeline.
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

/// Parallelism regime the architecture prefers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParallelismRegime {
    /// Reliant on high occupancy to hide latency.
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

/// Numeric engine specialization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericEngineClass {
    /// Basic single-precision only.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionArchClass {
    /// Pre-Tensor Core architectures (Pascal and older).
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
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub preferred_kernel_family: &'static str,
}

/// Estimates of achievable reuse and staging envelopes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReuseEnvelope {
    /// Expected reuse depth achievable given register file constraints.
    pub expected_reuse_depth: &'static str,
    /// Maximum feasible staging depth through Shared Memory.
    pub smem_staging_depth: &'static str,
    /// Headroom for scaling register tiling.
    pub register_tiling_headroom: &'static str,
    /// Impact/leverage of register reuse on overall throughput.
    pub register_reuse_leverage: &'static str,
    /// Which bandwidth threshold is the primary architectural bottleneck?
    pub bandwidth_pressure_regime: &'static str,
}

/// High-level behavioral regime the GPU prefers.
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceRegime {
    /// Overall balance of compute vs memory.
    pub compute_vs_memory_balance: &'static str,
    /// Sensitivity to register pressure during kernel execution.
    pub register_pressure_sensitivity: &'static str,
    /// Sensitivity to the number of active warps fed to the scheduler.
    pub scheduler_pressure_sensitivity: &'static str,
    /// Sensitivity to instruction-level parallelism (ILP).
    pub instruction_level_parallelism_sensitivity: &'static str,
    /// Target issue slot utilization for latency hiding.
    pub issue_slot_utilization_target: &'static str,
}

/// Sensitivity to the scale and size of the dispatched kernel grid.
#[derive(Debug, Clone, Serialize)]
pub struct ScaleSensitivity {
    /// Efficiency of dispatching small grids (e.g. inference).
    pub small_kernel_efficiency: &'static str,
    /// Scaling behavior on massive grids.
    pub large_kernel_scaling: &'static str,
    /// Point at which thread blocks enter a latency-dominated execution mode.
    pub latency_dominated_regime_threshold: &'static str,
    /// Viability of warp specialization (producer/consumer patterns).
    pub warp_specialization_viability: &'static str,
}

/// Model dictating how data movement should be approached.
#[derive(Debug, Clone, Serialize)]
pub struct DataMovementModel {
    /// Cost of DRAM latency cache misses.
    pub dram_latency_cost: &'static str,
    /// Expected performance leverage from keeping data resident in the L2 cache.
    pub l2_reuse_leverage: &'static str,
    /// The necessity of amortizing loads through Shared Memory.
    pub smem_amortization: &'static str,
}

/// The shape of the GEMM problem to be solved.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ProblemShape {
    /// Dimension M: rows of matrix A and matrix C.
    pub m: usize,
    /// Dimension N: columns of matrix B and matrix C.
    pub n: usize,
    /// Dimension K: columns of matrix A and rows of matrix B.
    pub k: usize,
}

/// Supported data types for kernel benchmarking and prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit floating point.
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

/// Description of the hardware support for a specific data type.
#[derive(Debug, Clone, Serialize)]
pub struct NumericFeasibility {
    /// The original data type requested by the user.
    pub requested_dtype: DType,
    /// Whether the hardware has native ISA support for this data type.
    pub is_native: bool,
    /// The data type actually used for computation (may be a fallback).
    pub effective_compute_dtype: DType,
    /// Description of the execution mode (Native, Emulated, Unsupported).
    pub execution_mode: &'static str,
    /// Optional reason for emulation or lack of support.
    pub penalty_reason: Option<&'static str>,
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

/// A ranked implementation strategy with its associated probability, confidence, and uncertainty.
/// Prediction for a single implementation strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelPrediction {
    /// Name of the kernel implementation strategy.
    pub strategy: &'static str,
    /// Probability of this strategy being the optimal choice (0.0 to 1.0).
    pub probability: f32,
    /// Model uncertainty for this specific strategy (0.0 to 1.0).
    pub uncertainty: f32,
    /// Qualitative reasoning for the score.
    pub reasoning: &'static str,
}

/// Comprehensive hardware cost model predictions.
#[derive(Debug, Clone, Serialize)]
pub struct HardwarePredictor {
    /// The architecture compute capability base (e.g., 80) used for this prediction.
    pub arch_base_cc: u32,
    // ── Phase 12 Execution Profile ──
    /// High-level summary of the architectural regime.
    pub execution_profile: ExecutionProfile,

    // ── Phases 13/14/15/16 Probabilistic Policy ──
    /// Ranked distribution of implementation strategies with probabilities and uncertainty.
    pub likelihood_distribution: Vec<KernelPrediction>,
    /// Overall confidence in the detected architectural regime (0.0 to 1.0).
    pub regime_confidence: f32,
    /// Hardware ISA feasibility for the requested data type.
    pub feasibility: NumericFeasibility,
    /// The specific GEMM dimensions used for this prediction.
    pub problem_shape: ProblemShape,

    // ── Diagnostic / Basic Predictors ──
    /// Likelihood of being optimal for compute-bound dense GEMMs.
    pub compute_dense_gemm: Affinity,
    /// Likelihood of being optimal for Tensor Core accelerated GEMMs.
    pub tensor_core_gemm: Affinity,
    /// Likelihood of being optimal for memory-streaming operations.
    pub memory_streaming: Affinity,
    /// Likelihood of being optimal for heavy reduction workloads (e.g., flash attention).
    pub reduction_heavy: Affinity,
    /// Likelihood of being optimal for persistent fusion kernels.
    pub persistent_fusion: Affinity,

    // ── Phase 11 Global Behaviors ──
    /// High-level performance regime the architecture falls into.
    pub performance_regime: PerformanceRegime,
    /// Response to different kernel dispatch scales.
    pub scale_sensitivity: ScaleSensitivity,
    /// How the architecture penalizes or rewards data movement strategies.
    pub data_movement_model: DataMovementModel,

    // ── Phase 9/10 Deep Models ──
    /// The SM's execution and concurrency model.
    pub sm_execution_model: SmExecutionModel,
    /// Equilibrium ratios governing bandwidth vs compute.
    pub throughput_ratios: ThroughputRatios,
    /// Software-driven execution capabilities (barriers, async pipelines).
    pub software_execution_limits: SoftwareExecutionLimits,
    /// Executable numeric compute regimes.
    pub numeric_execution_modes: NumericExecutionModes,
    /// Limitations on data reuse per thread block.
    pub memory_reuse_envelope: MemoryReuseEnvelope,
    /// Specific architectural signals tailored for GEMM development.
    pub gemm_specific_signals: GemmSpecificSignals,

    // ── Estimated Resource Pressures ──
    /// Risk of hitting registry capacity limits.
    pub register_pressure_risk: PressureRisk,
    /// Risk of hitting shared memory capacity limits.
    pub shared_memory_pressure_risk: PressureRisk,
    /// Risk of saturating DRAM bandwidth before reaching compute limits.
    pub bandwidth_saturation_risk: PressureRisk,

    // ── Modeling Predictions ──
    /// A conservative range of likely optimal theoretical occupancy (e.g., "25 - 50").
    pub achievable_occupancy_range: (u32, u32),
    /// Qualitative estimation of the architecture's latency hiding capability.
    pub latency_hiding_sufficiency: &'static str,

    // ── Strategy Ranking ──
    /// A ranked list of implementation strategies suggested by the hardware traits.
    pub implementation_strategies: Vec<&'static str>,
    /// Suggested parametric search space for autotuning.
    pub suggested_search_space: SearchSpace,
    /// Classification of the problem's shape physics.
    pub shape_classification: ShapeClassification,
    /// Predicted performance signature of this implementation.
    pub performance_signature: PerformanceSignature,
    /// Whether the predictor is using empirical calibration data.
    pub is_calibrated: bool,
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchSpace {
    /// Plausible thread block sizes (M x N).
    pub plausible_block_sizes: Vec<(u32, u32)>,
    /// Plausible K-loop unrolling/blocking factors.
    pub plausible_k_blocking: Vec<u32>,
    /// Valid Shared Memory staging depths.
    pub smem_stages: Vec<u32>,
}

/// Categorization of GEMM shape physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeClassification {
    /// The detected shape category.
    pub case: ShapeCase,
    /// The axis that dominates the performance or constraints.
    pub dominant_axis: &'static str,
    /// The rationale for the suggested tiling strategy.
    pub tiling_rationale: String,
}

/// Predicted performance footprint and resource utilization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSignature {
    /// The bottleneck regime the kernel is expected to fall into.
    pub regime: &'static str,
    /// Predicted percentage of peak FLOPs achievable (0.0 to 1.0).
    pub flop_utilization: f32,
    /// Predicted percentage of available bandwidth used (0.0 to 1.0).
    pub bw_utilization: f32,
    /// Estimated pressure on the warp schedulers (0.0 to 1.0).
    pub scheduler_pressure: f32,
    /// Predicted risk of register spilling (0.0 to 1.0).
    pub risk_of_spill: f32,
    /// The primary hardware limit restricting performance.
    pub dominant_limit: &'static str,
}

/// Precise verification targets for hardware profilers like Nsight Compute.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationTargets {
    /// Expected percentage of issue slots utilized (0.0 to 1.0).
    pub expected_issue_utilization: f32,
    /// Qualitative expectation of shared memory reuse.
    pub expected_smem_reuse: &'static str,
    /// The primary expected reason for warp stalls.
    pub expected_stall_reason: &'static str,
}

impl HardwarePredictor {
    /// Evaluates the architecture to provide implementation predictions.
    pub fn evaluate(
        arch: &GpuArch,
        obs: &ArchObservables,
        derived: &DerivedProperties,
        shape: ProblemShape,
        dtype: DType,
    ) -> Self {
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
            PressureRisk::Low
        } else {
            PressureRisk::Medium
        };

        let shared_memory_pressure_risk = if obs.shared_mem_per_sm.value >= 96 * 1024 {
            PressureRisk::Low
        } else {
            PressureRisk::Medium
        };

        let bandwidth_saturation_risk = if derived.roofline_fp32 > 70.0 {
            PressureRisk::High // Very high compute-to-BW means BW will saturate quickly
        } else if derived.roofline_fp32 > 40.0 {
            PressureRisk::Medium
        } else {
            PressureRisk::Low
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

        let latency_hiding_sufficiency = if arch.base >= 80 && derived.tensor_core_dominance > 0.0 {
            "Requires async pipelines (cp.async/TMA) to hide memory latency"
        } else if obs.max_threads_per_sm.value >= 2048 {
            "Sufficient via thread-level parallelism (high occupancy required)"
        } else {
            "Borderline; requires careful ILP instruction unrolling"
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
            ExecutionArchClass::DataflowGpu => "Tensor Dataflow (TMEM operands)",
            ExecutionArchClass::PipelineGpu => "WGMMA Pipeline (TMA orchestrated)",
            ExecutionArchClass::AsyncTensor => "Async Tensor Pipeline (cp.async)",
            ExecutionArchClass::EarlyTensor => "Synchronous Tensor Core (WMMA)",
            ExecutionArchClass::PreTensor => "Register-blocked CUDA Core",
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
        let fp32_math_class = if arch.base >= 60 { "native fused multi-add" } else { "standard multi-add" };
        let fp16_math = if arch.base >= 53 { "viable (native ISA)" } else { "emulated (via FP32)" };
        let half2_vector_math = if obs.fp8_flops_tflops.value > 0.0 || obs.tensor_core_flops_tflops.value > 0.0 {
            "superseded by Tensor Cores"
        } else if arch.base >= 70 || arch.base == 60 || arch.base == 62 || arch.base == 53 {
            "viable (half2 native support)"
        } else {
            "unviable (throughput throttled, prefer FP32 compute)"
        };
        let int8_dot_product = if arch.base >= 61 { "viable (DP4A native support)" } else { "unavailable / emulated" };
        let mixed_precision_accumulate = if derived.tensor_core_dominance > 0.0 {
            "highly viable (native TC accumulation)" 
        } else if arch.base >= 60 {
             "limited (scalar accumulation)"
        } else {
             "expensive"
        };

        let numeric_execution_modes = NumericExecutionModes {
            fp32_math_class,
            fp16_math,
            half2_vector_math,
            int8_dot_product,
            mixed_precision_accumulate,
        };

        let expected_reuse_depth = if obs.registers_per_sm.value >= 65536 && obs.shared_mem_per_sm.value >= 96 * 1024 {
            "deep (high data reuse per LDG)"
        } else {
            "shallow (register capped)"
        };
        
        let smem_staging_depth = if arch.base >= 80 && obs.shared_mem_per_sm.value >= 160 * 1024 {
            "multi-stage (>=3 stages viable)"
        } else if obs.shared_mem_per_sm.value >= 96 * 1024 {
            "double-buffered (2 stages)"
        } else {
            "single-stage or limited"
        };

        let register_tiling_headroom = if obs.max_threads_per_sm.value >= 2048 {
            "abundant"
        } else if obs.max_threads_per_sm.value >= 1536 {
            "moderate"
        } else {
            "constrained"
        };

        let register_reuse_leverage = if arch.base == 61 {
            "very high (RF large, spills catastrophic)"
        } else if arch.base >= 80 {
            "moderate (mitigated by async loads)"
        } else {
            "high (standard blocking leverage)"
        };

        let bandwidth_pressure_regime = if derived.roofline_fp32 > 60.0 {
            "DRAM severely constrained (requires max locality)"
        } else if derived.roofline_fp32 > 35.0 {
            "L2/DRAM balanced constraint"
        } else {
            "compute latency bound"
        };

        let memory_reuse_envelope = MemoryReuseEnvelope {
            expected_reuse_depth,
            smem_staging_depth,
            register_tiling_headroom,
            register_reuse_leverage,
            bandwidth_pressure_regime,
        };

        // --- 1.8. Phase 11 Global Execution Behaviors ---
        let compute_vs_memory_balance = if derived.roofline_fp32 > 60.0 {
            "memory-sensitive (requires high arithmetic intensity)"
        } else if derived.roofline_fp32 < 30.0 {
            "compute-bound (easily fed by memory)"
        } else {
            "balanced (sensitive to both limits)"
        };

        let register_pressure_sensitivity = if obs.max_threads_per_sm.value >= 2048 {
            "low (abundant resources per SM)"
        } else {
            "high (spilling heavily penalizes occupancy)"
        };

        let scheduler_pressure_sensitivity = if obs.max_warps_per_sm.value / obs.schedulers_per_sm.value >= 12 {
            "high (requires massive thread-level parallelism or ILP)"
        } else {
            "medium (standard warp scheduling suffices)"
        };

        let instruction_level_parallelism_sensitivity = if arch.base == 61 {
            "high (dual-issue absent, relies on ILP)"
        } else {
            "moderate (arch supports dual-issue or warp-specialization)"
        };

        let issue_slot_utilization_target = if arch.base <= 61 {
             "≥ 70% to hide latency"
        } else {
             "optimized via hardware dependency tracking"
        };

        let performance_regime = PerformanceRegime {
            compute_vs_memory_balance,
            register_pressure_sensitivity,
            scheduler_pressure_sensitivity,
            instruction_level_parallelism_sensitivity,
            issue_slot_utilization_target,
        };

        let small_kernel_efficiency = if arch.base >= 80 {
            "low (pipeline depth requires massive grids to hide latency)"
        } else {
            "moderate (better suited for smaller work drops)"
        };

        let large_kernel_scaling = "excellent (saturates SMs linearly)";

        let latency_dominated_regime_threshold = if arch.base >= 80 {
            "< 1024 threads per block"
        } else {
            "< 256 threads per block"
        };

        let warp_specialization_viability = if arch.base >= 80 {
            "high (async pipelines enable producer/consumer)"
        } else {
            "low (no async copy or TC pipeline)"
        };

        let scale_sensitivity = ScaleSensitivity {
            small_kernel_efficiency,
            large_kernel_scaling,
            latency_dominated_regime_threshold,
            warp_specialization_viability,
        };

        let dram_latency_cost = if arch.base == 61 {
            "unhidable without occupancy (no async mechanisms)"
        } else if arch.base <= 75 {
            "high (no async overlap mechanisms)"
        } else {
            "high (requires algorithmic latency hiding)"
        };

        let l2_reuse_leverage = if arch.base == 61 {
             "moderate (streaming-biased, limited reuse cache)"
        } else if obs.l2_bytes.value >= 16 * 1024 * 1024 {
            "critical (L2 residency dominates performance)"
        } else {
            "moderate (L2 acts strictly as a victim/streaming cache)"
        };

        let smem_amortization = if arch.base <= 61 {
            "strongly beneficial (essential for peak GEMM)"
        } else if derived.roofline_fp32 > 50.0 {
            "mandatory for peak performance"
        } else {
            "beneficial but not strictly mandatory"
        };

        let data_movement_model = DataMovementModel {
            dram_latency_cost,
            l2_reuse_leverage,
            smem_amortization,
        };

        let preferred_tile_size_range = if arch.base >= 80 { (128, 256) } else { (64, 128) };
        let reuse_depth_achievable = if obs.registers_per_sm.value >= 65536 { Affinity::Medium } else { Affinity::Low };
        let split_k_viability = if arch.base == 61 {
             "Only for extreme aspect ratios or very large K"
        } else if derived.roofline_fp32 > 60.0 {
             "Highly advisable for MxN < 4096"
        } else {
             "Only for extreme edge cases"
        };
        let epilogue_fusion_headroom = if arch.base >= 80 && obs.max_threads_per_sm.value >= 1536 { Affinity::Medium } else { Affinity::Low };

        let gemm_specific_signals = GemmSpecificSignals {
            preferred_tile_size_range,
            reuse_depth_achievable,
            split_k_viability,
            epilogue_fusion_headroom,
        };

        let is_calibrated = obs.dram_bandwidth_gbps.calibration_factor.is_some()
            || obs.fp32_flops_tflops.calibration_factor.is_some()
            || obs.tensor_core_flops_tflops.calibration_factor.is_some();

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
        let mut candidates = Vec::new();

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
        candidates.push(("Blocked CUDA-core GEMM with double-buffered SMEM", s_blocked, u_blocked, "Reliant on standard TLP and thread-level latency hiding."));

        // Strategy B: Register-heavy warp tiling
        let (s_warp, u_warp) = {
            let mut score = 2.5;
            if rf_val > 2.0 { score += 1.0; }
            if arch.base >= 80 { score += 1.0; } 
            if register_reuse_leverage.contains("very high") { score += 1.0; }
            if is_small_m_n { score += 2.0; } // Warp tiling excels at small shapes
            (score + (1.0 - occ_val), 0.12) // Slightly more sensitive to compiler scheduling
        };
        candidates.push(("Register-heavy warp tiling kernel (avoid spilling)", s_warp, u_warp, "Optimized for large register files and high instruction-level parallelism."));

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
            candidates.push((name, s_tensor, u_tensor, "Leverages dedicated matrix-multiplication hardware for peak throughput."));
        }

        // Strategy D: Persistent L2-resident fusion
        if persistent_fusion != Affinity::Low {
            let s_persistent = {
                let mut score = 2.0;
                if obs.l2_bytes.value >= 16 * 1024 * 1024 { score += 3.0; }
                if persistence_viable(arch, obs) { score += 1.0; }
                if is_small_m_n { score += 1.5; } // Fusion is great for removing bandwidth bounds on small shapes
                score
            };
            candidates.push(("Persistent L2-resident fusion kernel", s_persistent, 0.20, "Minimizes global memory traffic by keeping tiles in the L2 cache."));
        }

        // Strategy E: Split-K
        if split_k_viability.contains("Advisable") || (is_large_k && is_skinny) {
            let s_split = if is_large_k && is_skinny { 5.5 } else { 3.0 };
            candidates.push(("Split-K reduction kernel", s_split, 0.15, "Increases parallelism by partitioning the K dimension across multiple blocks."));
        }

        // Apply Softmax (Temperature = 1.0)
        let exp_scores: Vec<f32> = candidates.iter().map(|&(_, s, _, _)| s.exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        
        for (i, &(strategy, _, uncertainty, reasoning)) in candidates.iter().enumerate() {
            likelihood_distribution.push(KernelPrediction {
                strategy,
                probability: exp_scores[i] / sum_exp,
                uncertainty,
                reasoning,
            });
        }
        
        likelihood_distribution.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());

        let implementation_strategies: Vec<&'static str> = likelihood_distribution.iter().map(|p| p.strategy).collect();
        let suggested_search_space = derive_search_space(arch, obs, &derived, &execution_profile);

        let shape_classification = classify_shape(obs, &shape);
        let performance_signature = estimate_performance_signature(
            arch, 
            obs, 
            &derived, 
            &execution_profile, 
            &shape,
            &feasibility
        );

        Self {
            arch_base_cc: arch.base as u32,
            execution_profile,
            likelihood_distribution,
            regime_confidence,
            feasibility,
            problem_shape: shape,

            compute_dense_gemm,
            tensor_core_gemm,
            memory_streaming,
            reduction_heavy,
            persistent_fusion,

            performance_regime,
            scale_sensitivity,
            data_movement_model,

            sm_execution_model,
            throughput_ratios,
            software_execution_limits,
            numeric_execution_modes,
            memory_reuse_envelope,
            gemm_specific_signals,

            register_pressure_risk,
            shared_memory_pressure_risk,
            bandwidth_saturation_risk,

            achievable_occupancy_range,
            latency_hiding_sufficiency,

            implementation_strategies,
            suggested_search_space,
            shape_classification,
            performance_signature,
            is_calibrated,
        }
    }

    /// Emits the top-ranked kernel intent for machine consumption (codegen/autotuning).
    pub fn emit_intent(&self) -> Option<KernelIntent> {
        self.implementation_strategies.first().map(|family| {
            // Determine tile strategy and pipeline depth based on architecture class
            let (tile_strategy, pipeline_depth) = match self.execution_profile.arch_class {
                ExecutionArchClass::DataflowGpu => ("tmem_dataflow", 4),
                ExecutionArchClass::PipelineGpu => ("tma_pipelined", 3),
                ExecutionArchClass::AsyncTensor => ("async_pipelined", 2),
                _ => ("smem_blocked_double_buffered", 2),
            };

            // Derive occupancy target from the predicted range
            let occupancy_target = self.achievable_occupancy_range.0 as f32 / 100.0;

            KernelIntent {
                kernel_family: family.to_string(),
                tile_strategy: tile_strategy.to_string(),
                pipeline_depth,
                dtype_compute: self.feasibility.effective_compute_dtype,
                dtype_storage: self.feasibility.requested_dtype,
                occupancy_target,
                priority: self.regime_confidence * 0.95, // Prioritize slightly below absolute confidence
                suggested_search_space: self.suggested_search_space.clone(),
                sampled_configs: self.sample_configs(3), // Top 3 samples
                shape_class: self.shape_classification.clone(),
                performance_signature: self.performance_signature.clone(),
                expected_runtime_us: self.calculate_expected_runtime(),
                runtime_interval_us: self.calculate_runtime_interval(),
                runtime_confidence: self.regime_confidence * 0.8, // Slightly lower confidence for runtime
                verification_targets: self.derive_verification_targets(),
            }
        })
    }

    /// Calculates expected runtime based on predicted utilization and flop count.
    fn calculate_expected_runtime(&self) -> f32 {
        let m = self.problem_shape.m as f64;
        let n = self.problem_shape.n as f64;
        let k = self.problem_shape.k as f64;
        let total_flops = 2.0 * m * n * k;
        
        // Use peak TFLOPS (analytical baseline)
        let est_tflops = 10.0 * self.performance_signature.flop_utilization as f64;
        if est_tflops > 0.0 {
            (total_flops / (est_tflops * 1e12) * 1e6) as f32
        } else {
            0.0
        }
    }

    /// Calculates a realistic uncertainty interval for the runtime prediction.
    fn calculate_runtime_interval(&self) -> [f32; 2] {
        let expected = self.calculate_expected_runtime();
        let confidence = self.regime_confidence * 0.8;
        
        let min_us = expected * 0.9; // Optimistic (assuming perfect overlap)
        let max_us = expected * (1.0 + (1.0 - confidence) * 0.5); // Pessimistic (noise/contention)
        
        [min_us, max_us]
    }

    /// Derives specific verification targets for hardware profiling.
    fn derive_verification_targets(&self) -> VerificationTargets {
        let sig = &self.performance_signature;
        
        let expected_smem_reuse = if self.problem_shape.k > 1024 {
            "high (K-blocking leverage)"
        } else if self.problem_shape.m > 256 || self.problem_shape.n > 256 {
            "medium (Tile-level reuse)"
        } else {
            "minimal (Shape too small for blocking residency)"
        };

        let expected_stall_reason = if sig.dominant_limit == "dram_bandwidth" {
            "memory_dependency (LDG/STG bound)"
        } else if sig.scheduler_pressure > 0.8 {
            "instruction_fetch / pipeline_busy"
        } else if sig.flop_utilization < 0.3 {
            "not_selected (Inadequate TLP/Grid scale)"
        } else {
            "execution_dependency (Scoreboard/Math pipe)"
        };

        VerificationTargets {
            expected_issue_utilization: sig.flop_utilization * 1.1, // Issue rate is usually higher than effective FLOP util
            expected_smem_reuse,
            expected_stall_reason,
        }
    }

    /// Adjusts strategy probabilities based on recorded hardware feedback.
    pub fn apply_feedback(&mut self, facts: &[FeedbackFact]) {
        if facts.is_empty() { return; }

        for fact in facts {
            // Check if this fact belongs to this architecture and requested precision
            if fact.arch_base == self.arch_base_cc && fact.dtype == self.feasibility.requested_dtype {
                if let Some(pred) = self.likelihood_distribution.iter_mut().find(|p| p.strategy == fact.strategy) {
                    // Update probability based on relative error (simple proportional shift)
                    // If observed performance is much lower than predicted (relative to peak), penalize.
                    // This is a placeholder for a true Bayesian update or error model.
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
        let sum: f32 = self.likelihood_distribution.iter().map(|p| p.probability).sum();
        if sum > 0.0 {
            for p in &mut self.likelihood_distribution {
                p.probability /= sum;
            }
        }
        
        // Re-sort strategies based on updated probabilities
        self.likelihood_distribution.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
        self.implementation_strategies = self.likelihood_distribution.iter().map(|p| p.strategy).collect();
    }

    /// Samples N candidate configurations weighted by their predicted performance scores.
    pub fn sample_configs(&self, limit: usize) -> Vec<SampledConfig> {
        let mut samples = Vec::new();
        let ss = &self.suggested_search_space;

        // For each strategy in our distribution, cross-product with the search space
        for prediction in &self.likelihood_distribution {
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
        "use asynchronous staging to hide latency."
    } else {
        "software double-buffering to overlap LDG and compute."
    };

    if m < 16.0 || n < 16.0 || k < 16.0 {
        return ShapeClassification {
            case: ShapeCase::SmallScale,
            dominant_axis: "Dispatch",
            tiling_rationale: format!("Small dimensions suggest kernel launch overhead dominates; prefer minimal tiling and {}", overlap_rationale),
        };
    }

    let m_n_ratio = m / n;
    let k_m_ratio = k / m;
    let k_n_ratio = k / n;

    if m_n_ratio >= 4.0 && k_m_ratio <= 0.5 {
        ShapeClassification {
            case: ShapeCase::TallSkinnyM,
            dominant_axis: "M",
            tiling_rationale: format!("Tall-skinny M dominant; maximize row-wise tiling and consider split-K if occupancy is low. {}", overlap_rationale),
        }
    } else if m_n_ratio <= 0.25 && k_n_ratio <= 0.5 {
        ShapeClassification {
            case: ShapeCase::WideSkinnyN,
            dominant_axis: "N",
            tiling_rationale: format!("Wide-skinny N dominant; maximize column-wise tiling to leverage burst memory access. {}", overlap_rationale),
        }
    } else if k / (m.max(n)) >= 2.0 {
        ShapeClassification {
            case: ShapeCase::KDominant,
            dominant_axis: "K",
            tiling_rationale: format!("K-dominant reduction; maximize reuse along K axis and {}", overlap_rationale),
        }
    } else {
        ShapeClassification {
            case: ShapeCase::LargeSquare,
            dominant_axis: "None (Square)",
            tiling_rationale: format!("Square-ish compute-bound GEMM; use balanced 2D tiling (e.g. 128x128) to maximize compute/memory ratio. {}", overlap_rationale),
        }
    }
}

fn estimate_performance_signature(
    arch: &GpuArch,
    obs: &ArchObservables,
    derived: &DerivedProperties,
    profile: &ExecutionProfile,
    shape: &ProblemShape,
    _feasibility: &NumericFeasibility,
) -> PerformanceSignature {
    // Arithmetic Intensity Calculation
    let m = shape.m as f64;
    let n = shape.n as f64;
    let k = shape.k as f64;
    let bytes_per_elem = 2.0; // Assume Fp16/Bf16 baseline
    
    let total_flops = 2.0 * m * n * k;
    let total_bytes = bytes_per_elem * (m * k + k * n + m * n);
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

    let is_compute_bound = intensity > machine_balance;
    
    let (regime, dominant_limit) = if has_tensor_cores {
        if is_compute_bound {
            ("Compute Bound (Tensor Core limited)", "tensor_unit_throughput")
        } else {
            ("Memory Bound (DRAM limited)", "dram_bandwidth")
        }
    } else {
        // Pascal / Older
        if is_compute_bound {
            ("Latency-masked compute regime", "scheduler_issue + ILP")
        } else {
            ("Memory Bound (DRAM limited)", "dram_bandwidth")
        }
    };

    // Utilization estimates (Nuanced Modeling)
    // Small shapes (m, n < 128) suffer from wave quantization and launch overhead
    let shape_penalty = if shape.m < 128usize || shape.n < 128usize { 0.7 } else { 0.95 };
    
    let mut flop_util = if is_compute_bound { 0.65 * shape_penalty } else { (intensity / machine_balance) * shape_penalty };
    
    // Bandwidth utilization:
    // If we have high K and SMEM blocking, we don't saturate BW (we reuse tiles)
    let reuse_multiplier = if shape.k > 1024usize { 0.6 } else { 0.9 };
    let mut bw_util = if is_compute_bound { (machine_balance / intensity) * reuse_multiplier } else { 0.8 * reuse_multiplier };

    // Final clamping
    flop_util = flop_util.min(0.9);
    bw_util = bw_util.min(0.9); // Rarely saturate 100% in sustainable regime

    // Scheduler pressure: High if persistent or many small blocks
    let scheduler_pressure = match profile.parallelism_model {
        ParallelismRegime::Persistent => 0.85,
        ParallelismRegime::OccupancyBound => 0.6,
        _ => 0.4,
    };

    // Risk of spill: High if register heavy or huge tiles
    let risk_of_spill = if derived.max_tile_k_fp32 < 16 { 0.7 } else { 0.1 };

    PerformanceSignature {
        regime,
        flop_utilization: flop_util as f32,
        bw_utilization: bw_util as f32,
        scheduler_pressure: scheduler_pressure as f32,
        risk_of_spill: risk_of_spill as f32,
        dominant_limit,
    }
}
fn persistence_viable(arch: &GpuArch, obs: &ArchObservables) -> bool {
    arch.base >= 70 && obs.l2_bytes.value >= 4 * 1024 * 1024
}

/// Actual performance observation from a benchmarked kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    let (is_native, effective_compute_dtype, execution_mode, penalty_reason) = match dtype {
        DType::Fp32 => (true, DType::Fp32, "Native", None),
        DType::Fp16 => {
            if arch.base >= 53 {
                (true, DType::Fp16, "Native", None)
            } else {
                (false, DType::Fp32, "Emulated", Some("Hardware lack FP16 ISA, remapped to FP32 compute."))
            }
        }
        DType::Bf16 => {
            if arch.base >= 80 {
                (true, DType::Bf16, "Native", None)
            } else {
                (false, DType::Fp32, "Emulated", Some("Hardware lack BF16 ISA, remapped to FP32 compute."))
            }
        }
        DType::Fp8 => {
            if arch.base >= 90 {
                (true, DType::Fp8, "Native", None)
            } else if arch.base >= 70 {
                (false, DType::Fp16, "Emulated", Some("Native FP8 requires CC 9.0+, emulated via FP16 compute (bandwidth gain only)."))
            } else {
                (false, DType::Fp32, "Emulated", Some("Native FP8 requires CC 9.0+, emulated via FP32 compute (bandwidth gain only)."))
            }
        }
        DType::Int8 => {
            if arch.base >= 61 {
                (true, DType::Int8, "Native", None)
            } else {
                (false, DType::Fp32, "Emulated", Some("Native DP4A/INT8 requires CC 6.1+, remapped to FP32 compute."))
            }
        }
    };

    NumericFeasibility {
        requested_dtype: dtype,
        is_native,
        effective_compute_dtype,
        execution_mode,
        penalty_reason,
    }
}
