//! Scientific GPU architecture metrics module.
//!
//! This module provides **post-compilation runtime risk estimators** and physical facts
//! about the GPU micro-architecture.
//!
//! # Hybrid Build-Time / Run-Time Design
//! This module is designed to be used in two primary contexts:
//! 1. **Build-Time (`build.rs`)**: Generate static C++ macros via [`to_c_defines()`]
//!    to inject hardware constraints directly into JIT-compiled kernels (e.g., NVRTC).
//!    This provides **zero runtime overhead**, as the metrics are baked into the binary.
//! 2. **Run-Time**: Query [`ArchObservables`] dynamically during execution to adapt
//!    workloads, configure memory pools, or dispatch specialized algorithms based
//!    on the host GPU's characteristics.
//!
//! # Four-layer architecture
//! 1. **Provenance** ([`DataSource`], [`Measured<T>`]) — tracks where data comes from
//! 2. **Observables** ([`ArchObservables`]) — physical, measurable GPU properties
//! 3. **Models** — mathematical functions with documented assumptions and validity domains
//! 4. **Derived** ([`DerivedProperties`]) — results computed from observables via models
//!
//! # Scientific classification
//! Every model function is annotated with a [`ModelType`] in its documentation:
//! - [`ModelType::PhysicalLaw`] — exact under stated assumptions
//! - [`ModelType::AnalyticalBound`] — proven upper/lower bound
//! - [`ModelType::EmpiricalFit`] — calibrated from measurements
//! - [`ModelType::Heuristic`] — rule of thumb, use with caution

use crate::compute_cap::GpuArch;
use serde::Serialize;

// ── Layer 1: Provenance ──────────────────────────────────────────────────────

/// Source of a measurement or data point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum DataSource {
    /// From NVIDIA architecture whitepaper or tuning guide
    NvidiaSpec,
    /// Queried at runtime via cudaDeviceGetAttribute or similar
    RuntimeQuery,
    /// Obtained from a microbenchmark (e.g. pointer-chasing for L2 latency)
    Microbenchmark,
    /// Provided by the user explicitly
    UserProvided,
}

/// Formal classification of a model function's scientific basis.
///
/// Used in documentation to communicate the confidence level of each model.
/// This enum is intentionally not referenced in runtime code — it serves as
/// a formal annotation system for the model functions' docstrings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum ModelType {
    /// Exact analytical result under stated assumptions (e.g. occupancy formula).
    PhysicalLaw,
    /// Proven upper or lower bound, not necessarily tight (e.g. max tile size).
    AnalyticalBound,
    /// Fitted from empirical measurements; requires re-calibration for new hardware.
    EmpiricalFit,
    /// Rule of thumb without formal derivation; document strongly or prefer alternatives.
    Heuristic,
}

/// Metrics that can be empirically calibrated via runtime probes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalibrationMetric {
    /// Peak DRAM bandwidth in GB/s.
    DramBandwidth,
    /// Peak L2 cache bandwidth in GB/s.
    L2Bandwidth,
    /// Peak FP32 arithmetic throughput in TFLOPS.
    Fp32Compute,
    /// Peak FP16 arithmetic throughput in TFLOPS.
    Fp16Compute,
    /// Peak Tensor Core (FP16/BF16) throughput in TFLOPS.
    TensorCoreCompute,
}

/// A measured fact about the current hardware used to calibrate the model.
#[derive(Debug, Clone)]
pub struct CalibrationFact {
    /// The metric being reported.
    pub metric: CalibrationMetric,
    /// The measured value (e.g., GiB/s or TFLOPS).
    pub measured_value: f32,
}

/// A semantic multiplier used to calibrate analytical models.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CalibrationFactor {
    /// The multiplicative scaling ratio applied to the base value.
    pub ratio: f32,
    /// Semantic meaning of this calibration adjustment.
    pub meaning: String,
    /// The specific measurement or methodology this factor was derived from.
    pub derived_from: String,
}

/// A value annotated with its provenance and optional calibration factor.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Measured<T> {
    /// The measured value
    pub value: T,
    /// Where this value comes from
    pub source: DataSource,
    /// Optional semantic calibration factor.
    /// When `Some(f)`, the effective value is `value * f.ratio`.
    pub calibration_factor: Option<CalibrationFactor>,
}

impl<T> Measured<T> {
    /// Create a new measurement sourced from NVIDIA specifications.
    pub fn from_spec(value: T) -> Self {
        Self {
            value,
            source: DataSource::NvidiaSpec,
            calibration_factor: None,
        }
    }

    /// Create a new measurement sourced from runtime queries.
    pub fn from_runtime(value: T) -> Self {
        Self {
            value,
            source: DataSource::RuntimeQuery,
            calibration_factor: None,
        }
    }

    /// Create a new measurement sourced from the user.
    pub fn from_user(value: T) -> Self {
        Self {
            value,
            source: DataSource::UserProvided,
            calibration_factor: None,
        }
    }
}

impl Measured<f32> {
    /// Returns the effective value (value * calibration_factor).
    pub fn effective(&self) -> f32 {
        self.value * self.calibration_factor.as_ref().map(|c| c.ratio).unwrap_or(1.0)
    }
}

// ── Layer 2: Observables ─────────────────────────────────────────────────────

/// Physical, measurable properties of a GPU SM generation.
///
/// All values come from NVIDIA architecture whitepapers, CUDA Programming Guide
/// Table 16 (Technical Specifications per Compute Capability), and official
/// product datasheets.
///
/// # ⚠ SKU caveat
/// **DRAM bandwidth and FLOPS figures correspond to representative flagship GPUs,
/// not architectural invariants.** Different SKUs within the same compute capability
/// (e.g. RTX 4090 vs RTX 4070) have different core counts, clock speeds, and memory
/// bandwidth. The [`reference_gpu`](ArchObservables::reference_gpu) field documents
/// which specific SKU the numbers come from.
///
/// For precise per-SKU data, override these values via [`Measured::from_runtime`]
/// or [`Measured::from_user`] with actual device queries.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArchObservables {
    /// Human-readable name of the flagship GPU these numbers come from.
    /// This is NOT an architectural property — it identifies the specific SKU
    /// used as reference for bandwidth/TFLOPS figures.
    pub reference_gpu: String,
    /// Number of 32-bit registers per SM (source: CUDA Prog Guide Table 16)
    pub registers_per_sm: Measured<u32>,
    /// Maximum configurable shared memory per SM in bytes (source: tuning guides)
    pub shared_mem_per_sm: Measured<u32>,
    /// Maximum resident threads per SM (source: CUDA Prog Guide Table 16)
    pub max_threads_per_sm: Measured<u32>,
    /// Warp size (always 32 for all NVIDIA GPUs to date)
    pub warp_size: u32,
    /// L2 cache size in bytes (source: architecture whitepapers)
    pub l2_bytes: Measured<u64>,
    /// Peak DRAM bandwidth in GB/s (source: product datasheet, **SKU-specific**)
    pub dram_bandwidth_gbps: Measured<f32>,
    /// Peak Tensor Core throughput in TFLOPS (FP16, source: product datasheet, **SKU-specific**)
    pub tensor_core_flops_tflops: Measured<f32>,
    /// Peak FP32 throughput in TFLOPS (source: product datasheet, **SKU-specific**)
    pub fp32_flops_tflops: Measured<f32>,
    /// Peak FP8 throughput in TFLOPS (dense, w/ Tensor Cores, **SKU-specific**)
    /// `0.0` if FP8 is not supported on this architecture.
    pub fp8_flops_tflops: Measured<f32>,
    /// Peak INT8 throughput in TOPS (dense, w/ Tensor Cores or DP4A, **SKU-specific**)
    pub int8_tops: Measured<f32>,
    /// Maximum concurrent warps per SM (source: CUDA Prog Guide Table 16)
    pub max_warps_per_sm: Measured<u32>,
    /// Maximum resident thread blocks per SM (source: CUDA Prog Guide Table 16)
    pub max_blocks_per_sm: Measured<u32>,
    /// Number of warp schedulers per SM
    pub schedulers_per_sm: Measured<u32>,
    /// Peak L2 cache bandwidth in GB/s (**SKU-specific**)
    pub l2_bandwidth_gbps: Measured<f32>,
    /// Peak Shared Memory bandwidth across the entire GPU in GB/s (**SKU-specific**)
    pub shared_mem_bandwidth_gbps: Measured<f32>,
}

impl ArchObservables {
    /// Applies empirical scaling coefficients to the architectural metrics.
    pub fn apply_coefficients(&mut self, flop_scale: f32, bw_scale: f32) {
        self.dram_bandwidth_gbps.calibration_factor = Some(CalibrationFactor {
            ratio: bw_scale,
            meaning: "empirical sustained throughput ratio vs theoretical peak".to_string(),
            derived_from: "microbench_bandwidth".to_string(),
        });
        self.fp32_flops_tflops.calibration_factor = Some(CalibrationFactor {
            ratio: flop_scale,
            meaning: "empirical compute efficiency under instruction-level pressure".to_string(),
            derived_from: "microbench_gemm".to_string(),
        });
        self.tensor_core_flops_tflops.calibration_factor = Some(CalibrationFactor {
            ratio: flop_scale,
            meaning: "empirical compute efficiency under instruction-level pressure".to_string(),
            derived_from: "microbench_gemm".to_string(),
        });
        self.fp8_flops_tflops.calibration_factor = Some(CalibrationFactor {
            ratio: flop_scale,
            meaning: "empirical compute efficiency under instruction-level pressure".to_string(),
            derived_from: "microbench_gemm".to_string(),
        });
        self.int8_tops.calibration_factor = Some(CalibrationFactor {
            ratio: flop_scale,
            meaning: "empirical compute efficiency under instruction-level pressure".to_string(),
            derived_from: "microbench_gemm".to_string(),
        });
        self.l2_bandwidth_gbps.calibration_factor = Some(CalibrationFactor {
            ratio: bw_scale,
            meaning: "cache hierarchy efficiency adjustment".to_string(),
            derived_from: "microbench_bandwidth".to_string(),
        });
        self.shared_mem_bandwidth_gbps.calibration_factor = Some(CalibrationFactor {
            ratio: bw_scale,
            meaning: "on-chip memory bus utilization correction".to_string(),
            derived_from: "microbench_bandwidth".to_string(),
        });
    }

    /// Applies empirical measurements to calibrate the architecture model.
    ///
    /// This updates the `calibration_factor` of internal metrics based on the
    /// ratio between the measured value and the architecturally expected (spec) value.
    pub fn calibrate(&mut self, facts: &[CalibrationFact]) {
        for fact in facts {
            let (target, meaning) = match fact.metric {
                CalibrationMetric::DramBandwidth => (&mut self.dram_bandwidth_gbps, "empirical sustained DRAM throughput"),
                CalibrationMetric::L2Bandwidth => (&mut self.l2_bandwidth_gbps, "empirical L2 cache throughput"),
                CalibrationMetric::Fp32Compute => (&mut self.fp32_flops_tflops, "empirical FP32 compute throughput"),
                CalibrationMetric::Fp16Compute | CalibrationMetric::TensorCoreCompute => 
                    (&mut self.tensor_core_flops_tflops, "empirical Tensor Core throughput"),
            };

            if target.value > 0.0 {
                target.calibration_factor = Some(CalibrationFactor {
                    ratio: fact.measured_value / target.value,
                    meaning: meaning.to_string(),
                    derived_from: "runtime_probe".to_string(),
                });
            }
        }
    }

    /// Build observables from a compute capability using NVIDIA specification data.
    ///
    /// Values correspond to **flagship SKUs** for each architecture. The
    /// [`reference_gpu`](ArchObservables::reference_gpu) field identifies which
    /// specific GPU model the DRAM bandwidth and TFLOPS figures come from.
    ///
    /// # Reference SKUs
    /// | CC | GPU | Die | Segment |
    /// |----|-----|-----|---------|
    /// | 61 | GTX 1080 Ti | GP102 | Consumer flagship |
    /// | 70 | V100 SXM2 | GV100 | Data center |
    /// | 75 | RTX 2080 Ti | TU102 | Consumer flagship |
    /// | 80 | A100 SXM | GA100 | Data center |
    /// | 86 | RTX 3090 | GA102 | Consumer flagship |
    /// | 87 | Jetson AGX Orin 64GB | GA10B | Edge / embedded |
    /// | 89 | RTX 4090 | AD102 | Consumer flagship |
    /// | 90 | H100 SXM | GH100 | Data center |
    /// | 100 | B200 | GB100 | Data center (preliminary) |
    /// | 120 | RTX 5090 | GB202 | Consumer (preliminary) |
    pub fn from_compute_cap(base: usize) -> Self {
        let su32 = |v: u32| Measured::from_spec(v);
        let su64 = |v: u64| Measured::from_spec(v);
        let sf32 = |v: f32| Measured::from_spec(v);

        match base {
            // ── Blackwell (preliminary) ──────────────────────────────
            100 | 120 => Self {
                reference_gpu: if base == 120 {
                    "RTX 5090 (GB202)".to_string()
                } else {
                    "B200 (GB100)".to_string()
                },
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(if base == 120 { 128 * 1024 } else { 228 * 1024 }),
                max_threads_per_sm: su32(2048),
                warp_size: 32,
                l2_bytes: su64(126 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(8000.0),
                tensor_core_flops_tflops: sf32(2500.0),
                fp32_flops_tflops: sf32(90.0),
                fp8_flops_tflops: sf32(4500.0),
                int8_tops: sf32(4500.0),
                max_warps_per_sm: su32(64),
                max_blocks_per_sm: su32(16),
                schedulers_per_sm: su32(4),
                l2_bandwidth_gbps: sf32(12000.0),
                shared_mem_bandwidth_gbps: sf32(25000.0),
            },
            // ── Hopper ───────────────────────────────────────────────
            90 => Self {
                reference_gpu: "H100 SXM (GH100)".to_string(),
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(228 * 1024),
                max_threads_per_sm: su32(2048),
                warp_size: 32,
                l2_bytes: su64(50 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(3350.0),
                tensor_core_flops_tflops: sf32(989.0),
                fp32_flops_tflops: sf32(66.9),
                fp8_flops_tflops: sf32(1979.0),
                int8_tops: sf32(1979.0),
                max_warps_per_sm: su32(64),
                max_blocks_per_sm: su32(32),
                schedulers_per_sm: su32(4),
                l2_bandwidth_gbps: sf32(12500.0), // ~12TB/s theoretical L2 BW H100
                shared_mem_bandwidth_gbps: sf32(23000.0), // ~23TB/s theoretical SMEM BW H100
            },
            // ── Ada Lovelace ─────────────────────────────────────────
            89 => Self {
                reference_gpu: "RTX 4090 (AD102)".to_string(),
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(100 * 1024),
                max_threads_per_sm: su32(1536),
                warp_size: 32,
                l2_bytes: su64(96 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(1008.0),
                tensor_core_flops_tflops: sf32(330.0),
                fp32_flops_tflops: sf32(82.6),
                fp8_flops_tflops: sf32(660.0),
                int8_tops: sf32(660.0),
                max_warps_per_sm: su32(48),
                max_blocks_per_sm: su32(24),
                schedulers_per_sm: su32(4),
                l2_bandwidth_gbps: sf32(5000.0),
                shared_mem_bandwidth_gbps: sf32(8500.0),
            },
            // ── Jetson AGX Orin ──────────────────────────────────────
            87 => Self {
                reference_gpu: "Jetson AGX Orin 64GB (GA10B)".to_string(),
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(128 * 1024),
                max_threads_per_sm: su32(1536),
                warp_size: 32,
                l2_bytes: su64(4 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(205.0),
                tensor_core_flops_tflops: sf32(170.0),
                fp32_flops_tflops: sf32(21.3),
                fp8_flops_tflops: sf32(0.0),
                int8_tops: sf32(170.0),
                max_warps_per_sm: su32(48),
                max_blocks_per_sm: su32(16),
                schedulers_per_sm: su32(4),
                l2_bandwidth_gbps: sf32(700.0),
                shared_mem_bandwidth_gbps: sf32(1200.0),
            },
            // ── Ampere consumer ──────────────────────────────────────
            86 => Self {
                reference_gpu: "RTX 3090 (GA102)".to_string(),
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(100 * 1024),
                max_threads_per_sm: su32(1536),
                warp_size: 32,
                l2_bytes: su64(6 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(936.0),
                tensor_core_flops_tflops: sf32(174.0),
                fp32_flops_tflops: sf32(29.8),
                fp8_flops_tflops: sf32(0.0),
                int8_tops: sf32(348.0),
                max_warps_per_sm: su32(48),
                max_blocks_per_sm: su32(16),
                schedulers_per_sm: su32(4),
                l2_bandwidth_gbps: sf32(2500.0),
                shared_mem_bandwidth_gbps: sf32(4500.0),
            },
            // ── Ampere data center ───────────────────────────────────
            80 => Self {
                reference_gpu: "A100 SXM (GA100)".to_string(),
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(164 * 1024),
                max_threads_per_sm: su32(2048),
                warp_size: 32,
                l2_bytes: su64(40 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(2039.0),
                tensor_core_flops_tflops: sf32(312.0),
                fp32_flops_tflops: sf32(19.5),
                fp8_flops_tflops: sf32(0.0),
                int8_tops: sf32(624.0),
                max_warps_per_sm: su32(64),
                max_blocks_per_sm: su32(32),
                schedulers_per_sm: su32(4),
                l2_bandwidth_gbps: sf32(6000.0),
                shared_mem_bandwidth_gbps: sf32(19000.0),
            },
            // ── Turing ───────────────────────────────────────────────
            75 => Self {
                reference_gpu: "RTX 2080 Ti (TU102)".to_string(),
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(64 * 1024),
                max_threads_per_sm: su32(1024),
                warp_size: 32,
                l2_bytes: su64(4 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(448.0),
                tensor_core_flops_tflops: sf32(65.0),
                fp32_flops_tflops: sf32(11.2),
                fp8_flops_tflops: sf32(0.0),
                int8_tops: sf32(130.0),
                max_warps_per_sm: su32(32),
                max_blocks_per_sm: su32(16),
                schedulers_per_sm: su32(4),
                l2_bandwidth_gbps: sf32(2000.0),
                shared_mem_bandwidth_gbps: sf32(3000.0),
            },
            // ── Volta ────────────────────────────────────────────────
            70 => Self {
                reference_gpu: "V100 SXM2 (GV100)".to_string(),
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(96 * 1024),
                max_threads_per_sm: su32(2048),
                warp_size: 32,
                l2_bytes: su64(6 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(900.0),
                tensor_core_flops_tflops: sf32(125.0),
                fp32_flops_tflops: sf32(15.7),
                fp8_flops_tflops: sf32(0.0),
                int8_tops: sf32(0.0),
                max_warps_per_sm: su32(64),
                max_blocks_per_sm: su32(32),
                schedulers_per_sm: su32(4),
                l2_bandwidth_gbps: sf32(3100.0),
                shared_mem_bandwidth_gbps: sf32(14000.0),
            },
            // ── Pascal and older ─────────────────────────────────────
            61 | 60 => Self {
                reference_gpu: "GTX 1080 Ti (GP102)".to_string(),
                registers_per_sm: su32(65536),
                shared_mem_per_sm: su32(96 * 1024),
                max_threads_per_sm: su32(2048),
                warp_size: 32,
                l2_bytes: su64(2 * 1024 * 1024),
                dram_bandwidth_gbps: sf32(320.0),
                tensor_core_flops_tflops: sf32(0.0),
                fp32_flops_tflops: sf32(11.3),
                fp8_flops_tflops: sf32(0.0),
                int8_tops: sf32(45.2), // DP4A throughput ≈ 4× FP32
                max_warps_per_sm: su32(64),
                max_blocks_per_sm: su32(32),
                schedulers_per_sm: su32(4), // GP102
                l2_bandwidth_gbps: sf32(1300.0),
                shared_mem_bandwidth_gbps: sf32(3500.0),
            },
            _ => Self {
                reference_gpu: "Generic Legacy GPU".to_string(),
                registers_per_sm: su32(32768),
                shared_mem_per_sm: su32(48 * 1024),
                max_threads_per_sm: su32(1024),
                warp_size: 32,
                l2_bytes: su64(1024 * 1024),
                dram_bandwidth_gbps: sf32(100.0),
                tensor_core_flops_tflops: sf32(0.0),
                fp32_flops_tflops: sf32(2.0),
                fp8_flops_tflops: sf32(0.0),
                int8_tops: sf32(0.0),
                max_warps_per_sm: su32(32),
                max_blocks_per_sm: su32(16),
                schedulers_per_sm: su32(2),
                l2_bandwidth_gbps: sf32(200.0),
                shared_mem_bandwidth_gbps: sf32(500.0),
            },
        }
    }
}

// ── Layer 3: Models ──────────────────────────────────────────────────────────
//
// Each model function documents:
// - ModelType classification
// - Assumptions
// - Validity domain (when the model is applicable)

/// Theoretical maximum occupancy given a per-thread register budget.
///
/// **Model type**: [`ModelType::PhysicalLaw`]
///
/// This is the exact NVIDIA occupancy calculator formula.
///
/// # Assumptions
/// - Even register allocation across all warps (no banking asymmetry)
/// - No register spill to local memory
/// - Scheduler distributes warps fairly across sub-partitions
///
/// # Validity domain
/// - All CUDA compute capabilities (CC 1.0+)
/// - Assumes shared memory is not the binding constraint
/// - Does not account for per-block shared memory limits
///
/// # Returns
/// Fraction in `[0.0, 1.0]` where 1.0 = 100% occupancy.
pub fn theoretical_occupancy_limit(obs: &ArchObservables, regs_per_thread: u32) -> f32 {
    if regs_per_thread == 0 {
        return 1.0;
    }
    let regs_per_warp = regs_per_thread * obs.warp_size;
    let max_warps_by_regs = obs.registers_per_sm.value / regs_per_warp;
    let max_warps_by_threads = obs.max_threads_per_sm.value / obs.warp_size;
    let actual_warps = max_warps_by_regs.min(max_warps_by_threads);
    actual_warps as f32 / max_warps_by_threads as f32
}

/// Roofline arithmetic intensity threshold for FP32 (FLOP/Byte).
///
/// **Model type**: [`ModelType::PhysicalLaw`]
///
/// Classic roofline model (Williams, Waterman, Patterson, 2009).
///
/// # Assumptions
/// - Peak FP32 throughput as compute ceiling
/// - Peak DRAM bandwidth as memory ceiling
/// - No cache effects (worst-case streaming access pattern)
///
/// # Validity domain
/// - Streaming kernels with no L2 reuse
/// - No instruction-level parallelism exploitation beyond peak issue rate
/// - Single-precision floating-point workloads
///
/// # Interpretation
/// Kernels with operational intensity below this value are memory-bound;
/// above are compute-bound. This is the "ridge point" of the roofline.
pub fn roofline_intensity(obs: &ArchObservables) -> f32 {
    let dram_bw = obs.dram_bandwidth_gbps.effective();
    if dram_bw <= 0.0 {
        return 0.0;
    }
    // Convert TFLOPS to GFLOPS for unit consistency with GB/s
    (obs.fp32_flops_tflops.effective() * 1000.0) / dram_bw
}

/// Roofline arithmetic intensity threshold for FP8 Tensor Core workloads (FLOP/Byte).
///
/// **Model type**: [`ModelType::PhysicalLaw`]
///
/// Same roofline model as [`roofline_intensity`], but using FP8 Tensor Core peak
/// as the compute ceiling. The ridge point shifts dramatically right for FP8
/// because throughput is much higher while memory bandwidth stays constant.
///
/// # Assumptions
/// - Peak FP8 TC throughput as compute ceiling
/// - Peak DRAM bandwidth as memory ceiling
/// - No cache effects (worst-case streaming)
///
/// # Validity domain
/// - CC 8.9+ (Ada Lovelace, Hopper, Blackwell)
/// - FP8 quantized GEMM / attention kernels
/// - Returns `0.0` on architectures without FP8 support
///
/// # Interpretation
/// Higher ridge point means harder to saturate compute — more kernels will
/// be memory-bound in FP8 than in FP32. Guides tile/prefetch strategy.
pub fn roofline_intensity_fp8(obs: &ArchObservables) -> f32 {
    let dram_bw = obs.dram_bandwidth_gbps.effective();
    let fp8_flops = obs.fp8_flops_tflops.effective();
    if dram_bw <= 0.0 || fp8_flops <= 0.0 {
        return 0.0;
    }
    (fp8_flops * 1000.0) / dram_bw
}

/// Roofline arithmetic intensity threshold for INT8 workloads (OP/Byte).
///
/// **Model type**: [`ModelType::PhysicalLaw`]
///
/// Same roofline model as [`roofline_intensity`], but using INT8 Tensor Core peak
/// (or DP4A on older architectures) as the compute ceiling.
///
/// # Assumptions
/// - Peak INT8 throughput (TOPS) as compute ceiling
/// - Peak DRAM bandwidth as memory ceiling
/// - No cache effects (worst-case streaming)
///
/// # Validity domain
/// - CC 6.1+ for DP4A (Pascal)
/// - CC 7.5+ for INT8 Tensor Cores (Turing+)
/// - Returns `0.0` on architectures without INT8 compute
///
/// # Interpretation
/// Guides W8A8 quantized inference kernel design: tile sizes, prefetch depth.
pub fn roofline_intensity_int8(obs: &ArchObservables) -> f32 {
    let dram_bw = obs.dram_bandwidth_gbps.effective();
    let int8_ops = obs.int8_tops.effective();
    if dram_bw <= 0.0 || int8_ops <= 0.0 {
        return 0.0;
    }
    // INT8 TOPS and DRAM GB/s are both in Giga-units, so ratio gives OP/Byte
    (int8_ops * 1000.0) / dram_bw
}

/// Ratio of Tensor Core FP16 peak throughput to FP32 scalar peak throughput.
///
/// **Model type**: [`ModelType::PhysicalLaw`] — this is a pure ratio of observables.
///
/// # Assumptions
/// - Both peaks are achievable (no pipeline contention)
/// - Tensor Core ops use FP16 accumulate
///
/// # Validity domain
/// - All architectures (returns 0.0 for pre-Volta without Tensor Cores)
///
/// # Interpretation
/// - `0.0` = no Tensor Cores (pre-Volta)
/// - `> 5.0` = TC strongly dominate; prefer TC paths for eligible ops
/// - `> 10.0` = TC are the primary compute engine (Hopper+)
pub fn tensor_core_dominance(obs: &ArchObservables) -> f32 {
    let fp32_flops = obs.fp32_flops_tflops.effective();
    if fp32_flops <= 0.0 {
        return 0.0;
    }
    obs.tensor_core_flops_tflops.effective() / fp32_flops
}

/// Ratio of FP8 Tensor Core peak throughput to FP32 scalar peak throughput.
///
/// **Model type**: [`ModelType::PhysicalLaw`] — pure ratio of observables.
///
/// # Validity domain
/// - CC 8.9+ (returns 0.0 for architectures without FP8 support)
///
/// # Interpretation
/// - `0.0` = no FP8 support
/// - `> 8.0` = FP8 path gives massive speedup over FP32 (Ada Lovelace)
/// - `> 20.0` = FP8 is the dominant compute mode (Hopper, Blackwell)
///
/// Useful for deciding whether to quantize to FP8: if ratio is high enough,
/// even moderate accuracy loss is compensated by throughput gain.
pub fn fp8_to_fp32_ratio(obs: &ArchObservables) -> f32 {
    let fp32_flops = obs.fp32_flops_tflops.effective();
    if fp32_flops <= 0.0 {
        return 0.0;
    }
    obs.fp8_flops_tflops.effective() / fp32_flops
}

/// Ratio of INT8 peak throughput to FP32 scalar peak throughput.
///
/// **Model type**: [`ModelType::PhysicalLaw`] — pure ratio of observables.
///
/// # Validity domain
/// - CC 6.1+ for DP4A (Pascal, ~4× FP32)
/// - CC 7.5+ for INT8 Tensor Cores (Turing+, ~10-32× FP32)
/// - Returns `0.0` on architectures without INT8 compute
///
/// # Interpretation
/// - `~4.0` = DP4A only (Pascal) — modest gain
/// - `~10.0` = INT8 Tensor Cores (Turing) — significant
/// - `>30.0` = INT8 is the dominant quantization path (Ampere datacenter, Hopper)
///
/// Useful for W8A8 quantized inference: if ratio is high, INT8 quantization
/// delivers massive throughput even with some accuracy degradation.
pub fn int8_to_fp32_ratio(obs: &ArchObservables) -> f32 {
    let fp32_flops = obs.fp32_flops_tflops.effective();
    if fp32_flops <= 0.0 {
        return 0.0;
    }
    obs.int8_tops.effective() / fp32_flops
}

/// Maximum number of elements that fit in shared memory for a GEMM tile.
///
/// **Model type**: [`ModelType::AnalyticalBound`]
///
/// This is a theoretical **upper bound**, not an achievable tile size.
/// Real kernels use less due to padding, swizzling, double-buffering overhead,
/// and fragment storage requirements.
///
/// # Assumptions
/// - Entire shared memory budget is available for tiling (no other shared mem users)
/// - Two tiles (A and B) are stored simultaneously
/// - No padding or alignment overhead
///
/// # Validity domain
/// - All architectures
/// - Most useful for GEMM/convolution tile size estimation
/// - Does NOT account for: double-buffering, register file pressure on tile loads,
///   warp specialization overhead, or bank conflict avoidance padding
///
/// # Returns
/// Number of elements per tile dimension (assumes square tiles).
pub fn max_tile_elements(obs: &ArchObservables, bytes_per_element: u32) -> u32 {
    if bytes_per_element == 0 {
        return 0;
    }
    // Two tiles (A, B) share the memory
    let bytes_per_tile = obs.shared_mem_per_sm.value / 2;
    let elements_per_tile = bytes_per_tile / bytes_per_element;
    // Square root for square tile dimension
    (elements_per_tile as f32).sqrt() as u32
}

// ── Layer 4: Derived Properties ──────────────────────────────────────────────

/// Computed performance properties derived from observables via documented models.
///
/// Unlike heuristic scores, every field here is computed from a named model
/// function with explicit assumptions. See each model's documentation for details.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DerivedProperties {
    /// Reference occupancy at 32 registers/thread (conventional baseline, not invariant).
    /// From [`theoretical_occupancy_limit`] with `regs_per_thread = 32`.
    ///
    /// **Note**: 32 regs/thread is a conventional reference point. Real kernels
    /// range from <16 (memory-bound) to 200+ (Tensor Core GEMM). Use
    /// [`theoretical_occupancy_limit`] directly for specific register counts.
    pub reference_occupancy_32regs: f32,

    /// Roofline ridge-point for FP32: arithmetic intensity threshold (FLOP/Byte).
    /// From [`roofline_intensity`].
    pub roofline_fp32: f32,

    /// Roofline ridge-point for FP8: arithmetic intensity threshold (FLOP/Byte).
    /// From [`roofline_intensity_fp8`]. `0.0` if FP8 not supported.
    pub roofline_fp8: f32,

    /// Roofline ridge-point for INT8: arithmetic intensity threshold (OP/Byte).
    /// From [`roofline_intensity_int8`]. `0.0` if INT8 not supported.
    pub roofline_int8: f32,

    /// Tensor Core FP16 to FP32 throughput ratio.
    /// From [`tensor_core_dominance`].
    pub tensor_core_dominance: f32,

    /// FP8 to FP32 throughput ratio. `0.0` if FP8 not supported.
    /// From [`fp8_to_fp32_ratio`].
    pub fp8_to_fp32_ratio: f32,

    /// INT8 to FP32 throughput ratio. `0.0` if INT8 not supported.
    /// From [`int8_to_fp32_ratio`].
    pub int8_to_fp32_ratio: f32,

    /// Max GEMM tile side length for FP16 (2 bytes/element) — analytical upper bound.
    /// From [`max_tile_elements`].
    pub max_tile_k_fp16: u32,

    /// Max GEMM tile side length for FP32 (4 bytes/element) — analytical upper bound.
    /// From [`max_tile_elements`].
    pub max_tile_k_fp32: u32,
}

impl DerivedProperties {
    /// Compute all derived properties from observables.
    pub fn from_observables(obs: &ArchObservables) -> Self {
        Self {
            reference_occupancy_32regs: theoretical_occupancy_limit(obs, 32),
            roofline_fp32: roofline_intensity(obs),
            roofline_fp8: roofline_intensity_fp8(obs),
            roofline_int8: roofline_intensity_int8(obs),
            tensor_core_dominance: tensor_core_dominance(obs),
            fp8_to_fp32_ratio: fp8_to_fp32_ratio(obs),
            int8_to_fp32_ratio: int8_to_fp32_ratio(obs),
            max_tile_k_fp16: max_tile_elements(obs, 2),
            max_tile_k_fp32: max_tile_elements(obs, 4),
        }
    }
}

// ── C-defines generation ─────────────────────────────────────────────────────

/// Generate C-preprocessor macros (`#define`) for JIT compilers like NVRTC.
///
/// Emits both physical observables and derived properties as compile-time
/// constants, allowing kernels to adapt at JIT-compile time.
pub fn to_c_defines(arch: &GpuArch) -> String {
    let obs = ArchObservables::from_compute_cap(arch.base);
    let derived = DerivedProperties::from_observables(&obs);

    format!(
        "// Auto-generated by cudaforge (SM {}, ref: {})\n\
         // Layer 2: Physical Observables (source: NvidiaSpec)\n\
         #define CF_REGS_PER_SM {}\n\
         #define CF_SHARED_MEM_PER_SM {}\n\
         #define CF_MAX_THREADS_PER_SM {}\n\
         #define CF_WARP_SIZE {}\n\
         #define CF_L2_BYTES {}\n\
         #define CF_DRAM_BW_GBPS {:.1}f\n\
         #define CF_TC_TFLOPS {:.1}f\n\
         #define CF_FP32_TFLOPS {:.1}f\n\
         #define CF_FP8_TFLOPS {:.1}f\n\
         #define CF_INT8_TOPS {:.1}f\n\
         // Layer 4: Derived Properties (computed from models)\n\
         #define CF_REF_OCCUPANCY_32REGS {:.4}f\n\
         #define CF_ROOFLINE_FP32 {:.4}f\n\
         #define CF_ROOFLINE_FP8 {:.4}f\n\
         #define CF_ROOFLINE_INT8 {:.4}f\n\
         #define CF_TC_DOMINANCE {:.4}f\n\
         #define CF_FP8_TO_FP32_RATIO {:.4}f\n\
         #define CF_INT8_TO_FP32_RATIO {:.4}f\n\
         #define CF_MAX_TILE_FP16 {}\n\
         #define CF_MAX_TILE_FP32 {}\n",
        arch.base,
        obs.reference_gpu,
        obs.registers_per_sm.value,
        obs.shared_mem_per_sm.value,
        obs.max_threads_per_sm.value,
        obs.warp_size,
        obs.l2_bytes.value,
        obs.dram_bandwidth_gbps.value,
        obs.tensor_core_flops_tflops.value,
        obs.fp32_flops_tflops.value,
        obs.fp8_flops_tflops.value,
        obs.int8_tops.value,
        derived.reference_occupancy_32regs,
        derived.roofline_fp32,
        derived.roofline_fp8,
        derived.roofline_int8,
        derived.tensor_core_dominance,
        derived.fp8_to_fp32_ratio,
        derived.int8_to_fp32_ratio,
        derived.max_tile_k_fp16,
        derived.max_tile_k_fp32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Occupancy model ──────────────────────────────────────────────────

    #[test]
    fn test_occupancy_32_regs_ampere() {
        let obs = ArchObservables::from_compute_cap(80);
        // A100: 65536 regs / (32 regs × 32 threads) = 64 warps
        // max warps = 2048/32 = 64 → occupancy = 64/64 = 1.0
        let occ = theoretical_occupancy_limit(&obs, 32);
        assert!((occ - 1.0).abs() < 0.001, "Expected 1.0, got {}", occ);
    }

    #[test]
    fn test_occupancy_128_regs_ampere() {
        let obs = ArchObservables::from_compute_cap(80);
        // A100: 65536 / (128 × 32) = 16 warps, max = 64 → 16/64 = 0.25
        let occ = theoretical_occupancy_limit(&obs, 128);
        assert!((occ - 0.25).abs() < 0.001, "Expected 0.25, got {}", occ);
    }

    #[test]
    fn test_occupancy_32_regs_turing() {
        let obs = ArchObservables::from_compute_cap(75);
        // Turing: 65536 / (32×32) = 64 warps, max = 1024/32 = 32 → 32/32 = 1.0
        let occ = theoretical_occupancy_limit(&obs, 32);
        assert!((occ - 1.0).abs() < 0.001, "Expected 1.0, got {}", occ);
    }

    // ── Roofline models ──────────────────────────────────────────────────

    #[test]
    fn test_roofline_fp32_hopper() {
        let obs = ArchObservables::from_compute_cap(90);
        let ri = roofline_intensity(&obs);
        // H100: 66.9 × 1000 / 3350 ≈ 19.97
        assert!(ri > 19.0 && ri < 21.0, "Expected ~20.0, got {}", ri);
    }

    #[test]
    fn test_roofline_fp8_hopper() {
        let obs = ArchObservables::from_compute_cap(90);
        let ri = roofline_intensity_fp8(&obs);
        // H100: 1979 × 1000 / 3350 ≈ 590.7
        assert!(ri > 580.0 && ri < 600.0, "Expected ~591, got {}", ri);
    }

    #[test]
    fn test_roofline_fp8_pascal_zero() {
        let obs = ArchObservables::from_compute_cap(61);
        let ri = roofline_intensity_fp8(&obs);
        assert!(
            (ri - 0.0).abs() < 0.001,
            "Expected 0.0 for Pascal, got {}",
            ri
        );
    }

    #[test]
    fn test_roofline_int8_ampere() {
        let obs = ArchObservables::from_compute_cap(80);
        let ri = roofline_intensity_int8(&obs);
        // A100: 624 × 1000 / 2039 ≈ 306.0
        assert!(ri > 295.0 && ri < 315.0, "Expected ~306, got {}", ri);
    }

    // ── TC dominance ─────────────────────────────────────────────────────

    #[test]
    fn test_tc_dominance_volta() {
        let obs = ArchObservables::from_compute_cap(70);
        let d = tensor_core_dominance(&obs);
        // V100: 125 / 15.7 ≈ 7.96
        assert!(d > 7.0 && d < 9.0, "Expected ~8.0, got {}", d);
    }

    #[test]
    fn test_tc_dominance_pascal_zero() {
        let obs = ArchObservables::from_compute_cap(61);
        let d = tensor_core_dominance(&obs);
        assert!((d - 0.0).abs() < 0.001, "Expected 0.0, got {}", d);
    }

    // ── FP8 to FP32 ratio ────────────────────────────────────────────────

    #[test]
    fn test_fp8_ratio_hopper() {
        let obs = ArchObservables::from_compute_cap(90);
        let r = fp8_to_fp32_ratio(&obs);
        // H100: 1979 / 66.9 ≈ 29.6
        assert!(r > 28.0 && r < 31.0, "Expected ~29.6, got {}", r);
    }

    #[test]
    fn test_fp8_ratio_ampere_zero() {
        let obs = ArchObservables::from_compute_cap(80);
        let r = fp8_to_fp32_ratio(&obs);
        assert!((r - 0.0).abs() < 0.001, "Expected 0.0 for A100, got {}", r);
    }

    // ── INT8 to FP32 ratio ───────────────────────────────────────────────

    #[test]
    fn test_int8_ratio_ampere() {
        let obs = ArchObservables::from_compute_cap(80);
        let r = int8_to_fp32_ratio(&obs);
        // A100: 624 / 19.5 = 32.0
        assert!(r > 31.0 && r < 33.0, "Expected ~32.0, got {}", r);
    }

    #[test]
    fn test_int8_ratio_pascal_dp4a() {
        let obs = ArchObservables::from_compute_cap(61);
        let r = int8_to_fp32_ratio(&obs);
        // GTX 1080 Ti: 45.2 / 11.3 ≈ 4.0
        assert!(r > 3.5 && r < 4.5, "Expected ~4.0, got {}", r);
    }

    #[test]
    fn test_tile_size_hopper_fp16() {
        let obs = ArchObservables::from_compute_cap(90);
        let tile = max_tile_elements(&obs, 2);
        // 228KB / 2 tiles = 114KB per tile, 114×1024/2 = 58368 elements, √ ≈ 241
        assert!(tile > 200 && tile < 300, "Expected ~241, got {}", tile);
    }

    // ── SKU / reference GPU ──────────────────────────────────────────────

    #[test]
    fn test_reference_gpu_field() {
        let obs = ArchObservables::from_compute_cap(90);
        assert_eq!(obs.reference_gpu, "H100 SXM (GH100)");
        let obs = ArchObservables::from_compute_cap(89);
        assert_eq!(obs.reference_gpu, "RTX 4090 (AD102)");
        let obs = ArchObservables::from_compute_cap(87);
        assert_eq!(obs.reference_gpu, "Jetson AGX Orin 64GB (GA10B)");
    }

    // ── C-defines ────────────────────────────────────────────────────────

    #[test]
    fn test_c_defines_output() {
        let arch = GpuArch::new(90);
        let defines = to_c_defines(&arch);
        assert!(defines.contains("#define CF_REGS_PER_SM 65536"));
        assert!(defines.contains("#define CF_WARP_SIZE 32"));
        assert!(defines.contains("CF_TC_DOMINANCE"));
        assert!(defines.contains("CF_ROOFLINE_FP32"));
        assert!(defines.contains("CF_ROOFLINE_FP8"));
        assert!(defines.contains("CF_ROOFLINE_INT8"));
        assert!(defines.contains("CF_FP8_TO_FP32_RATIO"));
        assert!(defines.contains("H100 SXM"));
    }

    // ── Provenance ───────────────────────────────────────────────────────

    #[test]
    fn test_provenance_tracking() {
        let obs = ArchObservables::from_compute_cap(90);
        assert_eq!(obs.registers_per_sm.source, DataSource::NvidiaSpec);
        assert_eq!(obs.l2_bytes.source, DataSource::NvidiaSpec);
        assert_eq!(obs.registers_per_sm.calibration_factor, None);
    }

    // ── Calibration factor ───────────────────────────────────────────────

    #[test]
    fn test_calibration_factor() {
        let mut m = Measured::from_spec(100.0_f32);
        assert_eq!(m.calibration_factor, None);
        let factor = CalibrationFactor {
            ratio: 0.85,
            meaning: "test".to_string(),
            derived_from: "test".to_string(),
        };
        m.calibration_factor = Some(factor.clone());
        assert_eq!(m.calibration_factor, Some(factor));
        // Effective value = 100.0 * 0.85 = 85.0
        let effective = m.effective();
        assert!((effective - 85.0).abs() < 0.01);
    }
}
