//! Hardware and toolkit capabilities registry for CUDA devices
//!
//! This module provides two capability registries:
//! - [`CAPABILITIES`]: Hardware capabilities gated on GPU compute capability (CC version)
//! - [`TOOLKIT_CAPABILITIES`]: Runtime/toolkit capabilities gated on CUDA toolkit version
//!   (and optionally compute capability)

/// Synchronize a capability flag between Rust (`cargo:rustc-cfg`) and
/// a [`KernelBuilder`](crate::KernelBuilder) (`-DNAME=1` for nvcc).
///
/// When `$value` is true:
///   - Emits `cargo:rustc-cfg=$name` (available via `#[cfg($name)]`)
///   - Adds `-D$NAME=1` to the builder (available via `#ifdef $NAME` in `.cu` files)
///
/// When `$value` is false:
///   - Does nothing (flag is absent in both Rust and C++)
///
/// # Example
/// ```rust,ignore
/// let mut builder = cudaforge::KernelBuilder::new();
/// let arch = cudaforge::detect_compute_cap()?;
/// cudaforge::set_cfg!(builder, "has_wgmma", arch.base >= 90);
/// ```
#[macro_export] 
macro_rules! set_cfg {
    ($builder:expr, $name:expr, $value:expr) => {
        println!("cargo::rustc-check-cfg=cfg({})", $name);
        if $value {
            println!("cargo:rustc-cfg={}", $name);
            $builder = $builder.arg(&format!("-D{}=1", $name.to_uppercase()));
        }
    };
}

/// Apply the same capability flag to two [`KernelBuilder`](crate::KernelBuilder)s simultaneously.
///
/// Useful when building separate static libraries (e.g., one for attention kernels,
/// one for quantization kernels) that both need the same hardware flags.
///
/// # Example
/// ```rust,ignore
/// let mut b_attn = cudaforge::KernelBuilder::new().source_dir("src/attn");
/// let mut b_quant = cudaforge::KernelBuilder::new().source_dir("src/quant");
/// cudaforge::dual_set!(b_attn, b_quant, "has_wgmma", arch.base >= 90);
/// ```
#[macro_export]
macro_rules! dual_set {
    ($b1:expr, $b2:expr, $name:expr, $value:expr) => {
        $crate::set_cfg!($b1, $name, $value);
        $crate::set_cfg!($b2, $name, $value);
    };
}
/// Represents a specific hardware capability for a CUDA architecture version
#[derive(Debug, Clone)]
pub struct Capability {
    /// The name of the capability (used as rustc-cfg and C++ macro)
    pub name: &'static str,
    /// Detailed description of what this capability does and when it was introduced
    pub description: &'static str,
    /// Function to check if the capability is supported on a given compute capability (e.g. 89)
    pub check: fn(&crate::compute_cap::GpuArch) -> bool,
}

/// List of all recognized hardware capabilities and their conditions
pub const CAPABILITIES: &[Capability] = &[
    // ── Kepler (CC 3.0 – 3.7) ── Foundational features ──────────────────────
    Capability {
        name: "has_unified_memory",
        description: "Unified Memory: Basic cudaMallocManaged support — single address space accessible by both CPU and GPU. (CC 3.0+)",
        check: |arch| arch.base >= 30,
    },
    Capability {
        name: "has_dynamic_parallelism",
        description: "Dynamic Parallelism: Launch kernels from within GPU kernels (nested kernel launch). Enables recursive algorithms on GPU. (CC 3.5+)",
        check: |arch| arch.base >= 35,
    },
    Capability {
        name: "has_memory_pools",
        description: "CUDA Memory Pools: Efficient stream-ordered memory allocation/deallocation via cudaMallocAsync/cudaFreeAsync, reducing allocation overhead. (CUDA 11.2+ runtime, CC 3.5+)",
        check: |arch| arch.base >= 35,
    },
    // ── Kepler (CC 3.0 – 3.7) ── Foundational features ──────────────────────
    Capability {
        name: "has_warp_shuffle",
        description: "Warp Shuffle (__shfl_sync): Direct data exchange between threads in a warp. Avoids shared memory for thread-sync within 32 threads. (CC 3.0+)",
        check: |arch| arch.base >= 30,
    },
    Capability {
        name: "has_f16",
        description: "FP16 (half precision): 16-bit float, half the size of f32. Enables faster inference on memory-bound workloads. Available since Maxwell (CC 5.3).",
        check: |arch| arch.base >= 53,
    },
    Capability {
        name: "has_half2_native",
        description: "half2 native: Packed FP16 arithmetic — two FP16 ops in a single instruction, doubling throughput. Only on GP100 (6.0), GP10B (6.2), and Volta+ (7.0+).",
        check: |arch| arch.base == 60 || arch.base == 62 || arch.base >= 70,
    },
    Capability {
        name: "has_pageable_memory_access",
        description: "Pageable memory access: Page Migration Engine — GPU can on-demand page-fault system memory without explicit pinning. Distinct from basic Unified Memory (CC 3.0+). (CC 6.0+)",
        check: |arch| arch.base >= 60,
    },
    Capability {
        name: "has_cooperative_groups",
        description: "Cooperative Groups: Grid-wide synchronization via cooperative_groups::grid_group and cooperative kernel launch. Enables inter-block sync. (CC 6.0+)",
        check: |arch| arch.base >= 60,
    },
    Capability {
        name: "has_int8",
        description: "INT8 arithmetic: Generic 8-bit integer support via DP4A (dot product) or Tensor Cores. (CC 6.1+)",
        check: |arch| arch.base >= 61,
    },
    Capability {
        name: "has_dp4a",
        description: "DP4A: Dot Product of 4x8-bit integers accumulated into 32-bit. (CC 6.1+)",
        check: |arch| arch.base >= 61,
    },
    Capability {
        name: "has_int4",
        description: "INT4 compute: 4-bit integer operations, primarily via Tensor Cores. (CC 7.5+)",
        check: |arch| arch.base >= 75,
    },
    // ── Volta/Turing (CC 7.0 – 7.5) ── Tensor Cores era ─────────────────────
    Capability {
        name: "has_shared_mem_optin",
        description: "Shared Memory Opt-in: Support for requesting more than 48KB of shared memory via cudaFuncSetAttribute. (CC 7.0+)",
        check: |arch| arch.base >= 70,
    },
    Capability {
        name: "has_unified_l1_shared",
        description: "Unified L1/Shared Memory: Hardware-level unification of L1 cache and shared memory, allowing configurable splits. (CC 7.0+)",
        check: |arch| arch.base >= 70,
    },
    Capability {
        name: "has_relaxed_memory_ordering",
        description: "Relaxed Memory Ordering: Hardware support for more efficient relaxed memory consistency models. (CC 7.0+)",
        check: |arch| arch.base >= 70,
    },
    Capability {
        name: "has_wmma",
        description: "WMMA (Warp Matrix Multiply-Accumulate): First-gen Tensor Core API. 16x16x16 matrix ops in hardware. The foundation of fast GEMM. (CC 7.0+)",
        check: |arch| arch.base >= 70,
    },
    Capability {
        name: "has_wmma_f16",
        description: "WMMA FP16: Tensor Core matrix multiply with FP16 inputs/outputs. (CC 7.0+)",
        check: |arch| arch.base >= 70,
    },
    Capability {
        name: "has_independent_thread_scheduling",
        description: "Independent thread scheduling: Each thread has its own program counter. Enables fine-grained synchronization & divergent warp execution. (CC 7.0+)",
        check: |arch| arch.base >= 70,
    },
    Capability {
        name: "has_int8_tensor_cores",
        description: "INT8 Tensor Cores: Hardware wmma instructions for INT8 matrices (m16n8k32/etc.). High throughput for W8A8 quantized inference. (CC 7.5+)",
        check: |arch| arch.base >= 75,
    },
    Capability {
        name: "has_int4_tensor_cores",
        description: "INT4 Tensor Cores: Hardware wmma instructions for INT4 matrices. Extremely high throughput for W4A16 or W4A4 setups. (CC 7.5+)",
        check: |arch| arch.base >= 75,
    },
    Capability {
        name: "has_ldmatrix",
        description: "ldmatrix: Instruction to load data directly from Shared Memory to Tensor Core registers. Crucial for custom MMA kernels. (CC 7.5+)",
        check: |arch| arch.base >= 75,
    },
    // ── Ampere (CC 8.0 – 8.9) ── Precision & efficiency ─────────────────────
    Capability {
        name: "has_bf16_conversions",
        description: "BF16 conversions: Hardware-accelerated FP32↔BF16 conversion instructions (__float2bfloat16 etc.). (CC 8.0+, Ampere)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_large_l2",
        description: "Large L2 cache: Significantly larger L2 cache (up to 40MB+) which affects tiling and data reuse patterns. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_dual_issue_fp32_int",
        description: "Dual-issue FP32+INT: Ability to execute FP32 and INT32 instructions simultaneously in the same SM. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_sparse_tensor_cores",
        description: "Sparse Tensor Cores: 2:4 structured sparsity — prune 50% of weights and get ~2x speedup on matrix ops with no accuracy loss. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_wmma_bf16",
        description: "WMMA BF16: Tensor Core matrix multiply with BF16 inputs. BF16 has the same exponent range as FP32 (better stability than FP16). (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_bf16",
        description: "BF16 native: Full BF16 arithmetic support (add, mul, fma). (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_tf32",
        description: "TF32 (TensorFloat-32): 19-bit format (8-bit exp + 10-bit mantissa + sign). Transparent FP32→TF32 on Tensor Cores, ~8x faster than FP32. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_tf32_tensor_cores",
        description: "TF32 Tensor Cores: Hardware-accelerated TF32 matrix operations on Tensor Cores. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_async_copy",
        description: "Async copy: memcpy_async from global to shared memory without blocking the compute pipeline. Key enabler for software pipelining. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_l2_cache_persistence",
        description: "L2 cache persistence: Pin frequently accessed data (e.g. KV cache) in L2. Reduces DRAM bandwidth pressure during inference. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_mbarrier",
        description: "mbarrier: Hardware-assisted asynchronous barriers. Threads can continue working while waiting for async copies to land. Required for Warp Specialization (producer/consumer kernel patterns). (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_mma_f64",
        description: "DMMA (Double-precision MMA): Tensor Core FP64 matrix multiply-accumulate. High-throughput FP64 GEMM for HPC workloads. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_redux",
        description: "redux.sync: Hardware warp-level reduction instruction (sum, min, max). Avoids manual shuffle-based reductions. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    Capability {
        name: "has_global_red_async",
        description: "Asynchronous Global Reduction: `red.async.global` instruction for asynchronous reduction in global memory. (CC 8.0+)",
        check: |arch| arch.base >= 80,
    },
    // ── Ada Lovelace (CC 8.9) ── FP8 ──────────────────────────────────────────
    Capability {
        name: "has_fp8",
        description: "FP8 data types: __nv_fp8_e4m3 and __nv_fp8_e5m2 storage types and conversion instructions. Enables FP8 quantized storage and data movement. (CC 8.9+, Ada/Hopper)",
        check: |arch| arch.base >= 89,
    },
    Capability {
        name: "has_fp8_tensor_cores",
        description: "FP8 Tensor Cores: Native 8-bit float matrix operations. ~2x throughput vs FP16 with minimal accuracy loss. (CC 8.9+, Ada/Hopper)",
        check: |arch| arch.base >= 89,
    },
    // ── Hopper (CC 9.0) ── Data logistics revolution ─────────────────────────
    Capability {
        name: "has_wgmma",
        description: "WGMMA (Warp Group MMA): Asynchronous matrix multiply-accumulate instruction operating on 128 threads simultaneously for massive GEMM throughput. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_tma",
        description: "TMA (Tensor Memory Accelerator): DMA engine that moves N-dimensional tiles between global↔shared memory. Zero CUDA cores used. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_clusters",
        description: "Thread Block Clusters: Group blocks that can cooperate via distributed shared memory. Enables cross-block synchronization. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_cluster_barrier",
        description: "Cluster Barrier: Hardware-assisted asynchronous barriers across a thread block cluster. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_distributed_shared_memory",
        description: "Distributed Shared Memory: Blocks within a cluster can directly load/store from each other's shared memory. Ideal for wide attention heads. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_l2_multicast",
        description: "L2 multicast: Broadcast data from L2 cache to multiple SMs simultaneously. Eliminates redundant DRAM reads when broadcasting model weights. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_confidential_computing",
        description: "Confidential Computing: Hardware-enforced memory encryption (AES) on VRAM and PCIe bus. Data remains encrypted even during GPU processing. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_wgmma",
        description: "WGMMA (Warp Group MMA): Asynchronous matrix multiplication across a 128-thread warp group using TMA registers directly. Massive throughput gain. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_stmatrix",
        description: "stmatrix: Hardware instruction to store data directly from Tensor Core registers back to Shared Memory seamlessly. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_elect_one",
        description: "elect.sync: Single-thread election instruction — exactly one thread per warp is elected without branching. Enables efficient warp-specialization patterns. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_setmaxnreg",
        description: "setmaxnreg: Dynamic register budget control per warp. Enables producer/consumer warp specialization with asymmetric register allocation. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_tensor_core_async_pipeline",
        description: "Async Tensor Core Pipeline: Hardware support for asynchronous matrix operations on Tensor Cores. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_fp8_wgmma",
        description: "FP8 WGMMA: Asynchronous matrix multiplication for FP8 formats using the Warp Group MMA pipeline. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_fp64_tensor_accum",
        description: "FP64 Tensor Accumulation: Hopper-specific ability to accumulate FP16 matrix operations into FP64. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    Capability {
        name: "has_sparse_tensor_cores_v2",
        description: "Sparse Tensor Cores v2: Enhanced structured sparsity support on Hopper architectures. (CC 9.0+)",
        check: |arch| arch.base >= 90,
    },
    // ── Blackwell (CC 10.0 / 12.0) ── Density & micro-scaling ────────────────
    Capability {
        name: "has_fp4",
        description: "NVFP4: NVIDIA's native 4-bit float format. Requires strict per-tensor or per-group scaling. Doubles throughput vs FP8. (CC 10.0+)",
        check: |arch| arch.base >= 100,
    },
    Capability {
        name: "has_mxfp4",
        description: "MXFP4 (OCP Microscaling): Standardized 4-bit format from the Open Compute Project. Uses micro-scaling with per-group (16/32 elements) scale factors. (CC 10.0+)",
        check: |arch| arch.base >= 100,
    },
    Capability {
        name: "has_mxfp6",
        description: "MXFP6 (OCP Microscaling): Standardized 6-bit format. Balances precision between MXFP4 and MXFP8, with per-group micro-scaling factors. (CC 10.0+)",
        check: |arch| arch.base >= 100,
    },
    Capability {
        name: "has_microscaling",
        description: "Microscaling: General support for the MX format family (MXFP4, MXFP6, MXFP8, MXINT8). Enables per-group scale factors shared across 16/32 elements. (CC 10.0+)",
        check: |arch| arch.base >= 100,
    },
    Capability {
        name: "has_tmem",
        description: "TMEM (Tensor Memory): Dedicated on-chip memory directly coupled to Tensor Cores. Replaces register file (RF) usage for TC operands. (CC 10.0+)",
        check: |arch| arch.base >= 100,
    },
    Capability {
        name: "has_dynamic_sparsity",
        description: "Dynamic Sparsity: Extends the 2:4 structured sparsity (Ampere) with data-dependent sparsity rates that vary per format (FP4/FP8). (CC 10.0+)",
        check: |arch| arch.base >= 100,
    },
    Capability {
        name: "has_tma_v2",
        description: "TMA v2: Enhanced Tensor Memory Accelerator with queuing & async scheduling primitives not available on CC 120. (CC == 100)",
        check: |arch| arch.base == 100,
    },
    Capability {
        name: "has_cluster_multicast",
        description: "Cluster Multicast: HW-accelerated broadcast from global memory to shared memory across all blocks in a cluster. Available on B200, but REMOVED from RTX 5090. (CC == 100)",
        check: |arch| arch.base == 100,
    },
    Capability {
        name: "has_hw_allreduce",
        description: "HW All-Reduce: Hardware-accelerated multi-GPU reduction without using compute cores. Crucial for Tensor Parallelism across 8+ GPUs. (CC == 100)",
        check: |arch| arch.base == 100,
    },
    Capability {
        name: "has_tma_indirect",
        description: "TMA Indirect: Scatter/gather via TMA — the accelerator can follow pointers to load non-contiguous data tiles. (CC == 100)",
        check: |arch| arch.base == 100,
    },
    Capability {
        name: "is_data_center_gpu",
        description: "Data center GPU: Distinguishes CC 100 (B200/B100), CC 90 (H100), and CC 70 (V100) from consumer parts. Guards kernels relying on advanced inter-GPU communication. (CC 70, 90, or 100)",
        check: |arch| arch.base == 100 || arch.base == 90 || arch.base == 70,
    },
    Capability {
        name: "has_amp",
        description: "AMP (AI Management Processor): Dedicated HW scheduler for AI workloads on consumer Blackwell. Manages AI task queuing independently of the CPU. (CC 12.0+)",
        check: |arch| arch.base >= 120,
    },
    // ── Emulation & Software Fallbacks ───────────────────────────────────────
    Capability {
        name: "allow_legacy_fp16",
        description: "Enables FP16 emulation/fallback via FP32 for architectures with limited FP16 throughput (CC 5.3 - 6.x).",
        check: |arch| arch.base >= 53 && arch.base < 70,
    },
    Capability {
        name: "allow_legacy_bf16",
        description: "Enables BF16 emulation/fallback via FP32 for architectures without native BF16 arithmetic (CC 5.3 - 7.x).",
        check: |arch| arch.base >= 53 && arch.base < 80,
    },
    Capability {
        name: "allow_legacy_fp8",
        description: "Enables FP8 emulation/fallback via FP16/FP32 for architectures without native FP8 support (CC 5.3 - 8.6).",
        check: |arch| arch.base >= 53 && arch.base < 89,
    },
];

/// Represents a capability that depends on the CUDA toolkit version
/// (and optionally the GPU compute capability).
#[derive(Debug, Clone)]
pub struct ToolkitCapability {
    /// The name of the capability (used as rustc-cfg and C++ macro)
    pub name: &'static str,
    /// Detailed description of what this capability does
    pub description: &'static str,
    /// Function to check if the capability is available given arch + toolkit version + cuDNN version
    pub check: fn(&crate::compute_cap::GpuArch, &crate::toolkit::CudaVersion, Option<&crate::toolkit::CudnnVersion>) -> bool,
}

/// List of capabilities that depend on the CUDA toolkit version.
///
/// These are runtime/library features that require a minimum toolkit version
/// (and sometimes a minimum compute capability).
pub const TOOLKIT_CAPABILITIES: &[ToolkitCapability] = &[
    // ── Runtime API features (toolkit version required) ──────────────────────
    ToolkitCapability {
        name: "has_nvrtc",
        description: "NVRTC: Runtime compilation of CUDA C++ to PTX. (CUDA 7.0+)",
        check: |_arch, ver, _cudnn| ver.at_least(7, 0),
    },
    ToolkitCapability {
        name: "has_p2p_copy",
        description: "Peer-to-Peer copy: Direct GPU-to-GPU memory transfers. (CUDA 4.0+, CC 3.0+)",
        check: |arch, ver, _cudnn| arch.base >= 30 && ver.at_least(4, 0),
    },
    ToolkitCapability {
        name: "has_cooperative_launch",
        description: "Cooperative Launch: cudaLaunchCooperativeKernel API for grid-wide synchronization. (CUDA 9.0+, CC 6.0+)",
        check: |arch, ver, _cudnn| arch.base >= 60 && ver.at_least(9, 0),
    },
    ToolkitCapability {
        name: "has_cuda_graphs",
        description: "CUDA Graphs: Capture a sequence of GPU operations and replay them with minimal CPU overhead. (CUDA 10.0+, CC 3.0+)",
        check: |arch, ver, _cudnn| arch.base >= 30 && ver.at_least(10, 0),
    },
    ToolkitCapability {
        name: "has_stream_capture_v2",
        description: "Stream Capture v2: Enhanced stream-to-graph capture API with improved handling of dependencies and performance. (CUDA 11.0+)",
        check: |_arch, ver, _cudnn| ver.at_least(11, 0),
    },
    ToolkitCapability {
        name: "has_cublas_lt",
        description: "cuBLAS Lt: Lightweight GEMM API with algorithm selection and fusion. (CUDA 10.1+)",
        check: |_arch, ver, _cudnn| ver.at_least(10, 1),
    },
    ToolkitCapability {
        name: "has_cublaslt_fused_epilogue",
        description: "cuBLAS Lt Fused Epilogue: Support for fusion of GEMM + Bias + Activation. (CUDA 11.0+)",
        check: |_arch, ver, _cudnn| ver.at_least(11, 0),
    },
    ToolkitCapability {
        name: "has_cublaslt_matmul_fp8",
        description: "cuBLAS Lt FP8 GEMM: Optimized 8-bit float matrix multiplication. (CUDA 11.8+, CC 8.9+)",
        check: |arch, ver, _cudnn| arch.base >= 89 && ver.at_least(11, 8),
    },
    ToolkitCapability {
        name: "has_cublaslt_matmul_bf16",
        description: "cuBLAS Lt BF16 GEMM: Native BFloat16 matrix multiplication support. (CC 8.0+)",
        check: |arch, _ver, _cudnn| arch.base >= 80,
    },
    ToolkitCapability {
        name: "has_cublaslt_matmul_int8",
        description: "cuBLAS Lt INT8 GEMM: Native accelerated 8-bit integer GEMM via IMMA. (CC 7.5+)",
        check: |arch, _ver, _cudnn| arch.base >= 75,
    },
    ToolkitCapability {
        name: "has_memory_pools",
        description: "CUDA Memory Pools: Stream-ordered allocation via cudaMallocAsync/cudaFreeAsync. Reduces allocation overhead. (CUDA 11.2+, CC 3.5+)",
        check: |arch, ver, _cudnn| arch.base >= 35 && ver.at_least(11, 2),
    },
    ToolkitCapability {
        name: "has_cusparselt",
        description: "cuSPARSELt: Library for structured sparsity (2:4) on Tensor Cores. (CUDA 11.2+, CC 8.0+)",
        check: |arch, ver, _cudnn| arch.base >= 80 && ver.at_least(11, 2),
    },
    ToolkitCapability {
        name: "has_nvjitlink",
        description: "NVJitLink: Runtime PTX JIT link-time library for linking PTX and objects into a executable. (CUDA 11.4+)",
        check: |_arch, ver, _cudnn| ver.at_least(11, 4),
    },
    ToolkitCapability {
        name: "has_cuda_graph_updates",
        description: "CUDA Graph Updates: Support for updating/reconfiguring graph nodes without full reconstruction. (CUDA 11.4+)",
        check: |_arch, ver, _cudnn| ver.at_least(11, 4),
    },
    ToolkitCapability {
        name: "has_transformer_engine",
        description: "Transformer Engine: Automatic FP8 mixed-precision with dynamic scaling. (CUDA 11.8+, CC 8.9+)",
        check: |arch, ver, _cudnn| arch.base >= 89 && ver.at_least(11, 8),
    },
    ToolkitCapability {
        name: "has_cuda12_features",
        description: "CUDA 12 Features: New APIs including cudaGraphInstantiateWithParams, improved driver API, green contexts. (CUDA 12.0+)",
        check: |_arch, ver, _cudnn| ver.at_least(12, 0),
    },
    ToolkitCapability {
        name: "has_cudnn9",
        description: "cuDNN 9.x compatibility: Graph-based API, fused attention backends, FP8 support. Requires CUDA 12.0+ and cuDNN 9+.",
        check: |_arch, ver, cudnn| ver.at_least(12, 0) && cudnn.map_or(false, |v| v.major >= 9),
    },
    // ── Granular cuDNN features ──────────────────────────────────────────────
    ToolkitCapability {
        name: "has_cudnn_v8",
        description: "cuDNN v8+: Modern cuDNN API with backend support and fused ops.",
        check: |_arch, ver, cudnn| ver.at_least(11, 0) || cudnn.map_or(false, |v| v.major >= 8),
    },
    // ── cuDNN 1D Operations ──────────────────────────────────────────────────
    ToolkitCapability {
        name: "has_cudnn_conv1d_f32",
        description: "cuDNN Conv1D (FP32): Standard 1D convolutions for audio/time-series.",
        check: |_arch, ver, cudnn| ver.at_least(11, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv1d_f16",
        description: "cuDNN Conv1D (FP16): Accelerated 1D convolutions on Pascal/Volta+. (CC 6.0+)",
        check: |arch, ver, cudnn| arch.base >= 60 && ver.at_least(11, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv1d_bf16",
        description: "cuDNN Conv1D (BF16): Accelerated 1D convolutions on Ampere+. (CC 8.0+)",
        check: |arch, ver, cudnn| arch.base >= 80 && ver.at_least(11, 0) && cudnn.is_some(),
    },
    // ── cuDNN 2D Operations ──────────────────────────────────────────────────
    ToolkitCapability {
        name: "has_cudnn_conv2d_f32",
        description: "cuDNN Conv2D (FP32): Standard 2D convolutions via legacy or backend API.",
        check: |_arch, ver, cudnn| ver.at_least(7, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv2d_f16",
        description: "cuDNN Conv2D (FP16): Native half-precision 2D convolutions. (CC 6.0+)",
        check: |arch, ver, cudnn| arch.base >= 60 && ver.at_least(11, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv2d_bf16",
        description: "cuDNN Conv2D (BF16): Native BFloat16 2D convolutions. (CC 8.0+)",
        check: |arch, ver, cudnn| arch.base >= 80 && ver.at_least(11, 0) && cudnn.is_some(),
    },
    // ── cuDNN Transposed Operations (Fractionally Strided) ───────────────────
    ToolkitCapability {
        name: "has_cudnn_conv_transpose1d_f32",
        description: "cuDNN ConvTranspose1d (FP32): Standard transposed 1D convolutions.",
        check: |_arch, ver, cudnn| ver.at_least(7, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv_transpose1d_f16",
        description: "cuDNN ConvTranspose1d (FP16): Native half-precision transposed 1D convolutions. (CC 6.0+)",
        check: |arch, ver, cudnn| arch.base >= 60 && ver.at_least(11, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv_transpose1d_bf16",
        description: "cuDNN ConvTranspose1d (BF16): Native BFloat16 transposed 1D convolutions. (CC 8.0+)",
        check: |arch, ver, cudnn| arch.base >= 80 && ver.at_least(11, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv_transpose2d_f32",
        description: "cuDNN ConvTranspose2d (FP32): Standard transposed 2D convolutions.",
        check: |_arch, ver, cudnn| ver.at_least(7, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv_transpose2d_f16",
        description: "cuDNN ConvTranspose2d (FP16): Native half-precision transposed 2D convolutions. (CC 6.0+)",
        check: |arch, ver, cudnn| arch.base >= 60 && ver.at_least(11, 0) && cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_conv_transpose2d_bf16",
        description: "cuDNN ConvTranspose2d (BF16): Native BFloat16 transposed 2D convolutions. (CC 8.0+)",
        check: |arch, ver, cudnn| arch.base >= 80 && ver.at_least(11, 0) && cudnn.is_some(),
    },
    // ── cuDNN Advanced Features ─────────────────────────────────────────────
    ToolkitCapability {
        name: "has_cudnn_fused_attention",
        description: "cuDNN Fused Attention: Specialized engines for multi-head attention (MHA). (cuDNN 8.9+, CC 8.0+)",
        check: |arch, ver, cudnn| arch.base >= 80 && ver.at_least(11, 8) && cudnn.map_or(false, |v| v.at_least(8, 9)),
    },
    ToolkitCapability {
        name: "has_cudnn_sdpa",
        description: "cuDNN SDPA: Scaled Dot-Product Attention support through the Backend API. (cuDNN 8.9+, CC 8.0+)",
        check: |arch, ver, cudnn| arch.base >= 80 && ver.at_least(11, 8) && cudnn.map_or(false, |v| v.at_least(8, 9)),
    },
    ToolkitCapability {
        name: "has_cudnn_backend_api",
        description: "cuDNN Backend API: Support for operation graphs, fusion, and runtime engine selection. (cuDNN 8.x+)",
        check: |_arch, ver, cudnn| ver.at_least(11, 0) && cudnn.map_or(false, |v| v.major >= 8),
    },
    // ── Interesting cuDNN Features ──────────────────────────────────────────
    ToolkitCapability {
        name: "has_cudnn_graph",
        description: "cuDNN Graph API: Modern API for building and executing operation graphs (cuDNN 9+).",
        check: |_arch, _ver, cudnn| cudnn.map_or(false, |v| v.major >= 9),
    },
    ToolkitCapability {
        name: "has_cudnn_normalization",
        description: "cuDNN Normalization: High-performance BatchNorm, LayerNorm, and RMSNorm kernels.",
        check: |_arch, _ver, cudnn| cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_pooling",
        description: "cuDNN Pooling: Optimized Max/Average pooling operations.",
        check: |_arch, _ver, cudnn| cudnn.is_some(),
    },
    ToolkitCapability {
        name: "has_cudnn_activation",
        description: "cuDNN Activation: Accelerated ReLu, Sigmoid, Tanh, and other point-wise activations.",
        check: |_arch, _ver, cudnn| cudnn.is_some(),
    },
];

/// Emits `cargo::rustc-check-cfg` to declare configurations prior to use
pub fn emit_check_cfgs() {
    // Legacy fallback flags
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_bf16)");
    println!("cargo::rustc-check-cfg=cfg(allow_legacy_fp8)");
    // Architecture identifier
    println!("cargo::rustc-check-cfg=cfg(cuda_arch, values(\"30\", \"35\", \"50\", \"52\", \"53\", \"60\", \"61\", \"62\", \"70\", \"75\", \"80\", \"86\", \"87\", \"89\", \"90\", \"100\", \"120\"))");
    // Threshold flags (inf = "at least this CC")
    println!("cargo::rustc-check-cfg=cfg(inf_cc_100)");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_90)");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_89)");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_80)");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_75)");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_70)");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_61)");
    println!("cargo::rustc-check-cfg=cfg(inf_cc_53)");
    // Toolkit version identifiers
    println!("cargo::rustc-check-cfg=cfg(cuda_toolkit_major, values(\"7\", \"8\", \"9\", \"10\", \"11\", \"12\", \"13\"))");
    println!("cargo::rustc-check-cfg=cfg(cuda_toolkit_at_least_12)");
    println!("cargo::rustc-check-cfg=cfg(cuda_toolkit_at_least_11_8)");
    println!("cargo::rustc-check-cfg=cfg(cuda_toolkit_at_least_11_2)");
    println!("cargo::rustc-check-cfg=cfg(cuda_toolkit_at_least_10)");
    // Hardware capabilities
    for cap in CAPABILITIES {
        println!("cargo::rustc-check-cfg=cfg({})", cap.name);
    }
    // Toolkit capabilities
    for cap in TOOLKIT_CAPABILITIES {
        println!("cargo::rustc-check-cfg=cfg({})", cap.name);
    }
}

/// Emits `cargo:rustc-cfg` bindings based on compute capability logic
pub fn emit_rustc_cfgs(arch: &crate::compute_cap::GpuArch) {
    println!("cargo:rustc-cfg=cuda_arch=\"{}\"", arch.base);
    if arch.base >= 100 {
        println!("cargo:rustc-cfg=inf_cc_100");
    }
    if arch.base >= 90 {
        println!("cargo:rustc-cfg=inf_cc_90");
    }
    if arch.base >= 89 {
        println!("cargo:rustc-cfg=inf_cc_89");
    }
    if arch.base >= 80 {
        println!("cargo:rustc-cfg=inf_cc_80");
    }
    if arch.base >= 75 {
        println!("cargo:rustc-cfg=inf_cc_75");
    }
    if arch.base >= 70 {
        println!("cargo:rustc-cfg=inf_cc_70");
    }
    if arch.base >= 61 {
        println!("cargo:rustc-cfg=inf_cc_61");
    }
    if arch.base >= 53 {
        println!("cargo:rustc-cfg=inf_cc_53");
    }
}

/// Emits `cargo:rustc-cfg` bindings based on CUDA toolkit version
pub fn emit_toolkit_cfgs(version: &crate::toolkit::CudaVersion) {
    println!("cargo:rustc-cfg=cuda_toolkit_major=\"{}\"", version.major);
    if version.at_least(12, 0) {
        println!("cargo:rustc-cfg=cuda_toolkit_at_least_12");
    }
    if version.at_least(11, 8) {
        println!("cargo:rustc-cfg=cuda_toolkit_at_least_11_8");
    }
    if version.at_least(11, 2) {
        println!("cargo:rustc-cfg=cuda_toolkit_at_least_11_2");
    }
    if version.at_least(10, 0) {
        println!("cargo:rustc-cfg=cuda_toolkit_at_least_10");
    }
}

/// Evaluates hardware capabilities without emitting any cargo directives.
///
/// Pure evaluation function usable in both build scripts and CLI tools.
/// Reads `ALLOW_LEGACY` env var to gate legacy fallback capabilities.
pub fn evaluate_hw_capabilities(arch: &crate::compute_cap::GpuArch) -> Vec<(&'static str, bool)> {
    // Parse ALLOW_LEGACY environment variable
    let allow_legacy = std::env::var("ALLOW_LEGACY").unwrap_or_else(|_| "".to_string());
    let allow_legacy = allow_legacy.to_lowercase();
    let permitted: std::collections::HashSet<&str> =
        allow_legacy.split(',').map(|s| s.trim()).collect();
    let allow_all = permitted.contains("all");

    CAPABILITIES
        .iter()
        .map(|cap| {
            let mut enabled = (cap.check)(arch);

            // Filter legacy features if they are enabled by CC
            if enabled && cap.name.starts_with("allow_legacy_") {
                let feature_type = &cap.name["allow_legacy_".len()..];
                if !allow_all && !permitted.contains(feature_type) {
                    enabled = false;
                }
            }
            (cap.name, enabled)
        })
        .collect()
}

/// Evaluates toolkit capabilities without emitting any cargo directives.
///
/// Pure evaluation function usable in both build scripts and CLI tools.
pub fn evaluate_toolkit_capabilities(
    arch: &crate::compute_cap::GpuArch,
    version: &crate::toolkit::CudaVersion,
    cudnn: Option<&crate::toolkit::CudnnVersion>,
) -> Vec<(&'static str, bool)> {
    TOOLKIT_CAPABILITIES
        .iter()
        .map(|cap| {
            let enabled = (cap.check)(arch, version, cudnn);
            (cap.name, enabled)
        })
        .collect()
}

/// Evaluates hardware capabilities AND emits cargo directives (check-cfg, rustc-cfg, rerun-if).
///
/// Use this in `build.rs` scripts. For CLI tools, use [`evaluate_hw_capabilities`] instead.
pub fn get_capabilities_results(arch: &crate::compute_cap::GpuArch) -> Vec<(&'static str, bool)> {
    println!("cargo::rerun-if-env-changed=ALLOW_LEGACY");
    emit_check_cfgs();
    emit_rustc_cfgs(arch);
    let results = evaluate_hw_capabilities(arch);
    for (name, enabled) in &results {
        if *enabled {
            println!("cargo:rustc-cfg={}", name);
        }
    }
    results
}

/// Evaluates toolkit capabilities AND emits cargo directives (rustc-cfg for each enabled cap).
///
/// Use this in `build.rs` scripts. For CLI tools, use [`evaluate_toolkit_capabilities`] instead.
pub fn get_toolkit_capabilities_results(
    arch: &crate::compute_cap::GpuArch,
    version: &crate::toolkit::CudaVersion,
    cudnn: Option<&crate::toolkit::CudnnVersion>,
) -> Vec<(&'static str, bool)> {
    emit_toolkit_cfgs(version);

    let results = evaluate_toolkit_capabilities(arch, version, cudnn);
    for (name, enabled) in &results {
        if *enabled {
            println!("cargo:rustc-cfg={}", name);
        }
    }
    results
}

/// Write `cudaforge_heuristics.rs` to `OUT_DIR`.
///
/// The generated file contains a `pub const CUDAFORGE_NVRTC_MACROS: &str`
/// with all C-preprocessor macros for the detected architecture.
///
/// Consumer usage in application code (not `build.rs`):
/// ```rust,ignore
/// include!(concat!(env!("OUT_DIR"), "/cudaforge_heuristics.rs"));
///
/// fn compile_kernel(kernel_source: &str) {
///     let source = format!("{}\n{}", CUDAFORGE_NVRTC_MACROS, kernel_source);
///     nvrtc_compile(&source); // 0ms detection overhead!
/// }
/// ```
#[cfg(feature = "heuristics")]
pub fn write_heuristics_rs(arch: &crate::compute_cap::GpuArch) -> std::io::Result<()> {
    let out_dir = std::env::var("OUT_DIR")
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::NotFound, e))?;
    let defines = arch.to_c_defines();
    let rs_file = std::path::Path::new(&out_dir).join("cudaforge_heuristics.rs");
    let content = format!(
        "/// Auto-generated NVRTC C-preprocessor macros for the detected GPU architecture.\n\
         pub const CUDAFORGE_NVRTC_MACROS: &str = r#\"\n{}\"#;\n",
        defines
    );
    std::fs::write(rs_file, content)
}

/// Prints a hardware feature summary on the terminal exactly once per build target.
///
/// To also generate `cudaforge_heuristics.rs` for JIT compilers, call
/// [`write_heuristics_rs`] explicitly.
pub fn print_summary_once(
    arch: &crate::compute_cap::GpuArch,
    toolkit_version: Option<&crate::toolkit::CudaVersion>,
    cudnn_version: Option<&crate::toolkit::CudnnVersion>,
) {
    let hw_results = get_capabilities_results(arch);
    let tk_results = toolkit_version.map(|ver| get_toolkit_capabilities_results(arch, ver, cudnn_version));

    if let Ok(out_dir) = std::env::var("OUT_DIR") {
        let marker_file = std::path::Path::new(&out_dir).join(".cudaforge_hw_summary_printed");

        if std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&marker_file)
            .is_ok()
        {
            emit_detailed_feature_summary(
                arch,
                &hw_results,
                toolkit_version,
                tk_results.as_deref(),
            );
        }
    } else {
        emit_detailed_feature_summary(arch, &hw_results, toolkit_version, tk_results.as_deref());
    }
}

/// Prints a detailed summary of enabled hardware and toolkit features.
///
/// Uses the same color scheme as `cargo cudaforge`:
/// - Green `✓` for enabled capabilities
/// - Gray `·` for disabled capabilities
pub fn emit_detailed_feature_summary(
    arch: &crate::compute_cap::GpuArch,
    hw_results: &[(&'static str, bool)],
    toolkit_version: Option<&crate::toolkit::CudaVersion>,
    tk_results: Option<&[(&'static str, bool)]>,
) {
    let suffix = arch.suffix.as_deref().unwrap_or("");
    let ver_str = toolkit_version
        .map(|v| format!(" | CUDA {}", v))
        .unwrap_or_default();
    println!("cargo:warning=\r\x1b[1;36m   ╔══ CUDA Capability Summary ═══════════════════════════╗\x1b[0m");
    println!(
        "cargo:warning=\r\x1b[1;36m   ║\x1b[0m  CC: \x1b[1;33mCC {}{}\x1b[0m{}\x1b[1;36m\x1b[0m",
        arch.base, suffix, ver_str
    );
    println!("cargo:warning=\r\x1b[1;36m   ╚════════════════════════════════════════════════════════╝\x1b[0m");

    // Hardware capabilities
    println!("cargo:warning=\r\x1b[1;32m   ┌─ Hardware (CC) ─────────────────────────────────────────┐\x1b[0m");
    for (name, enabled) in hw_results {
        let cap = CAPABILITIES.iter().find(|c| c.name == *name).unwrap();
        let (mark, color) = if *enabled {
            ("✓", "\x1b[32m")
        } else {
            ("·", "\x1b[90m")
        };
        println!(
            "cargo:warning=\r\x1b[1;32m   │\x1b[0m {color} {mark} {:<30}\x1b[0m {}",
            cap.name, cap.description
        );
    }

    // Toolkit capabilities
    if let Some(results) = tk_results {
        println!("cargo:warning=\r\x1b[1;32m   ├─ Toolkit (Runtime) ──────────────────────────────────┤\x1b[0m");
        for (name, enabled) in results {
            let cap = TOOLKIT_CAPABILITIES
                .iter()
                .find(|c| c.name == *name)
                .unwrap();
            let (mark, color) = if *enabled {
                ("✓", "\x1b[32m")
            } else {
                ("·", "\x1b[90m")
            };
            println!(
                "cargo:warning=\r\x1b[1;32m   │\x1b[0m {color} {mark} {:<30}\x1b[0m {}",
                cap.name, cap.description
            );
        }
    }

    println!("cargo:warning=\r\x1b[1;32m   └─────────────────────────────────────────────────────────┘\x1b[0m");
}

/// The primary NVIDIA library that utilizes a specific hardware affordance.
#[cfg(feature = "heuristics")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TargetLibrary {
    /// Primarily used by cuBLAS (classic GEMM, throughput-oriented)
    CuBLAS,
    /// Primarily used by cuBLASLt (modern GEMM, fusion, fine-grained control)
    CuBLASLt,
    /// Primarily used by cuDNN (convolutions, attention, specialized DL ops)
    CuDNN,
}

/// Represents a high-level capability or "affordance" that libraries like
/// cuDNN or cuBLASLt can exploit on a specific GPU architecture.
#[cfg(feature = "heuristics")]
#[derive(Debug, Clone)]
pub struct LibraryCapability {
    /// The name of the capability
    pub name: &'static str,
    /// Detailed description of what this capability enables for libraries
    pub description: &'static str,
    /// The primary library that benefits from this capability
    pub intended_for: TargetLibrary,
    /// Function to check if the capability is supported based on formal architectural metrics
    pub check: fn(
        &crate::compute_cap::GpuArch,
        &crate::arch_metrics::ArchObservables,
        &crate::arch_metrics::DerivedProperties,
    ) -> bool,
}

/// List of high-level library-facing capabilities (Affordances)
#[cfg(feature = "heuristics")]
pub const LIBRARY_CAPABILITIES: &[LibraryCapability] = &[
    // ── Tensor Core Accumulation Modes ───────────────────────────────────────
    LibraryCapability {
        name: "has_fp64_tensorop",
        description: "FP64 Tensor Cores: Double-precision matrix operations for HPC. (Derived: FP64 TFLOPS > 0 & CC 8.0+)",
        intended_for: TargetLibrary::CuBLAS,
        // Assuming we rely on CC 8.0+ for now since fp64_flops_tflops isn't tracked yet
        check: |arch, _obs, _derived| arch.base >= 80,
    },
    LibraryCapability {
        name: "tensorop_accum_fp32",
        description: "Tensor Core FP32 Accumulation: High-precision accumulation for low-precision inputs. (Derived: TC Dominance > 0)",
        intended_for: TargetLibrary::CuBLAS,
        check: |_arch, _obs, derived| derived.tensor_core_dominance > 0.0,
    },
    LibraryCapability {
        name: "tensorop_accum_fp64",
        description: "Tensor Core FP64 Accumulation: Double-precision accumulation for FP64 Tensor Cores. (Derived: FP64 Tensor Cores present)",
        intended_for: TargetLibrary::CuBLAS,
        check: |arch, _obs, _derived| arch.base >= 80,
    },
    LibraryCapability {
        name: "tensorop_accum_mixed_precision",
        description: "Mixed Precision Accumulation: Native support for mixed output/accumulation types. (Derived: TC Dominance > 0)",
        intended_for: TargetLibrary::CuBLAS,
        check: |_arch, _obs, derived| derived.tensor_core_dominance > 0.0,
    },
    // ── Atomic & Reduction Support ───────────────────────────────────────────
    LibraryCapability {
        name: "has_fp16_atomic_add",
        description: "FP16 Atomic Add: Hardware support for atomic addition on 16-bit floats. (Derived: TC Dominance > 0 & CC 7.0+)",
        intended_for: TargetLibrary::CuBLAS,
        check: |arch, _obs, derived| derived.tensor_core_dominance > 0.0 && arch.base >= 70,
    },
    LibraryCapability {
        name: "has_fp32_atomic_add",
        description: "FP32 Atomic Add: Hardware support for atomic addition on 32-bit floats. (Derived: CC 3.0+)",
        intended_for: TargetLibrary::CuBLAS,
        check: |arch, _obs, _derived| arch.base >= 30,
    },
    LibraryCapability {
        name: "has_fp64_atomic_add",
        description: "FP64 Atomic Add: Hardware support for atomic addition on 64-bit floats. (Derived: CC 6.0+)",
        intended_for: TargetLibrary::CuBLAS,
        check: |arch, _obs, _derived| arch.base >= 60,
    },
    LibraryCapability {
        name: "supports_split_k_atomic_reduction",
        description: "Split-K Reduction: Efficient GEMM reduction using atomics. (Derived: High FP32 Roofline || TC Dominance > 0)",
        intended_for: TargetLibrary::CuBLAS,
        // Split-K relies on accumulating quickly across stream processors;
        // highly requested on V100/Turing where compute is fast but waves are small.
        check: |arch, _obs, derived| arch.base >= 70 || derived.roofline_fp32 > 10.0,
    },
    // ── Layout & Vectorization ───────────────────────────────────────────────
    LibraryCapability {
        name: "supports_vectorized_ldmatrix",
        description: "Vectorized ldmatrix: Fast loading of matrix fragments to registers. (Derived: TC Dominance > 0 & Shared Memory >= 64KB)",
        intended_for: TargetLibrary::CuBLASLt,
        check: |_arch, obs, derived| derived.tensor_core_dominance > 0.0 && obs.shared_mem_per_sm.value >= 64 * 1024,
    },
    LibraryCapability {
        name: "supports_128b_global_ld",
        description: "128-bit Global Loads: Vectorized loads (float4, int4) for optimal bandwidth. (Derived: Always True)",
        intended_for: TargetLibrary::CuBLAS,
        check: |_arch, _obs, _derived| true, // Basically true since Fermi, but we check CC > 30 previously
    },
    LibraryCapability {
        name: "supports_ldmatrix_x4",
        description: "ldmatrix.x4: Maximum throughput matrix load from shared memory. (Derived: Max Tile FP16 >= 128)",
        intended_for: TargetLibrary::CuBLAS,
        check: |_arch, _obs, derived| derived.max_tile_k_fp16 >= 128,
    },
    // ── Fusion & Epilogue ────────────────────────────────────────────────────
    LibraryCapability {
        name: "supports_tensor_epilogue_fusion",
        description: "Epilogue Fusion: Ability to fuse Bias, ReLU, and GELU directly into GEMM. (Derived: High L2 Cache >= 4MB & Registers >= 64k)",
        intended_for: TargetLibrary::CuBLASLt,
        check: |_arch, obs, _derived| obs.l2_bytes.value >= 4 * 1024 * 1024 && obs.registers_per_sm.value >= 65536,
    },
    LibraryCapability {
        name: "supports_gelu_poly_fastpath",
        description: "GELU Polynomial Fastpath: Optimized hardware approximation for GELU activation. (Derived: TC Dominance > 5)",
        intended_for: TargetLibrary::CuBLASLt,
        check: |_arch, _obs, derived| derived.tensor_core_dominance > 5.0,
    },
    // ── Workspace & Autotuning Affordances ───────────────────────────────────
    LibraryCapability {
        name: "supports_large_smem_tiles",
        description: "Large SMEM Tiling: Architecture supports/encourages >100KB shared memory tiles per block. (Derived: SMEM >= 96KB)",
        intended_for: TargetLibrary::CuBLAS,
        check: |_arch, obs, _derived| obs.shared_mem_per_sm.value >= 96 * 1024,
    },
    LibraryCapability {
        name: "supports_high_register_pressure_kernels",
        description: "High Register Pressure: Large register file supports complex, unrolled GEMM kernels. (Derived: Max Threads >= 1536)",
        intended_for: TargetLibrary::CuBLAS,
        // High register pressure kernels are typically used when you have plenty of threads to hide latency,
        // Ampere and above provide 1536-2048 threads and allow larger register allocations per thread before spilling.
        check: |_arch, obs, _derived| obs.max_threads_per_sm.value >= 1536,
    },
];

/// Evaluates high-level library affordances for a given architecture.
#[cfg(feature = "heuristics")]
pub fn evaluate_library_capabilities(
    arch: &crate::compute_cap::GpuArch,
    obs: &crate::arch_metrics::ArchObservables,
    derived: &crate::arch_metrics::DerivedProperties,
) -> Vec<(&'static LibraryCapability, bool)> {
    LIBRARY_CAPABILITIES
        .iter()
        .map(|cap| (cap, (cap.check)(arch, obs, derived)))
        .collect()
}
