# CudaForge

<!-- Uncomment these after publishing to crates.io:
[![Crates.io](https://img.shields.io/crates/v/cudaforge.svg)](https://crates.io/crates/cudaforge)
[![Documentation](https://docs.rs/cudaforge/badge.svg)](https://docs.rs/cudaforge)
![License](https://img.shields.io/crates/l/cudaforge.svg)
-->
![Version](https://img.shields.io/badge/version-0.1.4-blue)
![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green)
![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)

Advanced CUDA kernel builder for Rust with **incremental builds**, **auto-detection**, and **external dependency support**.

## Features

- 🚀 **Incremental Builds** - Only recompile modified kernels using content hashing
- 🔍 **Auto-Detection** - Automatically find CUDA toolkit, nvcc, and compute capability
- 🎯 **Per-Kernel Compute Cap** - Override compute capability for specific kernels by filename
- 📦 **External Dependencies** - Built-in CUTLASS support, or fetch any git repo
- ⚡ **Parallel Compilation** - Configurable thread percentage for parallel builds
- 📁 **Flexible Sources** - Directory, glob, files, or exclude patterns
- 🔬 **Hardware Capabilities** - Auto-detect and expose GPU features as `cfg` flags and C++ macros
- 📊 **Architectural Metrics** - Roofline models, occupancy, precision ratios as C defines for JIT
- 🔮 **Hardware Predictor** - Prescriptive Decision Oracle for runtimes providing shape physics and performance signatures

## Installation

Add to your `Cargo.toml`:

```toml
[build-dependencies]
cudaforge = "0.1"
```

## Quick Start

### Building a Static Library

```rust
// build.rs
use cudaforge::{KernelBuilder, Result};

fn main() -> Result<()> {
    let out_dir = std::env::var("OUT_DIR")?;
    
    KernelBuilder::new()
        .source_dir("src/kernels")
        .arg("-O3")
        .arg("-std=c++17")
        .arg("--use_fast_math")
        .build_lib(format!("{}/libkernels.a", out_dir))?;
    
    println!("cargo:rustc-link-search={}", out_dir);
    println!("cargo:rustc-link-lib=kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    
    Ok(())
}
```

### Building PTX Files

```rust
use cudaforge::KernelBuilder;

fn main() -> cudaforge::Result<()> {
    let output = KernelBuilder::new()
        .source_glob("src/**/*.cu")
        .build_ptx()?;
    
    // Generate Rust file with const declarations
    output.write("src/kernels.rs")?;
    
    Ok(())
}
```

## Compute Capability

### Auto-Detection

CudaForge automatically detects compute capability in this order:
1. `CUDA_COMPUTE_CAP` environment variable (supports "90", "90a", "100a")
2. `nvidia-smi --query-gpu=compute_cap`

For sm_90+ architectures, the 'a' suffix is automatically added for async features.

### Per-Kernel Override

Override compute capability for specific kernels using filename patterns:

```rust
KernelBuilder::new()
    .source_dir("src")
    .with_compute_override("sm90_*.cu", 90)  // Hopper (auto → sm_90a)
    .with_compute_override("sm80_*.cu", 80)  // Ampere (sm_80)
    .with_compute_override_arch("sm100_*.cu", "100a")  // Explicit arch string
    .build_lib("libkernels.a")?;
```

### String-Based Architecture

For explicit control over GPU architecture including suffix:

```rust
KernelBuilder::new()
    .compute_cap_arch("90a")  // Explicit sm_90a
    .source_dir("src")
    .build_lib("libkernels.a")?;
```

### Numeric (Auto-Suffix)

```rust
KernelBuilder::new()
    .compute_cap(90)  // Auto-selects sm_90a for 90+
    .source_dir("src")
    .build_lib("libkernels.a")?;
```

## Source Selection

### Directory (Recursive)

```rust
KernelBuilder::new()
    .source_dir("src/kernels")  // All .cu files recursively
```

### Glob Pattern

```rust
KernelBuilder::new()
    .source_glob("src/**/*.cu")
```

### Specific Files

```rust
KernelBuilder::new()
    .source_files(vec!["src/kernel1.cu", "src/kernel2.cu"])
```

### With Exclusions

```rust
KernelBuilder::new()
    .source_dir("src/kernels")
    .exclude(&["*_test.cu", "deprecated/*", "wip_*.cu"])
```

### Watch Additional Files

Track header files that should trigger rebuilds:

```rust
KernelBuilder::new()
    .source_dir("src/kernels")
    .watch(vec!["src/common.cuh", "src/utils.cuh"])
```

## External Dependencies

### CUTLASS Integration

```rust
KernelBuilder::new()
    .source_dir("src")
    .with_cutlass(Some("7127592069c2fe01b041e174ba4345ef9b279671"))
    .arg("-DUSE_CUTLASS")
    .arg("-std=c++17")
    .build_lib("libkernels.a")?;
```

### Custom Git Repository

Fetch include directories from any git repository:

```rust
KernelBuilder::new()
    .source_dir("src")
    .with_git_dependency(
        "my_lib",                              // Name
        "https://github.com/org/my_lib.git",   // Repository
        "abc123def456",                        // Commit hash
        vec!["include", "src/include"],        // Include paths
        false,                                 // Do not recurse submodules
    )
    .build_lib("libkernels.a")?;
```

### Local Include Paths

```rust
KernelBuilder::new()
    .source_dir("src")
    .include_path("third_party/include")
    .include_path("/opt/cuda/samples/common/inc")
    .build_lib("libkernels.a")?;
```

## Parallel Compilation

### Thread Percentage

Use a percentage of available threads:

```rust
KernelBuilder::new()
    .thread_percentage(0.5)  // 50% of available threads
    .source_dir("src")
    .build_lib("libkernels.a")?;
```

### Maximum Threads

Set an absolute limit:

```rust
KernelBuilder::new()
    .max_threads(8)  // Use at most 8 threads
    .source_dir("src")
    .build_lib("libkernels.a")?;
```

### Environment Variables

- `CUDAFORGE_THREADS` - Override thread count
- `RAYON_NUM_THREADS` - Alternative for compatibility

### Pattern-Based Threading

Enable multiple nvcc threads only for specific files (supports globs):

```rust
KernelBuilder::new()
    .nvcc_thread_patterns(&[
        "gemm_*.cu",       // Matches filename (gemm_vp8.cu)
        "**/special/*.cu", // Matches path
        "flash_api",       // Matches substring
    ], 4)  // Use 4 nvcc threads for matching files
    .build_lib("libkernels.a")?;
```

## CUDA Toolkit Detection

CudaForge automatically locates the CUDA toolkit in this order:

1. `NVCC` environment variable
2. `nvcc` in `PATH`
3. `CUDA_HOME/bin/nvcc`
4. `/usr/local/cuda/bin/nvcc`
5. Common installation paths

### Manual Override

```rust
KernelBuilder::new()
    .cuda_root("/opt/cuda-12.1")
```

## Hardware & Toolkit Capabilities

CudaForge provides powerful, scientifically formalized metrics and capability detection for GPU architectures and CUDA toolkits.

### Build-Time Capability Detection

Enable the `capabilities` feature to emit `cargo:rustc-cfg` flags and `-D` macros for nvcc simultaneously.

```rust
// Cargo.toml
// cudaforge = { version = "0.1", features = ["capabilities"] }

// build.rs
fn main() -> cudaforge::Result<()> {
    let compute_cap = cudaforge::detect_compute_cap()?.base;

    let builder = cudaforge::KernelBuilder::new()
        .compute_cap(compute_cap)
        .source_dir("src/kernels")
        .arg("-std=c++17")
        .arg("-O3")
        .register_capabilities()    // emits Rust cfg flags (e.g. #[cfg(has_wgmma)])
        .emit_defines()             // injects -DHAS_WGMMA=1 etc. into nvcc args
        .print_capabilities_once(); // prints a colored summary to the console

    builder.build_lib(format!("{}/libkernels.a", std::env::var("OUT_DIR")?))?;
    Ok(())
}
```

If `.emit_defines()` is chained, the flags are available in both Rust and C++:

```rust
#[cfg(has_wgmma)] // Only compiled on SM 90+
fn fast_attention() { /* ... */ }
```

```cpp
// In your matrix_multiply.cu — same flags, uppercase:
#ifdef HAS_WGMMA
    // Hopper TMA & WGMMA path
#elif defined(HAS_INT8_TENSOR_CORES)
    // Turing/Ampere m16n8k32
#else
    // Fallback scalar loop
#endif
```

> **Tip**: For custom flags not in the registry, use the `set_cfg!` macro directly:
> `cudaforge::set_cfg!(builder, "my_custom_flag", some_condition);`

You can inspect your system's capabilities using the standalone CLI tool:
```bash
cargo install cudaforge --features capabilities
cudaforge-inspect
```

### Scientific Architectural Metrics

Enable the `heuristics` feature to access mathematically formalized architectural models (Roofline points, occupancy, precision throughput ratios).

These metrics can be exported as C pre-processor macros for JIT compilers (like NVRTC), offering advanced optimization *with zero runtime overhead*:

```rust
// build.rs
fn main() -> cudaforge::Result<()> {
    let arch = cudaforge::detect_compute_cap()?;
    
    // Extract observables (TFLOPS, GB/s) explicitly mapped to a reference SKU
    let obs = cudaforge::ArchObservables::from_compute_cap(arch.base);
    println!("Metrics based on: {}", obs.reference_gpu); // e.g., "H100 SXM (GH100)"
    
    // Write cudaforge_heuristics.rs to OUT_DIR (contains CUDAFORGE_NVRTC_MACROS constant)
    cudaforge::write_heuristics_rs(&arch).expect("Failed to write heuristics");
    
    Ok(())
}
```

```cpp
#define CF_REGS_PER_SM 65536
#define CF_DRAM_BW_GBPS 3350.0f
#define CF_ROOFLINE_FP32 19.9701f
#define CF_ROOFLINE_FP8 590.7463f
#define CF_INT8_TO_FP32_RATIO 29.5815f
#define CF_REF_OCCUPANCY_32REGS 1.0000f
```

You can then inject these directly into your JIT compilation (e.g. using `cuda-driver-sys` or `nvrtc`):

```rust
// In your application code
include!(concat!(env!("OUT_DIR"), "/cudaforge_heuristics.rs"));

// This `include!` macro automatically injects the generated constants:
// pub const CUDAFORGE_NVRTC_MACROS: &str = r#"
// // Auto-generated by cudaforge (SM 90, ref: H100)
// #define CF_REGS_PER_SM 65536
// ...
// "#;

fn compile_kernel(kernel_source: &str) {
    // Prepend the architecture metrics to your kernel source
    let source_with_metrics = format!("{}\n{}", CUDAFORGE_NVRTC_MACROS, kernel_source);
    
    // Compile with NVRTC — 100% static metrics, 0ms runtime detection overhead!
    nvrtc_compile(&source_with_metrics); 
}
```

### Prescriptive Decision Oracle (Hardware Predictor)

CudaForge includes a `"compiler-grade"` cognitive predictor that acts as a Prescriptive Decision Oracle for graph compilers and runtimes. By passing the `ProblemShape` (e.g., GEMM M, N, K), the oracle provides executable intents to bypass brute-force autotuning:

- **Shape Physics**: Categorizes workloads (e.g., `LargeSquare`, `KDominant`, `TallSkinnyM`) and projects them into a **Shape Manifold** to enable clustering and topological reuse of learned parameters.
- **Probabilistic Scheduler Model**: Replaces scalar heuristics with a formal causal model of the GPU pipeline, evaluating `issue_rate`, `warp_ready_prob`, and `pipe_pressure` based on instruction dependencies and structural hazards.
- **Hierarchical Memory Model**: Tracks hit probabilities across the cache hierarchy (`P(L1)`, `P(L2)`, `P(DRAM)`) to predict traffic-induced stalls and bandwidth bottlenecks with microarchitectural precision.
- **Calibrated Uncertainty Theory**: Provides statistically rigorous bounds (`σ_runtime`) by separating **Epistemic** (lack of data), **Aleatoric** (task variance), and **Transfer** (cross-arch drift) uncertainty.
- **Verification Targets**: Emits actionable assertions (`expected_issue_utilization`, `expected_stall_reason`) that runtimes can use to self-validate against Nsight Compute.
- **Universal Telemetry (`GpuEEGLog`)**: A standardized JSON schema acting as the bridge between compilers (MLIR, Luminal), predictors, and profilers, tracking the full lifecycle from stimulus to Bayesian posterior update.

```bash
# Output the predictor intent as machine-consumable JSON
cargo run --bin cudaforge-inspect --features heuristics -- --json --m 2048 --n 2048 --k 2048
```

## Incremental Builds

Incremental builds are enabled by default. CudaForge tracks:
- File content hashes (SHA-256)
- Compute capability used
- Compiler arguments

To disable:

```rust
KernelBuilder::new()
    .no_incremental()
```

## Full Example

```rust
// Cargo.toml
// cudaforge = { version = "0.1", features = ["capabilities", "heuristics"] }

// build.rs
use cudaforge::{KernelBuilder, Result};

fn main() -> Result<()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    
    let out_dir = std::env::var("OUT_DIR")?;
    let arch = cudaforge::detect_compute_cap()?;
    
    // Build with full feature set
    KernelBuilder::new()
        .compute_cap(arch.base)
        
        // Source selection
        .source_dir("src/kernels")
        .exclude(&["*_test.cu", "deprecated/*"])
        .watch(vec!["src/common.cuh"])
        
        // Per-kernel compute cap
        .with_compute_override("sm90_*.cu", 90)
        .with_compute_override("sm80_*.cu", 80)
        
        // External dependencies
        .with_cutlass(None)
        
        // Compiler options
        .arg("--expt-relaxed-constexpr")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        
        // Hardware capabilities → Rust #[cfg(has_*)] + C++ -DHAS_*=1
        .register_capabilities()
        .emit_defines()
        .print_capabilities_once()
        
        // Parallel build
        .thread_percentage(0.5)
        
        // Build
        .build_lib(format!("{}/libkernels.a", out_dir))?;
    
    // Generate architectural metrics for JIT compilers (NVRTC)
    cudaforge::write_heuristics_rs(&arch)?;
    
    println!("cargo:rustc-link-search={}", out_dir);
    println!("cargo:rustc-link-lib=kernels");
    println!("cargo:rustc-link-lib=dylib=cudart");
    
    Ok(())
}
```

## Multiple Builders

Use multiple builders in sequence for different output types or configurations:

```rust
use cudaforge::KernelBuilder;

fn main() -> cudaforge::Result<()> {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    
    // Builder 1: Static library for main kernels
    KernelBuilder::new()
        .source_dir("src/kernels")
        .exclude(&["*_ptx.cu"])
        .compute_cap(90)
        .arg("-O3")
        .build_lib(format!("{}/libkernels.a", out_dir))?;
    
    // Builder 2: PTX files for runtime compilation
    let ptx_output = KernelBuilder::new()
        .source_glob("src/ptx_kernels/*.cu")
        .compute_cap(80)
        .build_ptx()?;
    
    ptx_output.write("src/kernels.rs")?;
    
    // Builder 3: Separate lib with CUTLASS
    KernelBuilder::new()
        .source_dir("src/cutlass_kernels")
        .with_cutlass(None)
        .compute_cap_arch("100a")
        .build_lib(format!("{}/libcutlass.a", out_dir))?;
    
    println!("cargo:rustc-link-search={}", out_dir);
    println!("cargo:rustc-link-lib=kernels");
    println!("cargo:rustc-link-lib=cutlass");
    
    Ok(())
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `CUDA_COMPUTE_CAP` | Default compute capability (e.g., `80`, `90`) |
| `NVCC` | Path to nvcc binary |
| `CUDA_HOME` | CUDA installation root |
| `NVCC_CCBIN` | C++ compiler for nvcc |
| `CUDAFORGE_THREADS` | Override thread count |
| `ALLOW_LEGACY` | Enable legacy dtype fallbacks (e.g., `"bf16,fp8"` or `"all"`) |

## Docker Builds

> [!IMPORTANT]
> **GPU is NOT accessible during `docker build`** — only during `docker run --gpus all`.

When building CUDA kernels inside a Dockerfile, `nvidia-smi` cannot be used to auto-detect compute capability. You must explicitly set `CUDA_COMPUTE_CAP`:

### Dockerfile Example

```dockerfile
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Set compute capability for the build
ARG CUDA_COMPUTE_CAP=90
ENV CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP}

# Build with explicit compute cap
WORKDIR /app
COPY . .
RUN cargo build --release
```

Build for different GPU architectures:

```bash
# Build for Hopper (sm_90)
docker build --build-arg CUDA_COMPUTE_CAP=90 -t myapp:hopper .

# Build for Blackwell (sm_100)
docker build --build-arg CUDA_COMPUTE_CAP=100 -t myapp:blackwell .

# Build for Ampere (sm_80)
docker build --build-arg CUDA_COMPUTE_CAP=80 -t myapp:ampere .
```

### Fail-Fast Mode

For CI/Docker builds, use `require_explicit_compute_cap()` to fail immediately if compute capability is not set:

```rust
KernelBuilder::new()
    .require_explicit_compute_cap()?  // Fails fast if CUDA_COMPUTE_CAP not set
    .source_dir("src/kernels")
    .build_lib("libkernels.a")?;
```

## Migration from bindgen_cuda

| Old API | New API |
|---------|---------|
| `Builder::default()` | `KernelBuilder::new()` |
| `.kernel_paths(vec![...])` | `.source_files(vec![...])` |
| `.kernel_paths_glob("...")` | `.source_glob("...")` |
| `.include_paths(vec![...])` | `.include_path("...")` |
| `Bindings` | `PtxOutput` |

Backward compatibility aliases are available:
- `cudaforge::Builder` → `KernelBuilder`
- `cudaforge::Bindings` → `PtxOutput`

## License

MIT OR Apache-2.0
