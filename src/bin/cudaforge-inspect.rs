//! cargo subcommand: `cargo cudaforge`
//!
//! Usage:
//!   cargo install cudaforge --features capabilities
//!   cargo cudaforge          # prints detected capabilities
//!   cargo cudaforge --help

fn main() {
    // Skip "cudaforge" arg inserted by cargo when invoked as `cargo cudaforge`
    let args: Vec<String> = std::env::args().collect();
    let args_slice: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

    match args_slice.as_slice() {
        [_, "cudaforge", "--help"] | [_, "--help"] | [_, "cudaforge", "-h"] | [_, "-h"] => {
            print_help();
        }
        _ => {
            let mut m = 0;
            let mut n = 0;
            let mut k = 0;
            let mut dtype = cudaforge::DType::Fp16;

            let mut calibrate = false;
            let mut json_mode = false;

            let mut i = 1;
            while i < args.len() {
                match args[i].as_str() {
                    "--m" if i + 1 < args.len() => {
                        m = args[i + 1].parse().unwrap_or(0);
                        i += 2;
                    }
                    "--n" if i + 1 < args.len() => {
                        n = args[i + 1].parse().unwrap_or(0);
                        i += 2;
                    }
                    "--k" if i + 1 < args.len() => {
                        k = args[i + 1].parse().unwrap_or(0);
                        i += 2;
                    }
                    "--dtype" if i + 1 < args.len() => {
                        dtype = match args[i + 1].to_lowercase().as_str() {
                            "fp32" => cudaforge::DType::Fp32,
                            "fp16" => cudaforge::DType::Fp16,
                            "bf16" => cudaforge::DType::Bf16,
                            "fp8" => cudaforge::DType::Fp8,
                            "int8" => cudaforge::DType::Int8,
                            _ => cudaforge::DType::Fp16,
                        };
                        i += 2;
                    }
                    "--calibrate" => {
                        calibrate = true;
                        i += 1;
                    }
                    "--json" => {
                        json_mode = true;
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            print_capabilities(m, n, k, dtype, calibrate, json_mode);
        }
    }
}

fn print_help() {
    eprintln!("cargo-cudaforge — CUDA capability inspector");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    cargo cudaforge [OPTIONS]");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("    --m <val>       GEMM M dimension");
    eprintln!("    --n <val>       GEMM N dimension");
    eprintln!("    --k <val>       GEMM K dimension");
    eprintln!("    --dtype <type>  Data type (fp32, fp16, bf16, fp8, int8)");
    eprintln!("    --calibrate     Run micro-benchmarks to calibrate model metrics");
    eprintln!("    --json          Emit machine-readable kernel intent in JSON");
    eprintln!();
    eprintln!("Detects the installed GPU (via nvidia-smi or CUDA_COMPUTE_CAP)");
    eprintln!("and CUDA toolkit (via nvcc), then prints which hardware and");
    eprintln!("toolkit capabilities are available.");
}

fn print_capabilities(
    m: usize,
    n: usize,
    k: usize,
    dtype: cudaforge::DType,
    calibrate: bool,
    json_mode: bool,
) {
    // ── Detect GPU ──────────────────────────────────────────────────────────
    let arch = match cudaforge::detect_compute_cap() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("❌ GPU detection failed: {}", e);
            eprintln!("   Set CUDA_COMPUTE_CAP=<cc> (e.g. 89) to override.");
            std::process::exit(1);
        }
    };

    // ── Detect Toolkit ──────────────────────────────────────────────────────
    let toolkit = cudaforge::CudaToolkit::detect().ok();
    let cuda_version = toolkit.as_ref().and_then(|t| t.parsed_version.clone());

    // ── Header ──────────────────────────────────────────────────────────────
    if !json_mode {
        let suffix = arch.suffix.as_deref().unwrap_or("");
        let ver_str = cuda_version
            .as_ref()
            .map(|v| format!(" | CUDA {}", v))
            .unwrap_or_default();
        let nvcc_str = toolkit
            .as_ref()
            .map(|t| format!(" ({})", t.nvcc_path.display()))
            .unwrap_or_default();

        println!();
        println!("  \x1b[1;36m╔══ CUDA Capability Report ═══════════════════════════════╗\x1b[0m");
        println!("  \x1b[1;36m║\x1b[0m  GPU Compute Capability: \x1b[1;33mCC {}{}\x1b[0m{:<20}\x1b[1;36m║\x1b[0m",
            arch.base, suffix, ver_str);
        if !nvcc_str.is_empty() {
            println!(
                "  \x1b[1;36m║\x1b[0m  Toolkit: {:<40} \x1b[1;36m║\x1b[0m",
                nvcc_str
            );
        }
        println!("  \x1b[1;36m╚════════════════════════════════════════════════════════╝\x1b[0m");
        println!();
    }

    // ── Evaluate capabilities (reuse library logic, no cargo: output) ─────
    let hw_results = cudaforge::evaluate_hw_capabilities(&arch);
    let cudnn_version = toolkit.as_ref().and_then(|t| t.cudnn_version.as_ref());
    let tk_results = cuda_version
        .as_ref()
        .map(|ver| cudaforge::evaluate_toolkit_capabilities(&arch, ver, cudnn_version));

    #[cfg(feature = "heuristics")]
    let (obs, derived) = {
        let mut obs = cudaforge::ArchObservables::from_compute_cap(arch.base);
        if calibrate {
            if let Some(tk) = toolkit.as_ref() {
                if !json_mode {
                    println!("  \x1b[1;33m· Running calibration probes...\x1b[0m");
                }
                let engine = cudaforge::CalibrationEngine::new(
                    tk.clone(),
                    std::path::PathBuf::from("target/calibration"),
                );
                let probes: Vec<Box<dyn cudaforge::CalibrationProbe>> = vec![
                    Box::new(cudaforge::BandwidthProbe),
                    Box::new(cudaforge::ComputeProbe),
                ];
                match engine.run_probes(&arch, &probes) {
                    Ok(facts) => {
                        if !json_mode {
                            println!(
                                "  \x1b[1;32m✓ Collected {} calibration facts.\x1b[0m",
                                facts.len()
                            );
                        }
                        obs.calibrate(&facts);
                    }
                    Err(e) => {
                        if !json_mode {
                            eprintln!("  \x1b[1;31m❌ Calibration failed: {}\x1b[0m", e)
                        }
                    }
                }
            } else {
                if !json_mode {
                    eprintln!("  \x1b[1;31m❌ CUDA Toolkit required for calibration.\x1b[0m")
                };
            }
        }
        let derived = cudaforge::DerivedProperties::from_observables(&obs);
        (obs, derived)
    };
    #[cfg(feature = "heuristics")]
    let obs_and_derived = (obs.clone(), derived.clone());

    if !json_mode {
        // ── Hardware capabilities ───────────────────────────────────────────────
        println!("  \x1b[1;32m┌─ Hardware (CC) ─────────────────────────────────────────┐\x1b[0m");
        for (name, enabled) in &hw_results {
            let cap = cudaforge::CAPABILITIES
                .iter()
                .find(|c| c.name == *name)
                .unwrap();
            let (mark, color) = if *enabled {
                ("✓", "\x1b[32m")
            } else {
                ("·", "\x1b[90m")
            };
            println!(
                "  \x1b[1;32m│\x1b[0m {color} {mark} {:<30}\x1b[0m {}",
                cap.name, cap.description
            );
        }
    }

    if !json_mode {
        // ── Toolkit capabilities ────────────────────────────────────────────────
        if let Some(ref results) = tk_results {
            // 1. CUDA Section
            println!(
                "  \x1b[1;32m├─ Toolkit (CUDA {}) ──────────────────────────────────┤\x1b[0m",
                cuda_version.as_ref().unwrap()
            );
            for (name, enabled) in results
                .iter()
                .filter(|(n, _)| !n.starts_with("has_cudnn") && !n.contains("cublas"))
            {
                let cap = cudaforge::TOOLKIT_CAPABILITIES
                    .iter()
                    .find(|c| c.name == *name)
                    .unwrap();
                let (mark, color) = if *enabled {
                    ("✓", "\x1b[32m")
                } else {
                    ("·", "\x1b[90m")
                };
                println!(
                    "  \x1b[1;32m│\x1b[0m {color} {mark} {:<30}\x1b[0m {}",
                    cap.name, cap.description
                );
            }

            #[cfg(feature = "heuristics")]
            let (cublas_caps, cublaslt_caps, cudnn_caps) = {
                // Evaluate library capabilities once and group them
                let (_obs, derived) = &obs_and_derived;
                let lib_results = cudaforge::evaluate_library_capabilities(&arch, _obs, derived);

                let mut cublas_caps = Vec::new();
                let mut cublaslt_caps = Vec::new();
                let mut cudnn_caps = Vec::new();

                for (cap, enabled) in lib_results {
                    match cap.intended_for {
                        cudaforge::TargetLibrary::CuBLAS => cublas_caps.push((cap, enabled)),
                        cudaforge::TargetLibrary::CuBLASLt => cublaslt_caps.push((cap, enabled)),
                        cudaforge::TargetLibrary::CuDNN => cudnn_caps.push((cap, enabled)),
                    }
                }
                (cublas_caps, cublaslt_caps, cudnn_caps)
            };

            #[cfg(feature = "heuristics")]
            let print_lib_caps = |caps: &Vec<(&cudaforge::LibraryCapability, bool)>| {
                for (cap, enabled) in caps {
                    let (mark, color) = if *enabled {
                        ("✓", "\x1b[32m")
                    } else {
                        ("·", "\x1b[90m")
                    };
                    println!(
                        "  \x1b[1;32m│\x1b[0m {color} {mark} {:<30}\x1b[0m {}",
                        cap.name, cap.description
                    );
                }
            };

            if results.iter().any(|(n, _)| *n == "has_cublas_lt") {
                // Indicates cuBLAS presence in toolkit detection
                println!(
                    "  \x1b[1;32m├─ Toolkit (CUBLAS) ───────────────────────────────────┤\x1b[0m"
                );
                #[cfg(feature = "heuristics")]
                print_lib_caps(&cublas_caps);

                println!(
                    "  \x1b[1;32m├─ Toolkit (cuBLASLt) ──────────────────────────────────┤\x1b[0m"
                );
                for (name, enabled) in results
                    .iter()
                    .filter(|(n, _)| n.starts_with("has_cublaslt") || *n == "has_cublas_lt")
                {
                    let cap = cudaforge::TOOLKIT_CAPABILITIES
                        .iter()
                        .find(|c| c.name == *name)
                        .unwrap();
                    let (mark, color) = if *enabled {
                        ("✓", "\x1b[32m")
                    } else {
                        ("·", "\x1b[90m")
                    };
                    println!(
                        "  \x1b[1;32m│\x1b[0m {color} {mark} {:<30}\x1b[0m {}",
                        cap.name, cap.description
                    );
                }
                #[cfg(feature = "heuristics")]
                print_lib_caps(&cublaslt_caps);
            }

            let cudnn_version = toolkit.as_ref().and_then(|t| t.cudnn_version.as_ref());
            if cudnn_version.is_some()
                || results
                    .iter()
                    .any(|(n, e)| n.starts_with("has_cudnn") && *e)
            {
                let cudnn_ver_str = cudnn_version
                    .map(|v| format!("{}", v))
                    .unwrap_or_else(|| "?".into());
                println!(
                    "  \x1b[1;32m├─ Toolkit (CUDNN {}) ───────────────────────────────┤\x1b[0m",
                    cudnn_ver_str
                );
                for (name, enabled) in results.iter().filter(|(n, _)| n.starts_with("has_cudnn")) {
                    let cap = cudaforge::TOOLKIT_CAPABILITIES
                        .iter()
                        .find(|c| c.name == *name)
                        .unwrap();
                    let (mark, color) = if *enabled {
                        ("✓", "\x1b[32m")
                    } else {
                        ("·", "\x1b[90m")
                    };
                    println!(
                        "  \x1b[1;32m│\x1b[0m {color} {mark} {:<30}\x1b[0m {}",
                        cap.name, cap.description
                    );
                }
                #[cfg(feature = "heuristics")]
                print_lib_caps(&cudnn_caps);
            }
        }
    }

    #[cfg(feature = "heuristics")]
    {
        let (_obs, derived) = &obs_and_derived;
        let shape = cudaforge::ProblemShape { m, n, k };
        let predictor_engine = cudaforge::predictor::HardwarePredictor::new(
            arch.clone(),
            cudaforge::predictor::AnalyticalModel::default(),
        );
        let predictor = predictor_engine
            .evaluate(&shape, dtype, None)
            .expect("Prediction failed");

        let status_line = if predictor.hardware_observed.is_calibrated {
            "\x1b[1;32m├─ Execution Profile & Architectural Regime (CALIBRATED) ──┤\x1b[0m"
        } else {
            "\x1b[1;32m├─ Execution Profile & Architectural Regime ────────────┤\x1b[0m"
        };

        if json_mode {
            println!("{}", serde_json::to_string_pretty(&predictor).unwrap());
            return;
        }

        println!("  {}", status_line);
        if m > 0 || n > 0 || k > 0 {
            println!(
                "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}x{}x{} ({})",
                "Target Problem Context:", m, n, k, dtype
            );
        }
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "Schema version:", predictor.schema_version
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "Inference role:", predictor.intended_usage.role
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m \x1b[1;33m{:.1}%\x1b[0m",
            "Inference confidence:",
            predictor.inference.confidence_breakdown.regime * 100.0
        );

        // Phase 16: Numeric Feasibility
        let f = &predictor.workload_observed.feasibility;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "Numeric feasibility mode:", f.execution_mode
        );
        if !f.is_native {
            println!(
                "  \x1b[1;32m│\x1b[0m \x1b[90m↳ effective compute dtype: \x1b[33m{}\x1b[0m",
                f.effective_compute_dtype
            );
            if let Some(reason) = &f.penalty_reason {
                println!(
                    "  \x1b[1;32m│\x1b[0m \x1b[90m↳ penalty: \x1b[31m{}\x1b[0m",
                    reason
                );
            }
        }
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "Architectural class:",
            predictor
                .derived_metrics
                .architectural_model
                .profile
                .arch_class
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {:?}",
            "Compute pipeline:",
            predictor
                .derived_metrics
                .architectural_model
                .profile
                .compute_pipeline
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {:?}",
            "Memory pipeline:",
            predictor
                .derived_metrics
                .architectural_model
                .profile
                .memory_pipeline
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {:?}",
            "Parallelism model:",
            predictor
                .derived_metrics
                .architectural_model
                .profile
                .parallelism_model
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {:?}",
            "Numeric engine class:",
            predictor
                .derived_metrics
                .architectural_model
                .profile
                .numeric_engine
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m \x1b[1;32m{}\x1b[0m",
            "Architectural kernel prior:",
            predictor
                .derived_metrics
                .architectural_model
                .profile
                .preferred_kernel_family
        );

        let print_calibrated = |label: &str, val: &cudaforge::Measured<f32>, unit: &str| {
            if let Some(c) = &val.calibration_factor {
                println!("  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m \x1b[1;32m{:.1} {} (calibrated, spec: {:.1})\x1b[0m", label, val.effective(), unit, val.value);
                println!(
                    "  \x1b[1;32m│\x1b[0m \x1b[90m↳ calibration meaning: \x1b[33m{}\x1b[0m",
                    c.meaning
                );
            } else {
                println!(
                    "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {:.1} {}",
                    label, val.value, unit
                );
            }
        };

        println!("  \x1b[1;32m├─ Architectural Ratios & Limits ───────────────────────┤\x1b[0m");
        let obs = &predictor.hardware_observed.observables;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Representative SKU for constants:", obs.reference_gpu
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {} bytes",
            "Bytes per register file (per SM):",
            obs.registers_per_sm.value * 4
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {} KB",
            "Shared Memory per SM:",
            obs.shared_mem_per_sm.value / 1024
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {} MB",
            "L2 Cache:",
            obs.l2_bytes.value / 1024 / 1024
        );
        print_calibrated("DRAM Bandwidth:", &obs.dram_bandwidth_gbps, "GB/s");
        print_calibrated("Peak FP32 throughput:", &obs.fp32_flops_tflops, "TFLOPS");

        println!("  \x1b[1;32m├─ Tensor Execution Mode Summary ───────────────────────┤\x1b[0m");
        let tc_support = if obs.fp8_flops_tflops.value > 0.0 {
            "FP8 / FP16"
        } else if obs.tensor_core_flops_tflops.value > 0.0 {
            "FP16"
        } else {
            "none"
        };
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "Tensor Core Support:", tc_support
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {:.1}x",
            "Tensor Throughput Ratio (vs FP32):", derived.tensor_core_dominance
        );
        let pref_gemm = if derived.tensor_core_dominance > 0.0 {
            "Tensor Cores"
        } else {
            "CUDA Cores"
        };
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "Recommended GEMM engine:", pref_gemm
        );

        println!("  \x1b[1;32m├─ Modeling & Limits ───────────────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {:.1} FLOP/B",
            "Ridge Point (FP32):", derived.roofline_fp32
        );
        if obs.tensor_core_flops_tflops.value > 0.0 {
            let ridge_tc =
                obs.tensor_core_flops_tflops.value * 1000.0 / obs.dram_bandwidth_gbps.value;
            println!(
                "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {:.1} FLOP/B",
                "Ridge Point (Tensor FP16):", ridge_tc
            );
        }
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {} / {}",
            "Register-limited tile upper bound (FP16/FP32):",
            derived.max_tile_k_fp16,
            derived.max_tile_k_fp32
        );

        println!("  \x1b[1;32m├─ Posterior Execution Regime ──────────────────────────┤\x1b[0m");
        let er = &predictor.inference.execution_regime;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m \x1b[1;31m{:.1}%\x1b[0m",
            "P(Compute Bound):",
            er.compute_bound * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m \x1b[1;33m{:.1}%\x1b[0m",
            "P(Balanced):",
            er.balanced * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m \x1b[1;34m{:.1}%\x1b[0m",
            "P(Memory Bound):",
            er.memory_bound * 100.0
        );

        println!("  \x1b[1;32m├─ Probabilistic Performance Posteriors ────────────────┤\x1b[0m");
        let pp = &predictor.inference.performance_posteriors;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m \x1b[1;32m{:.0} μs (σ={:.2})\x1b[0m",
            "Estimated runtime:", pp.runtime_us.mean, pp.runtime_us.stddev
        );
        println!("  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m \x1b[1;32m{:.1}% (μ={:.2}, σ={:.3})\x1b[0m", "Expected FLOP utilization:", pp.flop_utilization.mean * 100.0, pp.flop_utilization.mean, pp.flop_utilization.stddev);
        println!("  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m \x1b[1;32m{:.1}% (μ={:.2}, σ={:.3})\x1b[0m", "Expected DRAM utilization:", pp.bw_utilization.mean * 100.0, pp.bw_utilization.mean, pp.bw_utilization.stddev);

        println!("  \x1b[1;32m├─ Resource Pressure Models ────────────────────────────┤\x1b[0m");
        let rpm = &predictor.inference.resource_pressure_models;
        println!("  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m \x1b[1;33m{:.1}% (μ={:.2}, σ={:.3})\x1b[0m", "Inferred scheduler pressure:", rpm.scheduler_pressure.mean * 100.0, rpm.scheduler_pressure.mean, rpm.scheduler_pressure.stddev);
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m \x1b[1;31m{:.1}%\x1b[0m",
            "Risk of register spilling:",
            rpm.spill_risk.mean * 100.0
        );

        println!("  \x1b[1;32m├─ Formal Memory Model (Posteriors) ────────────────────┤\x1b[0m");
        let mm = &predictor.inference.memory_model;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[34m· {:<46}\x1b[0m {:.1}%",
            "P(Hit L1/SMEM):",
            mm.l1_hit_rate.mean * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[34m· {:<46}\x1b[0m {:.1}%",
            "P(Hit L2):",
            mm.l2_hit_rate.mean * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[34m· {:<46}\x1b[0m {:.1} MB",
            "Estimated DRAM Traffic:",
            mm.dram_traffic_bytes / 1024.0 / 1024.0
        );

        println!("  \x1b[1;32m├─ Scale Sensitivity ───────────────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Small-kernel efficiency:",
            predictor
                .derived_metrics
                .reuse_model
                .scale_sensitivity
                .small_kernel_efficiency
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Large-kernel scaling:",
            predictor
                .derived_metrics
                .reuse_model
                .scale_sensitivity
                .large_kernel_scaling
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Latency-dominated regime:",
            predictor
                .derived_metrics
                .reuse_model
                .scale_sensitivity
                .latency_dominated_regime_threshold
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Warp specialization viability:",
            predictor
                .derived_metrics
                .reuse_model
                .scale_sensitivity
                .warp_specialization_viability
        );

        println!("  \x1b[1;32m├─ Data Movement Model ─────────────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "DRAM latency cost:",
            predictor
                .derived_metrics
                .reuse_model
                .data_movement
                .dram_latency_cost
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "L2 reuse leverage:",
            predictor
                .derived_metrics
                .reuse_model
                .data_movement
                .l2_reuse_leverage
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "SMEM amortization:",
            predictor
                .derived_metrics
                .reuse_model
                .data_movement
                .smem_amortization
        );

        println!("  \x1b[1;32m├─ Execution Affordances ───────────────────────────────┤\x1b[0m");
        let large_reg = if obs.max_threads_per_sm.value >= 1536 {
            "yes"
        } else {
            "limited"
        };
        let persistent = if obs.l2_bytes.value >= 4 * 1024 * 1024 {
            "yes"
        } else {
            "limited"
        };
        let async_pipe = if arch.base >= 80 {
            "yes (cp.async)"
        } else {
            "none"
        };
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Large-register kernels viable:", large_reg
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Persistent kernels viable:", persistent
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Async pipelines:", async_pipe
        );

        println!("  \x1b[1;32m├─ SM Execution Model ──────────────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Max warps per SM:",
            predictor
                .derived_metrics
                .architectural_model
                .sm_limits
                .max_warps_per_sm
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Max blocks per SM:",
            predictor
                .derived_metrics
                .architectural_model
                .sm_limits
                .max_blocks_per_sm
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Schedulers per SM:",
            predictor
                .derived_metrics
                .architectural_model
                .sm_limits
                .schedulers_per_sm
        );

        println!("  \x1b[1;32m├─ Throughput Ratios ───────────────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {:.1} ops/cycle",
            "FP32 per SM:",
            predictor
                .derived_metrics
                .architectural_model
                .throughput_ratios
                .fp32_per_sm_ops_cycle
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {:.1} bytes/cycle",
            "LD/ST per SM:",
            predictor
                .derived_metrics
                .architectural_model
                .throughput_ratios
                .ldst_per_sm_bytes_cycle
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {:.1}",
            "Compute / Shared Memory Ratio:",
            predictor
                .derived_metrics
                .architectural_model
                .throughput_ratios
                .compute_to_shared_ratio
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {:.1}",
            "Compute / L2 Cache Ratio:",
            predictor
                .derived_metrics
                .architectural_model
                .throughput_ratios
                .compute_to_l2_ratio
        );

        println!("  \x1b[1;32m├─ Numeric Execution Modes ─────────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "FP32 math:",
            predictor
                .derived_metrics
                .architectural_model
                .math_modes
                .fp32_math_class
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "FP16 math:",
            predictor
                .derived_metrics
                .architectural_model
                .math_modes
                .fp16_math
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "Half2 vector math:",
            predictor
                .derived_metrics
                .architectural_model
                .math_modes
                .half2_vector_math
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "INT8 dot-product path:",
            predictor
                .derived_metrics
                .architectural_model
                .math_modes
                .int8_dot_product
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {}",
            "Mixed-precision accumulate:",
            predictor
                .derived_metrics
                .architectural_model
                .math_modes
                .mixed_precision_accumulate
        );

        println!("  \x1b[1;32m├─ Execution Constraints & Pressure ────────────────────┤\x1b[0m");
        let ec = &predictor.derived_metrics.execution_constraints;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Register pressure risk:", ec.register_pressure
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Shared memory pressure risk:", ec.shared_memory_pressure
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Bandwidth saturation risk:", ec.bandwidth_saturation
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Max async copies in flight:", ec.software_limits.max_async_copies_in_flight
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Max barriers limit:", ec.software_limits.max_barriers
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Cluster launch supported:",
            if ec.software_limits.cluster_launch_supported {
                "yes"
            } else {
                "no"
            }
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Cooperative grid viable:",
            if ec.software_limits.cooperative_grid_viable {
                "yes"
            } else {
                "no"
            }
        );

        println!("  \x1b[1;32m├─ Memory Reuse Envelope ───────────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "Expected reuse depth achievability:",
            predictor
                .derived_metrics
                .reuse_model
                .memory_envelope
                .expected_reuse_depth
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "SMEM staging depth:",
            predictor
                .derived_metrics
                .reuse_model
                .memory_envelope
                .smem_staging_depth
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "Register tiling headroom:",
            predictor
                .derived_metrics
                .reuse_model
                .memory_envelope
                .register_tiling_headroom
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "Register reuse leverage:",
            predictor
                .derived_metrics
                .reuse_model
                .memory_envelope
                .register_reuse_leverage
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "Bandwidth pressure regime:",
            predictor
                .derived_metrics
                .reuse_model
                .memory_envelope
                .bandwidth_pressure_regime
        );

        println!("  \x1b[1;32m├─ Modeler Predictions & Ranks ─────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}% - {}%",
            "Estimated occupancy window:",
            predictor.derived_metrics.reuse_model.occupancy_window.0,
            predictor.derived_metrics.reuse_model.occupancy_window.1
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {}",
            "Latency hiding sufficiency:", predictor.derived_metrics.reuse_model.latency_hiding
        );

        println!("  \x1b[1;32m├─ Scheduler Model Posteriors ──────────────────────────┤\x1b[0m");
        let smrm = &predictor
            .inference
            .resource_pressure_models
            .scheduler_pressure;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m \x1b[1;32m{:.1}%\x1b[0m",
            "Inferred Issue Rate (μ):",
            smrm.mean * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {:.3}",
            "Issue stddev (σ):", smrm.stddev
        );

        println!("  \x1b[1;32m├─ Formal Memory Model (Posteriors) ────────────────────┤\x1b[0m");
        let mm = &predictor.inference.memory_model;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[34m· {:<46}\x1b[0m {:.1}%",
            "P(Hit L1/SMEM):",
            mm.l1_hit_rate.mean * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[34m· {:<46}\x1b[0m {:.1}%",
            "P(Hit L2):",
            mm.l2_hit_rate.mean * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[34m· {:<46}\x1b[0m {:.1} MB",
            "Estimated DRAM Traffic:",
            mm.dram_traffic_bytes / 1024.0 / 1024.0
        );

        println!("  \x1b[1;32m├─ Calibrated Uncertainty Theory ───────────────────────┤\x1b[0m");
        let u = &predictor.uncertainty;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {:.3} (Lack of calibration)",
            "Epistemic Uncertainty:", u.epistemic
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {:.3} (Inherent task variance)",
            "Aleatoric Uncertainty:", u.aleatoric
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[33m· {:<46}\x1b[0m {:.3} (Cross-GPU drift)",
            "Transfer Uncertainty:", u.transfer
        );

        println!("  \x1b[1;32m├─ Workload-to-Arch Affinities ─────────────────────────┤\x1b[0m");
        let aff = &predictor.workload_affinities;
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Compute-dense affinity (GEMM):", aff.compute_dense_gemm
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Tensor Core offload affinity:", aff.tensor_core_gemm
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Memory streaming affinity:", aff.memory_streaming
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Reduction-heavy affinity:", aff.reduction_heavy
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[36m· {:<46}\x1b[0m {}",
            "Persistent grid fusion affinity:", aff.persistent_fusion
        );

        println!("  \x1b[1;32m├─ Implementation Strategies ───────────────────────────┤\x1b[0m");
        for likelihood in predictor.search_prior.candidate_kernels.iter() {
            println!(
                "  \x1b[1;32m│\x1b[0m [\x1b[1;32m{:>5.1}%\x1b[0m] {}",
                likelihood.probability * 100.0,
                likelihood.strategy
            );
            println!("  \x1b[1;32m│\x1b[0m         \x1b[90m↳ uncertainty: \x1b[33m{:.2}\x1b[0m | regime_conf: \x1b[33m{:.2}\x1b[0m | \x1b[90m{}\x1b[0m", likelihood.uncertainty, predictor.inference.confidence_breakdown.regime, likelihood.reasoning);
        }

        println!("  \x1b[1;32m├─ Confidence Breakdown ────────────────────────────────┤\x1b[0m");
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m \x1b[1;32m{:.1}%\x1b[0m",
            "Overall Regime Confidence:",
            predictor.inference.confidence_breakdown.regime * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {:.1}%",
            "Memory Traffic Model Confidence:",
            predictor.inference.confidence_breakdown.memory_model * 100.0
        );
        println!(
            "  \x1b[1;32m│\x1b[0m \x1b[35m· {:<46}\x1b[0m {:.1}%",
            "Scheduler & Pipeline Confidence:",
            predictor.inference.confidence_breakdown.scheduler_model * 100.0
        );

        if !predictor.inference.diagnostics.is_empty() {
            println!(
                "  \x1b[1;31m├─ Architectural Diagnostics & Anomalies ───────────────┤\x1b[0m"
            );
            for diag in &predictor.inference.diagnostics {
                println!("  \x1b[1;31m│\x1b[0m \x1b[33m· {}\x1b[0m", diag);
            }
        }
    }

    println!("  \x1b[1;32m└─────────────────────────────────────────────────────────┘\x1b[0m");
    println!();
}
