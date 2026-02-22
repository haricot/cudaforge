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
        _ => print_capabilities(),
    }
}

fn print_help() {
    eprintln!("cargo-cudaforge — CUDA capability inspector");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    cargo cudaforge");
    eprintln!();
    eprintln!("Detects the installed GPU (via nvidia-smi or CUDA_COMPUTE_CAP)");
    eprintln!("and CUDA toolkit (via nvcc), then prints which hardware and");
    eprintln!("toolkit capabilities are available.");
}

fn print_capabilities() {
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
    println!("  \x1b[1;36m║\x1b[0m  GPU Compute Capability: \x1b[1;33mSM {}{}\x1b[0m{:<20}\x1b[1;36m║\x1b[0m",
        arch.base, suffix, ver_str);
    if !nvcc_str.is_empty() {
        println!(
            "  \x1b[1;36m║\x1b[0m  Toolkit:{:<48}\x1b[1;36m║\x1b[0m",
            nvcc_str
        );
    }
    println!("  \x1b[1;36m╚════════════════════════════════════════════════════════╝\x1b[0m");
    println!();

    // ── Evaluate capabilities (reuse library logic, no cargo: output) ─────
    let hw_results = cudaforge::evaluate_hw_capabilities(&arch);
    let tk_results = cuda_version
        .as_ref()
        .map(|ver| cudaforge::evaluate_toolkit_capabilities(&arch, ver));

    // ── Hardware capabilities ───────────────────────────────────────────────
    println!("  \x1b[1;32m┌─ Hardware (SM) ─────────────────────────────────────────┐\x1b[0m");
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

    // ── Toolkit capabilities ────────────────────────────────────────────────
    if let Some(ref ver) = cuda_version {
        println!(
            "  \x1b[1;32m├─ Toolkit (CUDA {}) ──────────────────────────────────┤\x1b[0m",
            ver
        );
        if let Some(ref results) = tk_results {
            for (name, enabled) in results {
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
        }
    }

    println!("  \x1b[1;32m└─────────────────────────────────────────────────────────┘\x1b[0m");
    println!();
}
