use crate::arch_metrics::{CalibrationFact, CalibrationMetric};
use crate::compute_cap::GpuArch;
use crate::error::{Error, Result};
use crate::toolkit::CudaToolkit;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// Trait for empirical hardware measurement probes.
pub trait CalibrationProbe {
    /// Name of the probe for logging.
    fn name(&self) -> &'static str;
    /// Generate the CUDA source code for the probe.
    fn generate_source(&self, arch: &GpuArch) -> String;
    /// Parse the output of the executed probe binary into calibration facts.
    fn parse_output(&self, stdout: &str) -> Vec<CalibrationFact>;
}

/// Measures peak DRAM bandwidth using a simple copy kernel.
pub struct BandwidthProbe;

impl CalibrationProbe for BandwidthProbe {
    fn name(&self) -> &'static str { "Memory Bandwidth" }

    fn generate_source(&self, _arch: &GpuArch) -> String {
        r#"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void copy_kernel(float* out, const float* in, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

int main() {
    const size_t n = 128 * 1024 * 1024; // 512MB
    float *d_in, *d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    copy_kernel<<<n/256, 256>>>(d_out, d_in, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for(int i=0; i<10; ++i) {
        copy_kernel<<<n/256, 256>>>(d_out, d_in, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Bandwidth = (Size * 2 (read+write) * Iterations) / Time
    double seconds = milliseconds / 1000.0;
    double bytes = (double)n * sizeof(float) * 2.0 * 10.0;
    double gb_s = (bytes / seconds) / 1e9;

    printf("BANDWIDTH: %.2f\n", gb_s);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
"#.to_string()
    }

    fn parse_output(&self, stdout: &str) -> Vec<CalibrationFact> {
        stdout.lines()
            .find(|l| l.starts_with("BANDWIDTH:"))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().parse::<f32>().ok())
            .map(|v| vec![CalibrationFact {
                metric: CalibrationMetric::DramBandwidth,
                measured_value: v,
            }])
            .unwrap_or_default()
    }
}

/// Measures peak FP32 compute throughput.
pub struct ComputeProbe;

impl CalibrationProbe for ComputeProbe {
    fn name(&self) -> &'static str { "FP32 Throughput" }

    fn generate_source(&self, arch: &GpuArch) -> String {
        let _arch_flag = format!("-arch=sm_{}", arch.base);
        format!(r#"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void flops_kernel(float* out, float a, float b) {{
    float val = (float)threadIdx.x;
    // Massive loop of FMAs to saturate arithmetic units
    #pragma unroll 128
    for(int i=0; i<1024; ++i) {{
        val = val * a + b;
        val = val * b + a;
        val = val * a + b;
        val = val * b + a;
    }}
    if (threadIdx.x == 0) out[blockIdx.x] = val;
}}

int main() {{
    float *d_out;
    cudaMalloc(&d_out, 1024 * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int blocks = 2048;
    int threads = 256;
    // Warmup
    flops_kernel<<<blocks, threads>>>(d_out, 1.01f, 1.02f);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    const int iterations = 100;
    for(int i=0; i<iterations; ++i) {{
        flops_kernel<<<blocks, threads>>>(d_out, 1.01f, 1.02f);
    }}
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Each iteration in the kernel has 1024 * 4 FMAs = 1024 * 8 FLOPs
    double total_flops = (double)blocks * threads * 1024.0 * 8.0 * (double)iterations;
    double seconds = milliseconds / 1000.0;
    double tflops = (total_flops / seconds) / 1e12;

    printf("TFLOPS: %.2f\n", tflops);

    cudaFree(d_out);
    return 0;
}}
"#)
    }

    fn parse_output(&self, stdout: &str) -> Vec<CalibrationFact> {
        stdout.lines()
            .find(|l| l.starts_with("TFLOPS:"))
            .and_then(|l| l.split(':').nth(1))
            .and_then(|v| v.trim().parse::<f32>().ok())
            .map(|v| vec![CalibrationFact {
                metric: CalibrationMetric::Fp32Compute,
                measured_value: v,
            }])
            .unwrap_or_default()
    }
}

/// Orchestrates the execution of hardware probes to calibrate the architecture model.
pub struct CalibrationEngine {
    toolkit: CudaToolkit,
    work_dir: PathBuf,
}

impl CalibrationEngine {
    /// Create a new calibration engine.
    pub fn new(toolkit: CudaToolkit, work_dir: PathBuf) -> Self {
        Self { toolkit, work_dir }
    }

    /// Run a series of probes and return the collected facts.
    pub fn run_probes(&self, arch: &GpuArch, probes: &[Box<dyn CalibrationProbe>]) -> Result<Vec<CalibrationFact>> {
        if !self.work_dir.exists() {
            fs::create_dir_all(&self.work_dir)?;
        }

        let mut all_facts = Vec::new();

        for probe in probes {
            let source = probe.generate_source(arch);
            let cu_file = self.work_dir.join(format!("{}.cu", probe.name().replace(' ', "_")));
            let bin_file = self.work_dir.join(format!("{}.bin", probe.name().replace(' ', "_")));

            fs::write(&cu_file, source)?;

            let nvcc = &self.toolkit.nvcc_path;

            let mut cmd = Command::new(nvcc);
            cmd.arg(&cu_file)
               .arg("-o").arg(&bin_file)
               .arg("-arch").arg(format!("sm_{}", arch.base))
               .arg("-O3");

            let status = cmd.status()?;
            if !status.success() {
                return Err(Error::CudaCompilationFailed(format!("Failed to compile probe: {}", probe.name())));
            }

            let output = Command::new(&bin_file).output()?;
            if !output.status.success() {
                return Err(Error::RuntimeError(format!("Failed to execute probe: {}", probe.name())));
            }

            let stdout = String::from_utf8_lossy(&output.stdout);
            all_facts.extend(probe.parse_output(&stdout));
        }

        Ok(all_facts)
    }
}
