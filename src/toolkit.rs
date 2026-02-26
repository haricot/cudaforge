//! CUDA toolkit auto-detection

use crate::error::{Error, Result};
use std::path::PathBuf;
use std::process::Command;

/// Standard CUDA installation paths to search
const CUDA_SEARCH_PATHS: &[&str] = &[
    "/usr/local/cuda",
    "/opt/cuda",
    "/usr/lib/cuda",
    "/usr",
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA",
    "C:/CUDA",
];

/// Parsed CUDA toolkit version for capability checks
///
/// Enables compile-time gating of features that depend on the CUDA toolkit
/// version rather than (or in addition to) the GPU compute capability.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaVersion {
    /// Major version (e.g. 12 for CUDA 12.6)
    pub major: u32,
    /// Minor version (e.g. 6 for CUDA 12.6)
    pub minor: u32,
}

/// Parsed cuDNN version
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CudnnVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl CudnnVersion {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    pub fn at_least(&self, major: u32, minor: u32) -> bool {
        (self.major, self.minor) >= (major, minor)
    }
}

impl std::fmt::Display for CudnnVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl CudaVersion {
    /// Create a new CUDA version
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// Parse from a version string like "12.1", "11.8", "12"
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if s.is_empty() {
            return None;
        }
        let mut parts = s.split('.');
        let major = parts.next()?.parse::<u32>().ok()?;
        let minor = parts
            .next()
            .and_then(|m| m.parse::<u32>().ok())
            .unwrap_or(0);
        Some(Self { major, minor })
    }

    /// Check if this version is at least (major, minor)
    pub fn at_least(&self, major: u32, minor: u32) -> bool {
        (self.major, self.minor) >= (major, minor)
    }
}

impl PartialOrd for CudaVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CudaVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.major, self.minor).cmp(&(other.major, other.minor))
    }
}

impl std::fmt::Display for CudaVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// CUDA toolkit information
#[derive(Debug, Clone)]
pub struct CudaToolkit {
    /// Path to nvcc binary
    pub nvcc_path: PathBuf,
    /// CUDA include directory
    pub include_dir: PathBuf,
    /// CUDA lib directory
    pub lib_dir: PathBuf,
    /// CUDA version string (if detected), e.g. "12.1"
    pub version: Option<String>,
    /// Parsed CUDA version for capability checks
    pub parsed_version: Option<CudaVersion>,
    /// Parsed cuDNN version (if detected)
    pub cudnn_version: Option<CudnnVersion>,
}

impl CudaToolkit {
    /// Auto-detect CUDA toolkit installation
    ///
    /// Search order:
    /// 1. NVCC environment variable
    /// 2. which nvcc (PATH lookup)
    /// 3. CUDA_HOME/bin/nvcc
    /// 4. Common installation paths
    pub fn detect() -> Result<Self> {
        let nvcc_path = find_nvcc()?;

        // Derive include and lib directories from nvcc location
        let cuda_root = nvcc_path
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| Error::CudaToolkitNotFound(nvcc_path.clone()))?;

        let include_dir = cuda_root.join("include");
        let lib_dir = if cfg!(target_os = "windows") {
            cuda_root.join("lib").join("x64")
        } else {
            cuda_root.join("lib64")
        };

        let version = detect_cuda_version(&nvcc_path);
        let parsed_version = version.as_deref().and_then(CudaVersion::parse);
        let cudnn_version = detect_cudnn_version(&include_dir);

        Ok(Self {
            nvcc_path,
            include_dir,
            lib_dir,
            version,
            parsed_version,
            cudnn_version,
        })
    }

    /// Create toolkit from explicit nvcc path
    pub fn from_nvcc_path(nvcc_path: PathBuf) -> Result<Self> {
        if !nvcc_path.exists() {
            return Err(Error::NvccNotFound(nvcc_path.display().to_string()));
        }

        let cuda_root = nvcc_path
            .parent()
            .and_then(|p| p.parent())
            .ok_or_else(|| Error::CudaToolkitNotFound(nvcc_path.clone()))?;

        let include_dir = cuda_root.join("include");
        let lib_dir = if cfg!(target_os = "windows") {
            cuda_root.join("lib").join("x64")
        } else {
            cuda_root.join("lib64")
        };

        let version = detect_cuda_version(&nvcc_path);
        let parsed_version = version.as_deref().and_then(CudaVersion::parse);
        let cudnn_version = detect_cudnn_version(&include_dir);

        Ok(Self {
            nvcc_path,
            include_dir,
            lib_dir,
            version,
            parsed_version,
            cudnn_version,
        })
    }

    /// Get supported GPU architectures from nvcc
    pub fn supported_architectures(&self) -> Vec<usize> {
        let output = Command::new(&self.nvcc_path)
            .arg("--list-gpu-code")
            .output();

        if let Ok(output) = output {
            let stdout = String::from_utf8_lossy(&output.stdout);
            parse_gpu_codes(&stdout)
        } else {
            Vec::new()
        }
    }

    /// Get the parsed CUDA version, or a sensible fallback
    ///
    /// If the version could not be detected, returns a conservative
    /// CUDA 10.0 assumption (oldest version still commonly encountered).
    pub fn cuda_version_or_default(&self) -> CudaVersion {
        self.parsed_version
            .clone()
            .unwrap_or_else(|| CudaVersion::new(10, 0))
    }
}

/// Find nvcc binary following search order
fn find_nvcc() -> Result<PathBuf> {
    // 1. Check NVCC environment variable
    if let Ok(nvcc) = std::env::var("NVCC") {
        let path = PathBuf::from(&nvcc);
        if path.exists() {
            return Ok(path);
        }
    }

    // 2. Try which::which for PATH lookup
    if let Ok(path) = which::which("nvcc") {
        return Ok(path);
    }

    // 3. Check CUDA_HOME
    if let Ok(cuda_home) = std::env::var("CUDA_HOME") {
        let nvcc = PathBuf::from(&cuda_home).join("bin").join("nvcc");
        if nvcc.exists() {
            return Ok(nvcc);
        }
    }

    // 4. Check CUDA_PATH (Windows)
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(&cuda_path).join("bin").join("nvcc.exe");
        if nvcc.exists() {
            return Ok(nvcc);
        }
    }

    // 5. Search common installation paths
    for base_path in CUDA_SEARCH_PATHS {
        let base = PathBuf::from(base_path);

        // Try direct path
        let nvcc = base.join("bin").join(if cfg!(target_os = "windows") {
            "nvcc.exe"
        } else {
            "nvcc"
        });
        if nvcc.exists() {
            return Ok(nvcc);
        }

        // On Windows, search versioned subdirectories
        if cfg!(target_os = "windows") && base.exists() {
            if let Ok(entries) = std::fs::read_dir(&base) {
                for entry in entries.flatten() {
                    let nvcc = entry.path().join("bin").join("nvcc.exe");
                    if nvcc.exists() {
                        return Ok(nvcc);
                    }
                }
            }
        }
    }

    Err(Error::NvccNotFound(
        "No nvcc found in PATH or standard locations".to_string(),
    ))
}

/// Parse GPU codes from nvcc --list-gpu-code output
fn parse_gpu_codes(output: &str) -> Vec<usize> {
    let mut codes = Vec::new();
    for line in output.lines() {
        let parts: Vec<&str> = line.split('_').collect();
        if parts.len() >= 2 && parts.contains(&"sm") {
            if let Ok(code) = parts[1].parse::<usize>() {
                codes.push(code);
            }
        }
    }
    codes.sort();
    codes.dedup();
    codes
}

/// Detect CUDA version from nvcc
fn detect_cuda_version(nvcc_path: &PathBuf) -> Option<String> {
    let output = Command::new(nvcc_path).arg("--version").output().ok()?;

    let stdout = String::from_utf8_lossy(&output.stdout);

    // Parse version from output like "release 12.1, V12.1.105"
    for line in stdout.lines() {
        if line.contains("release") {
            if let Some(version_part) = line.split("release").nth(1) {
                let version = version_part.trim().split(',').next()?;
                return Some(version.trim().to_string());
            }
        }
    }

    None
}

/// Detect cuDNN version by scanning headers
fn detect_cudnn_version(cuda_include: &PathBuf) -> Option<CudnnVersion> {
    let mut search_paths = Vec::new();

    // 1. Check CUDNN_HOME
    if let Ok(home) = std::env::var("CUDNN_HOME") {
        let home_path = PathBuf::from(home);
        search_paths.push(home_path.join("include"));
        search_paths.push(home_path.clone());
    }
    // 2. Check CUDNN_LIB
    if let Ok(lib) = std::env::var("CUDNN_LIB") {
        let lib_path = PathBuf::from(lib);
        search_paths.push(lib_path.join("include"));
        search_paths.push(lib_path.clone());
        if let Some(parent) = lib_path.parent() {
            search_paths.push(parent.join("include"));
            search_paths.push(parent.to_path_buf());
        }
    }
    // 3. Scan LD_LIBRARY_PATH
    if let Ok(ld_path) = std::env::var("LD_LIBRARY_PATH") {
        for path_str in std::env::split_paths(&ld_path) {
            search_paths.push(path_str.join("include"));
            if let Some(parent) = path_str.parent() {
                search_paths.push(parent.join("include"));
            }
        }
    }
    // 3. CUDA include path
    search_paths.push(cuda_include.clone());
    // 4. Common system paths
    search_paths.push(PathBuf::from("/usr/include"));
    search_paths.push(PathBuf::from("/usr/include/x86_64-linux-gnu"));

    for path in search_paths {
        if !path.exists() {
            continue;
        }

        // Prefer cudnn_version.h (v8+), fallback to cudnn.h
        let v_h = path.join("cudnn_version.h");
        let h = path.join("cudnn.h");

        let target = if v_h.exists() {
            Some(v_h)
        } else if h.exists() {
            Some(h)
        } else {
            None
        };

        if let Some(header) = target {
            if let Ok(content) = std::fs::read_to_string(&header) {
                let major = extract_define(&content, "CUDNN_MAJOR");
                let minor = extract_define(&content, "CUDNN_MINOR");
                let patch = extract_define(&content, "CUDNN_PATCHLEVEL");

                if let (Some(maj), Some(min), Some(pat)) = (major, minor, patch) {
                    return Some(CudnnVersion::new(maj, min, pat));
                }
            }
        }
    }
    None
}

fn extract_define(content: &str, name: &str) -> Option<u32> {
    let re = re_define(name);
    for line in content.lines() {
        if let Some(caps) = re.captures(line) {
            return caps.val.parse().ok();
        }
    }
    None
}

// Simple regex replacement since we don't have regex crate in dependencies (checked Cargo.toml earlier)
// Actually Cargo.toml had: glob, num_cpus, rayon, which, sha2, walkdir, anyhow, serde, serde_json, thiserror, fs2. No regex.
// I'll use simple string matching.

fn re_define(name: &str) -> SimpleDefineMatcher {
    SimpleDefineMatcher {
        name: name.to_string(),
    }
}

struct SimpleDefineMatcher {
    name: String,
}

impl SimpleDefineMatcher {
    fn captures<'a>(&self, line: &'a str) -> Option<SimpleCaps<'a>> {
        let line = line.trim();
        if !line.starts_with("#define") {
            return None;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 && parts[1] == self.name {
            return Some(SimpleCaps { val: parts[2] });
        }
        None
    }
}

struct SimpleCaps<'a> {
    val: &'a str,
}

impl<'a> SimpleCaps<'a> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gpu_codes() {
        let output = "sm_52\nsm_60\nsm_70\nsm_75\nsm_80\nsm_86\nsm_89\nsm_90";
        let codes = parse_gpu_codes(output);
        assert!(codes.contains(&80));
        assert!(codes.contains(&90));
    }

    #[test]
    fn test_cuda_version_parse() {
        let v = CudaVersion::parse("12.1").unwrap();
        assert_eq!(v.major, 12);
        assert_eq!(v.minor, 1);

        let v = CudaVersion::parse("11.8").unwrap();
        assert_eq!(v.major, 11);
        assert_eq!(v.minor, 8);

        let v = CudaVersion::parse("12").unwrap();
        assert_eq!(v.major, 12);
        assert_eq!(v.minor, 0);

        assert!(CudaVersion::parse("").is_none());
        assert!(CudaVersion::parse("abc").is_none());
    }

    #[test]
    fn test_cuda_version_at_least() {
        let v = CudaVersion::new(12, 1);
        assert!(v.at_least(12, 0));
        assert!(v.at_least(12, 1));
        assert!(!v.at_least(12, 2));
        assert!(v.at_least(11, 8));
        assert!(!v.at_least(13, 0));
    }

    #[test]
    fn test_cuda_version_ordering() {
        assert!(CudaVersion::new(12, 1) > CudaVersion::new(11, 8));
        assert!(CudaVersion::new(12, 1) > CudaVersion::new(12, 0));
        assert!(CudaVersion::new(12, 1) == CudaVersion::new(12, 1));
        assert!(CudaVersion::new(10, 0) < CudaVersion::new(11, 0));
    }

    #[test]
    fn test_cuda_version_display() {
        assert_eq!(format!("{}", CudaVersion::new(12, 6)), "12.6");
        assert_eq!(format!("{}", CudaVersion::new(11, 0)), "11.0");
    }
}
