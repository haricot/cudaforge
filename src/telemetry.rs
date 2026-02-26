use serde::Serialize;

use crate::predictor::{CalibrationFeedback, CalibrationState, KernelIntent, ShapeManifold};

/// A standardized diagnostic and execution log for a specific GPU kernel run.
///
/// The `GpuEEGLog` (Electroencephalogram Log) is designed to be the universal telemetry
/// format connecting high-level graph compilers (like Luminal or MLIR) with the underlying
/// analytical predictor (`cudaforge`) and hardware profilers (`ncu`).
///
/// It tracks the full cognitive pipeline:
/// 1. What was the initial stimulus (the problem)?
/// 2. What was the Oracle's prior belief (the prediction)?
/// 3. What did the hardware actually do (the empirical reality)?
/// 4. How wrong was the Oracle (the divergence delta)?
/// 5. How did the Oracle adjust its internal model (the posterior update)?
#[derive(Debug, Clone, Serialize)]
pub struct GpuEEGLog {
    /// Timestamp or unique identifier for this kernel execution.
    pub session_id: String,

    /// The target architecture's compute capability (e.g., 80 for Ampere).
    pub arch_base: u32,

    /// Step 1: The initial algorithmic Stimulus
    pub stimulus: EegStimulus,

    /// Step 2: The Predictor's Prior Belief
    pub prior_prediction: EegPrior,

    /// Step 3: Raw Hardware Telemetry
    pub empirical_response: CalibrationFeedback,

    /// Step 4: Analytical Divergence (Error Calculation)
    pub divergence_delta: EegDivergence,

    /// Step 5: The Learned Posterior State
    pub posterior_state: CalibrationState,
}

/// The algorithmic stimulus sent to the GPU.
#[derive(Debug, Clone, Serialize)]
pub struct EegStimulus {
    /// The normalized grouping this shape belongs to (enabling cross-kernel learning).
    pub shape_manifold: ShapeManifold,
    /// The suggested tiling and staging strategy.
    pub tile_strategy: String,
    /// The precision to use for calculations.
    pub dtype_compute: String,
}

/// The state of the Oracle's beliefs *before* the kernel executed.
#[derive(Debug, Clone, Serialize)]
pub struct EegPrior {
    /// Expected runtime in microseconds.
    pub expected_runtime_us: f32,
    /// Expected issue slot utilization.
    pub expected_issue_util: f32,
    /// Expected bandwidth utilization.
    pub expected_bw_util: f32,
    /// The kernel predictability score at the time of prediction (0.0 to 1.0).
    pub prediction_predictability: f32,
    /// The probability distribution of expected bottlenecks.
    pub regime_posterior: std::collections::HashMap<String, f32>,
}

/// The calculated error between the Oracle's prior and the hardware's reality.
#[derive(Debug, Clone, Serialize)]
pub struct EegDivergence {
    /// Absolute error in runtime prediction (microseconds).
    pub runtime_error_us: f32,
    /// Ratio of measured vs expected runtime (1.0 = perfect prediction).
    pub runtime_ratio: f32,
    /// Ratio of measured vs expected issue utilization.
    pub issue_ratio: f32,
    /// Ratio of measured vs expected bandwidth utilization.
    pub bw_ratio: f32,
}

impl GpuEEGLog {
    /// Constructs a new EEGLog by processing a predicted Intent against empirical Feedback.
    pub fn build(
        session_id: impl Into<String>,
        arch_base: u32,
        intent: &KernelIntent,
        feedback: &CalibrationFeedback,
        _pre_update_state: &CalibrationState,
        post_update_state: &CalibrationState,
    ) -> Self {
        let prior = EegPrior {
            expected_runtime_us: intent.expected_runtime_us,
            expected_issue_util: intent.performance_signature.flop_utilization.mean,
            expected_bw_util: intent.performance_signature.bw_utilization.mean,
            prediction_predictability: intent.performance_signature.predictability.score,
            regime_posterior: intent
                .performance_signature
                .dominant_limits
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        };

        let runtime_ratio = if intent.expected_runtime_us > 0.0 {
            feedback.measured_runtime_us / intent.expected_runtime_us
        } else {
            1.0
        };

        let issue_ratio = if prior.expected_issue_util > 0.0 {
            feedback.measured_issue_utilization / prior.expected_issue_util
        } else {
            1.0
        };

        let bw_ratio = if prior.expected_bw_util > 0.0 {
            feedback.measured_bw_utilization / prior.expected_bw_util
        } else {
            1.0
        };

        let divergence = EegDivergence {
            runtime_error_us: feedback.measured_runtime_us - intent.expected_runtime_us,
            runtime_ratio,
            issue_ratio,
            bw_ratio,
        };

        Self {
            session_id: session_id.into(),
            arch_base,
            stimulus: EegStimulus {
                shape_manifold: intent.shape_manifold.clone(),
                tile_strategy: intent.tile_strategy.clone(),
                dtype_compute: format!("{:?}", intent.dtype_compute),
            },
            prior_prediction: prior,
            empirical_response: feedback.clone(),
            divergence_delta: divergence,
            posterior_state: post_update_state.clone(),
        }
    }
}
