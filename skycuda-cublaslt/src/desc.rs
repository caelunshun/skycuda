// --- LLM-generated --- //

//! The problem description (`MatmulDesc`) and the heuristic's search
//! constraints (`MatmulPreference`), plus the owned cuBLASLt descriptors
//! built from them.

use crate::{
    enums::{
        ComputeType, DataType, Epilogue, NumericalImplFlags, Operation, PointerMode,
        ReductionSchemeMask, ScaleMode,
    },
    error::{Result, check},
    layout::MatrixLayout,
    sys,
};
use std::ptr;

/// Which operands carry a scaling factor, and how cuBLASLt interprets it.
///
/// `Some(mode)` makes the matching `*_scale` operand required at execute time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct Scales {
    pub a: Option<ScaleMode>,
    pub b: Option<ScaleMode>,
    pub c: Option<ScaleMode>,
    /// Scales D on the way *in* to the compute range.
    pub d: Option<ScaleMode>,
    pub aux: Option<ScaleMode>,
    /// Scales D on the way *out* to storage (`D_OUT_SCALE`).
    pub d_out: Option<ScaleMode>,
}

/// A complete description of `D = epilogue(alpha * op(A) op(B) + beta * C)`.
///
/// Build one with [`MatmulDesc::new`] and the builder-style setters, then hand
/// it to [`CublasLtHandle::build_plan`](crate::CublasLtHandle::build_plan).
///
/// FP8 matmuls are the fussiest case: cuBLASLt requires `trans_a == T` and
/// `trans_b == N` (the column-major "TN" form), 16-byte aligned operands and
/// [`ComputeType::F32`]. Anything else makes the heuristic return
/// [`Status::NotSupported`](crate::Status::NotSupported), which `build_plan`
/// surfaces verbatim.
#[derive(Debug, Clone)]
pub struct MatmulDesc {
    pub compute_type: ComputeType,
    /// Type of alpha and beta. Defaults to
    /// [`ComputeType::default_scale_type`].
    pub scale_type: DataType,
    pub trans_a: Operation,
    pub trans_b: Operation,
    pub a: MatrixLayout,
    pub b: MatrixLayout,
    pub d: MatrixLayout,
    /// `None` means the matmul is in place: C aliases D and shares its layout.
    pub c: Option<MatrixLayout>,
    pub epilogue: Epilogue,
    /// Element type of the bias/bias-gradient vector. `None` leaves cuBLASLt's
    /// own default, which follows D's type.
    pub bias_data_type: Option<DataType>,
    pub bias_batch_stride: i64,
    /// Element type of the epilogue auxiliary buffer. `None` leaves cuBLASLt's
    /// default.
    pub aux_data_type: Option<DataType>,
    pub aux_ld: i64,
    pub aux_batch_stride: i64,
    /// Only [`PointerMode::Host`] is supported in this version.
    pub pointer_mode: PointerMode,
    /// FP8 fast accumulation: faster, but intermediate results are not
    /// periodically promoted to a higher precision.
    pub fast_accum: bool,
    /// Number of SMs the heuristic should assume are available. 0 means all.
    pub sm_count_target: i32,
    pub scales: Scales,
    pub alpha_batch_stride: i64,
    pub beta_batch_stride: i64,
}

impl MatmulDesc {
    pub fn new(
        compute_type: ComputeType,
        a: MatrixLayout,
        b: MatrixLayout,
        d: MatrixLayout,
    ) -> Self {
        Self {
            compute_type,
            scale_type: compute_type.default_scale_type(),
            trans_a: Operation::N,
            trans_b: Operation::N,
            a,
            b,
            d,
            c: None,
            epilogue: Epilogue::Default,
            bias_data_type: None,
            bias_batch_stride: 0,
            aux_data_type: None,
            aux_ld: 0,
            aux_batch_stride: 0,
            pointer_mode: PointerMode::Host,
            fast_accum: false,
            sm_count_target: 0,
            scales: Scales::default(),
            alpha_batch_stride: 0,
            beta_batch_stride: 0,
        }
    }

    pub fn scale_type(mut self, scale_type: DataType) -> Self {
        self.scale_type = scale_type;
        self
    }

    pub fn trans_a(mut self, op: Operation) -> Self {
        self.trans_a = op;
        self
    }

    pub fn trans_b(mut self, op: Operation) -> Self {
        self.trans_b = op;
        self
    }

    /// Reads C from a separate matrix instead of aliasing D.
    pub fn c(mut self, c: MatrixLayout) -> Self {
        self.c = Some(c);
        self
    }

    pub fn epilogue(mut self, epilogue: Epilogue) -> Self {
        self.epilogue = epilogue;
        self
    }

    pub fn bias_data_type(mut self, data_type: DataType) -> Self {
        self.bias_data_type = Some(data_type);
        self
    }

    pub fn bias_batch_stride(mut self, stride: i64) -> Self {
        self.bias_batch_stride = stride;
        self
    }

    pub fn aux_data_type(mut self, data_type: DataType) -> Self {
        self.aux_data_type = Some(data_type);
        self
    }

    /// Leading dimension of the auxiliary buffer, in elements (bits for the
    /// ReLU bit-mask). Required whenever the epilogue uses the buffer.
    pub fn aux_ld(mut self, ld: i64) -> Self {
        self.aux_ld = ld;
        self
    }

    pub fn aux_batch_stride(mut self, stride: i64) -> Self {
        self.aux_batch_stride = stride;
        self
    }

    pub fn pointer_mode(mut self, mode: PointerMode) -> Self {
        self.pointer_mode = mode;
        self
    }

    pub fn fast_accum(mut self, enabled: bool) -> Self {
        self.fast_accum = enabled;
        self
    }

    pub fn sm_count_target(mut self, sm_count: i32) -> Self {
        self.sm_count_target = sm_count;
        self
    }

    pub fn scales(mut self, scales: Scales) -> Self {
        self.scales = scales;
        self
    }

    pub fn alpha_batch_stride(mut self, stride: i64) -> Self {
        self.alpha_batch_stride = stride;
        self
    }

    pub fn beta_batch_stride(mut self, stride: i64) -> Self {
        self.beta_batch_stride = stride;
        self
    }

    /// C's layout, which is D's when the matmul is in place.
    pub fn c_layout(&self) -> &MatrixLayout {
        self.c.as_ref().unwrap_or(&self.d)
    }
}

/// Constraints the heuristic search must respect.
#[derive(Debug, Clone)]
pub struct MatmulPreference {
    /// `None` selects NVIDIA's recommended default: 32 MiB on sm_90 and newer,
    /// 4 MiB otherwise.
    pub max_workspace_bytes: Option<usize>,
    /// Alignment, in bytes, the caller guarantees for A, B, C and D. cuBLASLt's
    /// own default is 256 for each; a smaller value admits fewer algorithms.
    /// [`MatmulPlan::execute`](crate::MatmulPlan::execute) asserts the real
    /// pointers meet it.
    pub min_alignment_bytes: [u32; 4],
    /// Alignment guaranteed for the bias, auxiliary, scale and amax pointers.
    /// These are also the placeholder pointer values used when running the
    /// heuristic, which infers alignment from the pointer value alone.
    pub aux_alignment_bytes: u32,
    /// Exclude algorithms reporting device utilization above this. 0 disables
    /// the filter.
    pub max_waves_count: f32,
    pub reduction_scheme_mask: ReductionSchemeMask,
    pub impl_mask: NumericalImplFlags,
    /// How many heuristic results to ask for. Eight keeps
    /// [`MatmulPlan::algorithms`](crate::MatmulPlan::algorithms) useful for
    /// hand-tuning without slowing the build down; set 1 for the leanest
    /// build. This is a convention of this crate, not an NVIDIA
    /// recommendation.
    pub requested_algo_count: usize,
}

impl Default for MatmulPreference {
    fn default() -> Self {
        Self {
            max_workspace_bytes: None,
            min_alignment_bytes: [256; 4],
            aux_alignment_bytes: 256,
            max_waves_count: 0.0,
            reduction_scheme_mask: ReductionSchemeMask::ALL,
            impl_mask: NumericalImplFlags::ALL,
            requested_algo_count: 8,
        }
    }
}

impl MatmulPreference {
    pub fn max_workspace_bytes(mut self, bytes: usize) -> Self {
        self.max_workspace_bytes = Some(bytes);
        self
    }

    /// Sets the same guaranteed alignment for A, B, C and D.
    pub fn min_alignment_bytes(mut self, bytes: u32) -> Self {
        self.min_alignment_bytes = [bytes; 4];
        self
    }

    pub fn aux_alignment_bytes(mut self, bytes: u32) -> Self {
        self.aux_alignment_bytes = bytes;
        self
    }

    pub fn max_waves_count(mut self, waves: f32) -> Self {
        self.max_waves_count = waves;
        self
    }

    pub fn reduction_scheme_mask(mut self, mask: ReductionSchemeMask) -> Self {
        self.reduction_scheme_mask = mask;
        self
    }

    pub fn impl_mask(mut self, mask: NumericalImplFlags) -> Self {
        self.impl_mask = mask;
        self
    }

    pub fn requested_algo_count(mut self, count: usize) -> Self {
        self.requested_algo_count = count;
        self
    }

    /// The workspace cap, resolving `None` against the device's compute
    /// capability.
    pub(crate) fn resolved_max_workspace_bytes(&self, compute_capability_major: u32) -> usize {
        self.max_workspace_bytes.unwrap_or({
            if compute_capability_major >= 9 {
                32 * 1024 * 1024
            } else {
                4 * 1024 * 1024
            }
        })
    }
}

/// Descriptor pointer attributes, which are rewritten on every execute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PointerAttr {
    Bias = 0,
    Aux,
    AScale,
    BScale,
    CScale,
    DScale,
    AuxScale,
    DOutScale,
    AmaxD,
    AmaxAux,
}

impl PointerAttr {
    pub(crate) const COUNT: usize = 10;

    pub(crate) const ALL: [PointerAttr; Self::COUNT] = [
        PointerAttr::Bias,
        PointerAttr::Aux,
        PointerAttr::AScale,
        PointerAttr::BScale,
        PointerAttr::CScale,
        PointerAttr::DScale,
        PointerAttr::AuxScale,
        PointerAttr::DOutScale,
        PointerAttr::AmaxD,
        PointerAttr::AmaxAux,
    ];

    fn to_ffi(self) -> sys::cublasLtMatmulDescAttributes_t {
        use sys::cublasLtMatmulDescAttributes_t::*;
        match self {
            PointerAttr::Bias => CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            PointerAttr::Aux => CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
            PointerAttr::AScale => CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            PointerAttr::BScale => CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            PointerAttr::CScale => CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
            PointerAttr::DScale => CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
            PointerAttr::AuxScale => CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER,
            PointerAttr::DOutScale => CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER,
            PointerAttr::AmaxD => CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
            PointerAttr::AmaxAux => CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER,
        }
    }
}

/// An owned `cublasLtMatmulDesc_t`, with the last pointer values written to it
/// so repeated executes only touch the attributes that actually changed.
#[derive(Debug)]
pub(crate) struct RawMatmulDesc {
    raw: sys::cublasLtMatmulDesc_t,
    pointers: [u64; PointerAttr::COUNT],
}

impl RawMatmulDesc {
    pub fn new(desc: &MatmulDesc) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(
            unsafe {
                sys::cublasLtMatmulDescCreate(
                    &mut raw,
                    desc.compute_type.to_ffi(),
                    desc.scale_type.to_ffi(),
                )
            },
            "cublasLtMatmulDescCreate",
        )?;
        let this = Self {
            raw,
            pointers: [0; PointerAttr::COUNT],
        };

        use sys::cublasLtMatmulDescAttributes_t::*;
        this.set(
            CUBLASLT_MATMUL_DESC_SCALE_TYPE,
            &(desc.scale_type.raw() as i32),
        )?;
        this.set(CUBLASLT_MATMUL_DESC_TRANSA, &(desc.trans_a.raw() as i32))?;
        this.set(CUBLASLT_MATMUL_DESC_TRANSB, &(desc.trans_b.raw() as i32))?;
        this.set(CUBLASLT_MATMUL_DESC_EPILOGUE, &desc.epilogue.raw())?;
        this.set(
            CUBLASLT_MATMUL_DESC_POINTER_MODE,
            &(desc.pointer_mode.raw() as i32),
        )?;
        this.set(CUBLASLT_MATMUL_DESC_FAST_ACCUM, &(desc.fast_accum as i8))?;
        this.set(CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &desc.sm_count_target)?;
        this.set(
            CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE,
            &desc.bias_batch_stride,
        )?;
        this.set(CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &desc.aux_ld)?;
        this.set(
            CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
            &desc.aux_batch_stride,
        )?;
        this.set(
            CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE,
            &desc.alpha_batch_stride,
        )?;
        this.set(
            CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE,
            &desc.beta_batch_stride,
        )?;

        if let Some(dt) = desc.bias_data_type {
            this.set(CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &(dt.raw() as i32))?;
        }
        if let Some(dt) = desc.aux_data_type {
            this.set(
                CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE,
                &(dt.raw() as i32),
            )?;
        }

        for (mode, attr) in [
            (desc.scales.a, CUBLASLT_MATMUL_DESC_A_SCALE_MODE),
            (desc.scales.b, CUBLASLT_MATMUL_DESC_B_SCALE_MODE),
            (desc.scales.c, CUBLASLT_MATMUL_DESC_C_SCALE_MODE),
            (desc.scales.d, CUBLASLT_MATMUL_DESC_D_SCALE_MODE),
            (
                desc.scales.aux,
                CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_MODE,
            ),
            (desc.scales.d_out, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE),
        ] {
            if let Some(mode) = mode {
                this.set(attr, &(mode.raw() as i32))?;
            }
        }

        Ok(this)
    }

    pub fn raw(&self) -> sys::cublasLtMatmulDesc_t {
        self.raw
    }

    /// Writes a pointer attribute, skipping the FFI call when the descriptor
    /// already holds this value.
    pub fn set_pointer(&mut self, attr: PointerAttr, value: u64) -> Result<()> {
        if self.pointers[attr as usize] == value {
            return Ok(());
        }
        let as_ptr = value as *const std::ffi::c_void;
        self.set(attr.to_ffi(), &as_ptr)?;
        self.pointers[attr as usize] = value;
        Ok(())
    }

    fn set<T>(&self, attr: sys::cublasLtMatmulDescAttributes_t, value: &T) -> Result<()> {
        check(
            unsafe {
                sys::cublasLtMatmulDescSetAttribute(
                    self.raw,
                    attr,
                    (value as *const T).cast(),
                    size_of::<T>(),
                )
            },
            "cublasLtMatmulDescSetAttribute",
        )
    }
}

impl Drop for RawMatmulDesc {
    fn drop(&mut self) {
        unsafe {
            sys::cublasLtMatmulDescDestroy(self.raw);
        }
    }
}

// SAFETY: the descriptor is only mutated behind the plan's mutex.
unsafe impl Send for RawMatmulDesc {}
unsafe impl Sync for RawMatmulDesc {}

/// An owned `cublasLtMatmulPreference_t`. Only lives for the duration of
/// `build_plan`.
pub(crate) struct RawMatmulPreference(sys::cublasLtMatmulPreference_t);

impl RawMatmulPreference {
    pub fn new(pref: &MatmulPreference, compute_capability_major: u32) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(
            unsafe { sys::cublasLtMatmulPreferenceCreate(&mut raw) },
            "cublasLtMatmulPreferenceCreate",
        )?;
        let this = Self(raw);

        use sys::cublasLtMatmulPreferenceAttributes_t::*;
        this.set(
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &(pref.resolved_max_workspace_bytes(compute_capability_major) as u64),
        )?;
        for (attr, bytes) in [
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
            CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        ]
        .into_iter()
        .zip(pref.min_alignment_bytes)
        {
            this.set(attr, &bytes)?;
        }
        this.set(CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT, &pref.max_waves_count)?;
        this.set(
            CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
            &pref.reduction_scheme_mask.0,
        )?;
        this.set(CUBLASLT_MATMUL_PREF_IMPL_MASK, &pref.impl_mask.0)?;
        Ok(this)
    }

    pub fn raw(&self) -> sys::cublasLtMatmulPreference_t {
        self.0
    }

    fn set<T>(&self, attr: sys::cublasLtMatmulPreferenceAttributes_t, value: &T) -> Result<()> {
        check(
            unsafe {
                sys::cublasLtMatmulPreferenceSetAttribute(
                    self.0,
                    attr,
                    (value as *const T).cast(),
                    size_of::<T>(),
                )
            },
            "cublasLtMatmulPreferenceSetAttribute",
        )
    }
}

impl Drop for RawMatmulPreference {
    fn drop(&mut self) {
        unsafe {
            sys::cublasLtMatmulPreferenceDestroy(self.0);
        }
    }
}
