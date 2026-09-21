// --- LLM-generated --- //

//! Safe, plan-then-execute bindings over cuBLASLt.
//!
//! [`CublasLtHandle::new`] binds a handle to a [`Device`]. A [`MatmulDesc`]
//! describes the problem — shapes, types, transposes, epilogue, FP8 scaling —
//! and [`CublasLtHandle::build_plan`] turns it into a [`MatmulPlan`], running
//! the heuristic once. Executing binds buffers through
//! [`MatmulPlan::operands`] and runs on a [`Stream`](skycuda::stream::Stream),
//! taking stream-ordered access through `skycuda`'s memory guards.
//!
//! Shape, length and alignment mistakes panic; cuBLASLt's own failures come
//! back as [`Error`].
//!
//! ```no_run
//! use skycuda::device::Device;
//! use skycuda_cublaslt::*;
//!
//! # fn main() -> Result<()> {
//! let device = Device::from_index(0)?;
//! let stream = device.create_stream()?;
//! let handle = CublasLtHandle::new(&device)?;
//!
//! // D = A * B, all column-major f32, 64x64x64.
//! let layout = MatrixLayout::new(DataType::F32, 64, 64, 64);
//! let desc = MatmulDesc::new(ComputeType::F32, layout, layout, layout);
//! let plan = handle.build_plan(&desc, &MatmulPreference::default())?;
//!
//! # let (a, b, mut d, mut workspace): (
//! #     skycuda::device_memory::DeviceMemory,
//! #     skycuda::device_memory::DeviceMemory,
//! #     skycuda::device_memory::DeviceMemory,
//! #     skycuda::device_memory::DeviceMemory,
//! # ) = todo!();
//! let mut operands = plan.operands();
//! operands.a(&a).b(&b).d(&mut d);
//! unsafe { plan.execute(&stream, &operands, &mut workspace)? };
//! # Ok(())
//! # }
//! ```

use cudarc::cublaslt::sys;
use skycuda::device::Device;
use std::{ptr, sync::Arc};

mod desc;
mod enums;
mod error;
mod layout;
#[cfg(feature = "tracing")]
mod logger;
mod plan;
mod scalar;

pub use desc::{MatmulDesc, MatmulPreference, Scales};
pub use enums::*;
use error::check;
pub use error::{Error, Result, Status};
pub use layout::MatrixLayout;
#[cfg(feature = "tracing")]
pub use logger::{LogLevel, install_tracing_logger};
pub use plan::{
    Algorithm, AlgorithmCaps, AlgorithmCheck, AlgorithmConfig, MatmulPlan, MatmulTypes, Operands,
};
pub use scalar::Scalar;

/// A device-bound cuBLASLt handle.
///
/// cuBLASLt handles are thread-safe, so this is cheap to clone and share; the
/// stream is chosen per call rather than baked into the handle.
#[derive(Debug, Clone)]
pub struct CublasLtHandle(Arc<Inner>);

#[derive(Debug)]
struct Inner {
    raw: sys::cublasLtHandle_t,
    device: Device,
}

impl CublasLtHandle {
    /// Creates a handle on `device`, loading `libcublasLt.so` on first use.
    pub fn new(device: &Device) -> Result<Self> {
        device.set_as_current()?;
        let mut raw = ptr::null_mut();
        check(unsafe { sys::cublasLtCreate(&mut raw) }, "cublasLtCreate")?;
        Ok(Self(Arc::new(Inner {
            raw,
            device: device.clone(),
        })))
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    pub fn raw(&self) -> sys::cublasLtHandle_t {
        self.0.raw
    }

    /// Describes the problem to cuBLASLt and runs the heuristic, producing a
    /// plan that can be executed many times.
    pub fn build_plan(&self, desc: &MatmulDesc, pref: &MatmulPreference) -> Result<MatmulPlan> {
        MatmulPlan::build(self, desc, pref)
    }

    /// Every algorithm id cuBLASLt has for this combination of types, for
    /// callers running their own search with [`Algorithm::init`].
    pub fn algo_ids(&self, types: &MatmulTypes) -> Result<Vec<i32>> {
        self.device().set_as_current()?;
        // cuBLASLt has no "how many?" query, so ask for more than it can have
        // and trust `returned`.
        let mut ids = vec![0i32; 256];
        let mut returned = 0i32;
        check(
            unsafe {
                sys::cublasLtMatmulAlgoGetIds(
                    self.raw(),
                    types.compute.to_ffi(),
                    types.scale.to_ffi(),
                    types.a.to_ffi(),
                    types.b.to_ffi(),
                    types.c.to_ffi(),
                    types.d.to_ffi(),
                    ids.len() as i32,
                    ids.as_mut_ptr(),
                    &mut returned,
                )
            },
            "cublasLtMatmulAlgoGetIds",
        )?;
        ids.truncate(returned.max(0) as usize);
        Ok(ids)
    }
}

impl PartialEq for CublasLtHandle {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for CublasLtHandle {}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            sys::cublasLtDestroy(self.raw);
        }
    }
}

// SAFETY: cuBLASLt documents its handles as thread-safe.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// The loaded cuBLASLt library version, e.g. `130400`.
pub fn version() -> usize {
    unsafe { sys::cublasLtGetVersion() }
}

/// The CUDA runtime version cuBLASLt was built against.
pub fn cudart_version() -> usize {
    unsafe { sys::cublasLtGetCudartVersion() }
}
