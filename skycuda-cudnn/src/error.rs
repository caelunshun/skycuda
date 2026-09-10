//! Error type mapping the shim's status codes + thread-local message.

use std::ffi::CStr;
use std::fmt;

use cudnn_frontend_sys as sys;

/// Error codes mirroring `cudnn_frontend::error_code_t` (plus a shim-internal
/// `ShimException` for C++ exceptions that escaped the frontend).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ErrorCode {
    AttributeNotSet = 1,
    ShapeDeductionFailed = 2,
    InvalidTensorName = 3,
    InvalidVariantPack = 4,
    GraphNotSupported = 5,
    GraphExecutionPlanCreationFailed = 6,
    GraphExecutionFailed = 7,
    HeuristicQueryFailed = 8,
    UnsupportedGraphFormat = 9,
    CudaApiFailed = 10,
    CudnnBackendApiFailed = 11,
    InvalidCudaDevice = 12,
    HandleError = 13,
    InvalidValue = 14,
    NvrtcCompilationFailed = 15,
    ShimException = 1000,
    /// A status code the wrapper did not recognize.
    Unknown = -1,
}

impl ErrorCode {
    fn from_raw(code: i32) -> ErrorCode {
        match code {
            1 => ErrorCode::AttributeNotSet,
            2 => ErrorCode::ShapeDeductionFailed,
            3 => ErrorCode::InvalidTensorName,
            4 => ErrorCode::InvalidVariantPack,
            5 => ErrorCode::GraphNotSupported,
            6 => ErrorCode::GraphExecutionPlanCreationFailed,
            7 => ErrorCode::GraphExecutionFailed,
            8 => ErrorCode::HeuristicQueryFailed,
            9 => ErrorCode::UnsupportedGraphFormat,
            10 => ErrorCode::CudaApiFailed,
            11 => ErrorCode::CudnnBackendApiFailed,
            12 => ErrorCode::InvalidCudaDevice,
            13 => ErrorCode::HandleError,
            14 => ErrorCode::InvalidValue,
            15 => ErrorCode::NvrtcCompilationFailed,
            1000 => ErrorCode::ShimException,
            _ => ErrorCode::Unknown,
        }
    }
}

/// An error returned by a fallible cudnn-frontend call.
#[derive(Debug, Clone)]
pub struct Error {
    pub code: ErrorCode,
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cudnn-frontend error {:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Reads the shim's thread-local error message.
fn last_message() -> String {
    // SAFETY: the shim returns a pointer to thread-local storage valid until the
    // next shim call on this thread; we copy it immediately.
    unsafe {
        let p = sys::cudnn_fe_last_error_message();
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    }
}

/// Turns a raw shim status into a `Result`, capturing the thread-local message.
pub(crate) fn check(status: i32) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(Error {
            code: ErrorCode::from_raw(status),
            message: last_message(),
        })
    }
}
