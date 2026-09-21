// --- LLM-generated --- //

//! Error type mapping `cublasStatus_t` plus the failing FFI entry point.

use crate::sys;
use std::ffi::CStr;

/// Status codes returned by cuBLASLt (`cublasStatus_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum Status {
    NotInitialized = 1,
    AllocFailed = 3,
    InvalidValue = 7,
    ArchMismatch = 8,
    MappingError = 11,
    ExecutionFailed = 13,
    InternalError = 14,
    NotSupported = 15,
    LicenseError = 16,
    /// A status code this wrapper did not recognize.
    Unknown = u32::MAX,
}

impl Status {
    /// Maps a raw status. `CUBLAS_STATUS_SUCCESS` maps to [`Status::Unknown`];
    /// callers are expected to have handled success already.
    pub(crate) fn from_ffi(status: sys::cublasStatus_t) -> Self {
        use sys::cublasStatus_t::*;
        match status {
            CUBLAS_STATUS_SUCCESS => Status::Unknown,
            CUBLAS_STATUS_NOT_INITIALIZED => Status::NotInitialized,
            CUBLAS_STATUS_ALLOC_FAILED => Status::AllocFailed,
            CUBLAS_STATUS_INVALID_VALUE => Status::InvalidValue,
            CUBLAS_STATUS_ARCH_MISMATCH => Status::ArchMismatch,
            CUBLAS_STATUS_MAPPING_ERROR => Status::MappingError,
            CUBLAS_STATUS_EXECUTION_FAILED => Status::ExecutionFailed,
            CUBLAS_STATUS_INTERNAL_ERROR => Status::InternalError,
            CUBLAS_STATUS_NOT_SUPPORTED => Status::NotSupported,
            CUBLAS_STATUS_LICENSE_ERROR => Status::LicenseError,
        }
    }

    fn to_ffi(self) -> Option<sys::cublasStatus_t> {
        use sys::cublasStatus_t::*;
        Some(match self {
            Status::NotInitialized => CUBLAS_STATUS_NOT_INITIALIZED,
            Status::AllocFailed => CUBLAS_STATUS_ALLOC_FAILED,
            Status::InvalidValue => CUBLAS_STATUS_INVALID_VALUE,
            Status::ArchMismatch => CUBLAS_STATUS_ARCH_MISMATCH,
            Status::MappingError => CUBLAS_STATUS_MAPPING_ERROR,
            Status::ExecutionFailed => CUBLAS_STATUS_EXECUTION_FAILED,
            Status::InternalError => CUBLAS_STATUS_INTERNAL_ERROR,
            Status::NotSupported => CUBLAS_STATUS_NOT_SUPPORTED,
            Status::LicenseError => CUBLAS_STATUS_LICENSE_ERROR,
            Status::Unknown => return None,
        })
    }

    /// cuBLASLt's own name for the status, e.g. `CUBLAS_STATUS_NOT_SUPPORTED`.
    pub fn name(self) -> String {
        match self.to_ffi() {
            Some(s) => c_string(unsafe { sys::cublasLtGetStatusName(s) }),
            None => "CUBLAS_STATUS_UNKNOWN".to_owned(),
        }
    }

    /// cuBLASLt's own human-readable description of the status.
    pub fn message(self) -> String {
        match self.to_ffi() {
            Some(s) => c_string(unsafe { sys::cublasLtGetStatusString(s) }),
            None => "unrecognized cuBLASLt status".to_owned(),
        }
    }
}

/// An error returned by a fallible cuBLASLt call.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cuBLASLt {status:?} ({name}) during {context}: {message}")]
    Cublas {
        status: Status,
        name: String,
        message: String,
        /// The FFI entry point that failed, so heuristic failures and execution
        /// failures are distinguishable.
        context: &'static str,
    },
    #[error(transparent)]
    Cuda(#[from] skycuda::error::Error),
}

impl Error {
    pub(crate) fn new(status: Status, context: &'static str) -> Self {
        Error::Cublas {
            status,
            name: status.name(),
            message: status.message(),
            context,
        }
    }

    /// Builds an error with a message of this crate's own making rather than
    /// cuBLASLt's.
    pub(crate) fn with_message(
        status: Status,
        context: &'static str,
        message: impl Into<String>,
    ) -> Self {
        Error::Cublas {
            status,
            name: status.name(),
            message: message.into(),
            context,
        }
    }

    /// The cuBLASLt status, if this is a cuBLASLt error.
    pub fn status(&self) -> Option<Status> {
        match self {
            Error::Cublas { status, .. } => Some(*status),
            Error::Cuda(_) => None,
        }
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Turns a raw status into a `Result`, tagging failures with the FFI call name.
pub(crate) fn check(status: sys::cublasStatus_t, context: &'static str) -> Result<()> {
    if status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::new(Status::from_ffi(status), context))
    }
}

fn c_string(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: cuBLASLt returns pointers to static string literals.
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}
