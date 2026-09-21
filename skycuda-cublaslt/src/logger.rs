// --- LLM-generated --- //

//! Bridges cuBLASLt's logger onto `tracing`. Enabled by the `tracing` feature.

use crate::{
    error::{Result, check},
    sys,
};
use std::{
    ffi::{CStr, c_char, c_int},
    sync::Once,
};

/// cuBLASLt's log verbosity levels, each including the ones before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum LogLevel {
    Off = 0,
    Error = 1,
    Trace = 2,
    Hints = 3,
    Heuristics = 4,
    ApiTrace = 5,
}

/// Routes cuBLASLt's log messages to `tracing` and sets its verbosity.
///
/// This is process-global state in cuBLASLt. The callback is installed once;
/// later calls only update the level.
pub fn install_tracing_logger(level: LogLevel) -> Result<()> {
    static INSTALLED: Once = Once::new();
    let mut result = Ok(());
    INSTALLED.call_once(|| {
        result = check(
            unsafe { sys::cublasLtLoggerSetCallback(Some(callback)) },
            "cublasLtLoggerSetCallback",
        );
    });
    result?;
    check(
        unsafe { sys::cublasLtLoggerSetLevel(level as c_int) },
        "cublasLtLoggerSetLevel",
    )
}

unsafe extern "C" fn callback(level: c_int, function_name: *const c_char, message: *const c_char) {
    let read = |p: *const c_char| {
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    };
    let function = read(function_name);
    let message = read(message);
    match level {
        1 => tracing::error!(target: "cublaslt", function, "{message}"),
        2 | 3 => tracing::debug!(target: "cublaslt", function, "{message}"),
        _ => tracing::trace!(target: "cublaslt", function, "{message}"),
    }
}
