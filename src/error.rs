use std::num::TryFromIntError;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    CudaDriver(#[from] cudarc::driver::result::DriverError),
    #[error(transparent)]
    IntConversion(#[from] TryFromIntError),
    #[error(transparent)]
    StringNul(#[from] std::ffi::NulError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
