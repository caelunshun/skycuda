// --- LLM-generated --- //

//! Host-side alpha/beta values, typed by the descriptor's scale type.

use crate::enums::DataType;
use half::{bf16, f16};
use std::ffi::c_void;

/// An alpha or beta value. Its variant must match
/// [`MatmulDesc::scale_type`](crate::MatmulDesc::scale_type).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    F32(f32),
    F64(f64),
    I32(i32),
    F16(f16),
    BF16(bf16),
}

impl Scalar {
    /// The value `1` in the given scale type.
    ///
    /// # Panics
    /// If `scale_type` is not one of the types [`Scalar`] can represent.
    pub fn one(scale_type: DataType) -> Self {
        match scale_type {
            DataType::F32 => Scalar::F32(1.0),
            DataType::F64 => Scalar::F64(1.0),
            DataType::I32 => Scalar::I32(1),
            DataType::F16 => Scalar::F16(f16::ONE),
            DataType::BF16 => Scalar::BF16(bf16::ONE),
            other => panic!("{other:?} is not a valid scale type"),
        }
    }

    /// The value `0` in the given scale type.
    ///
    /// # Panics
    /// If `scale_type` is not one of the types [`Scalar`] can represent.
    pub fn zero(scale_type: DataType) -> Self {
        match scale_type {
            DataType::F32 => Scalar::F32(0.0),
            DataType::F64 => Scalar::F64(0.0),
            DataType::I32 => Scalar::I32(0),
            DataType::F16 => Scalar::F16(f16::ZERO),
            DataType::BF16 => Scalar::BF16(bf16::ZERO),
            other => panic!("{other:?} is not a valid scale type"),
        }
    }

    pub fn data_type(&self) -> DataType {
        match self {
            Scalar::F32(_) => DataType::F32,
            Scalar::F64(_) => DataType::F64,
            Scalar::I32(_) => DataType::I32,
            Scalar::F16(_) => DataType::F16,
            Scalar::BF16(_) => DataType::BF16,
        }
    }

    /// Pointer to the value, valid as long as `self` is borrowed.
    pub(crate) fn as_ptr(&self) -> *const c_void {
        match self {
            Scalar::F32(v) => (v as *const f32).cast(),
            Scalar::F64(v) => (v as *const f64).cast(),
            Scalar::I32(v) => (v as *const i32).cast(),
            Scalar::F16(v) => (v as *const f16).cast(),
            Scalar::BF16(v) => (v as *const bf16).cast(),
        }
    }
}
