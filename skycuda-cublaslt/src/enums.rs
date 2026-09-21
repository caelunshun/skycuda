// --- LLM-generated --- //

//! Rust enums mirroring cuBLASLt's C enums.
//!
//! Values are written straight from `cublasLt.h`, so callers never see
//! `cudarc`'s CUDA-version-gated enum types. Most of them reach cuBLASLt as a
//! plain integer inside an attribute buffer; only [`DataType`] and
//! [`ComputeType`] are also passed as real FFI enums (by
//! `cublasLtMatrixLayoutCreate`, `cublasLtMatmulDescCreate` and friends), so
//! only those two carry a `to_ffi`.

use crate::sys;

macro_rules! ffi_enum {
    ($(#[$m:meta])* $name:ident { $($(#[$vm:meta])* $variant:ident = $val:expr),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u32)]
        pub enum $name {
            $($(#[$vm])* $variant = $val),+
        }

        impl $name {
            /// The value cuBLASLt uses for this variant.
            #[inline]
            pub fn raw(self) -> u32 {
                self as u32
            }

            /// The inverse of [`Self::raw`], or `None` for a value cuBLASLt
            /// gave us that this wrapper does not know.
            pub fn from_raw(v: u32) -> Option<Self> {
                match v {
                    $($val => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
}

ffi_enum! {
    /// Element type of a matrix, bias or scale (the subset of `cudaDataType_t`
    /// cuBLASLt accepts).
    DataType {
        F32 = 0,
        F64 = 1,
        F16 = 2,
        I8 = 3,
        U8 = 8,
        I32 = 10,
        BF16 = 14,
        F8E4M3 = 28,
        F8E5M2 = 29,
        /// Unsigned 8-bit power-of-two scale factor (`CUDA_R_8F_UE8M0`).
        F8UE8M0 = 30,
        F6E2M3 = 31,
        F6E3M2 = 32,
        F4E2M1 = 33,
        F8UE5M3 = 34,
    }
}

impl DataType {
    /// Width of one element in bits. Sub-byte types report fewer than 8.
    pub fn size_bits(self) -> usize {
        match self {
            DataType::F4E2M1 => 4,
            DataType::F6E2M3 | DataType::F6E3M2 => 6,
            DataType::I8
            | DataType::U8
            | DataType::F8E4M3
            | DataType::F8E5M2
            | DataType::F8UE8M0
            | DataType::F8UE5M3 => 8,
            DataType::F16 | DataType::BF16 => 16,
            DataType::F32 | DataType::I32 => 32,
            DataType::F64 => 64,
        }
    }

    pub(crate) fn to_ffi(self) -> sys::cudaDataType_t {
        use sys::cudaDataType_t::*;
        match self {
            DataType::F32 => CUDA_R_32F,
            DataType::F64 => CUDA_R_64F,
            DataType::F16 => CUDA_R_16F,
            DataType::I8 => CUDA_R_8I,
            DataType::U8 => CUDA_R_8U,
            DataType::I32 => CUDA_R_32I,
            DataType::BF16 => CUDA_R_16BF,
            DataType::F8E4M3 => CUDA_R_8F_E4M3,
            DataType::F8E5M2 => CUDA_R_8F_E5M2,
            DataType::F8UE8M0 => CUDA_R_8F_UE8M0,
            DataType::F6E2M3 => CUDA_R_6F_E2M3,
            DataType::F6E3M2 => CUDA_R_6F_E3M2,
            DataType::F4E2M1 => CUDA_R_4F_E2M1,
            DataType::F8UE5M3 => CUDA_R_8F_UE5M3,
        }
    }
}

ffi_enum! {
    /// Accumulator precision of the multiply-accumulate (`cublasComputeType_t`).
    ComputeType {
        F16 = 64,
        F16Pedantic = 65,
        F32 = 68,
        F32Pedantic = 69,
        F32Fast16F = 74,
        F32Fast16BF = 75,
        F32FastTF32 = 77,
        /// FP32 emulated by nine BF16 products.
        F32Emulated16BFx9 = 78,
        F64 = 70,
        F64Pedantic = 71,
        F64EmulatedFixedPoint = 79,
        I32 = 72,
        I32Pedantic = 73,
    }
}

impl ComputeType {
    /// The scale (alpha/beta) type cuBLASLt defaults to for this compute type.
    pub fn default_scale_type(self) -> DataType {
        match self {
            ComputeType::F16 | ComputeType::F16Pedantic => DataType::F16,
            ComputeType::F64 | ComputeType::F64Pedantic | ComputeType::F64EmulatedFixedPoint => {
                DataType::F64
            }
            ComputeType::I32 | ComputeType::I32Pedantic => DataType::I32,
            _ => DataType::F32,
        }
    }

    pub(crate) fn to_ffi(self) -> sys::cublasComputeType_t {
        use sys::cublasComputeType_t::*;
        match self {
            ComputeType::F16 => CUBLAS_COMPUTE_16F,
            ComputeType::F16Pedantic => CUBLAS_COMPUTE_16F_PEDANTIC,
            ComputeType::F32 => CUBLAS_COMPUTE_32F,
            ComputeType::F32Pedantic => CUBLAS_COMPUTE_32F_PEDANTIC,
            ComputeType::F32Fast16F => CUBLAS_COMPUTE_32F_FAST_16F,
            ComputeType::F32Fast16BF => CUBLAS_COMPUTE_32F_FAST_16BF,
            ComputeType::F32FastTF32 => CUBLAS_COMPUTE_32F_FAST_TF32,
            ComputeType::F32Emulated16BFx9 => CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
            ComputeType::F64 => CUBLAS_COMPUTE_64F,
            ComputeType::F64Pedantic => CUBLAS_COMPUTE_64F_PEDANTIC,
            ComputeType::F64EmulatedFixedPoint => CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT,
            ComputeType::I32 => CUBLAS_COMPUTE_32I,
            ComputeType::I32Pedantic => CUBLAS_COMPUTE_32I_PEDANTIC,
        }
    }
}

ffi_enum! {
    /// Memory order of a matrix (`cublasLtOrder_t`).
    #[allow(non_camel_case_types)]
    Order {
        /// Column-major: `ld` is the stride between columns.
        Col = 0,
        /// Row-major: `ld` is the stride between rows.
        Row = 1,
        Col32 = 2,
        Col4_4R2_8C = 3,
        Col32_2R_4R4 = 4,
    }
}

ffi_enum! {
    /// Whether an operand is used as-is, transposed or conjugate-transposed
    /// (`cublasOperation_t`).
    Operation {
        N = 0,
        T = 1,
        C = 2,
    }
}

/// Whether an epilogue reads or writes the auxiliary buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuxRole {
    /// The epilogue writes the buffer (ReLU bit-mask or GELU input).
    Output,
    /// The epilogue reads the buffer (backward pass).
    Input,
}

ffi_enum! {
    /// Fused post-processing applied to the matmul result (`cublasLtEpilogue_t`).
    Epilogue {
        Default = 1,
        Relu = 2,
        ReluAux = 130,
        Bias = 4,
        ReluBias = 6,
        ReluAuxBias = 134,
        DRelu = 136,
        DReluBGrad = 152,
        Gelu = 32,
        GeluAux = 160,
        GeluBias = 36,
        GeluAuxBias = 164,
        DGelu = 192,
        DGeluBGrad = 208,
        BGradA = 256,
        BGradB = 512,
    }
}

impl Epilogue {
    /// Whether the epilogue reads a bias vector.
    pub fn needs_bias(self) -> bool {
        matches!(
            self,
            Epilogue::Bias
                | Epilogue::ReluBias
                | Epilogue::ReluAuxBias
                | Epilogue::GeluBias
                | Epilogue::GeluAuxBias
        )
    }

    /// Whether the epilogue writes a bias gradient vector.
    pub fn writes_bias_grad(self) -> bool {
        matches!(
            self,
            Epilogue::DReluBGrad | Epilogue::DGeluBGrad | Epilogue::BGradA | Epilogue::BGradB
        )
    }

    /// How the epilogue uses the auxiliary buffer, if at all.
    pub fn aux_role(self) -> Option<AuxRole> {
        match self {
            Epilogue::ReluAux
            | Epilogue::ReluAuxBias
            | Epilogue::GeluAux
            | Epilogue::GeluAuxBias => Some(AuxRole::Output),
            Epilogue::DRelu | Epilogue::DReluBGrad | Epilogue::DGelu | Epilogue::DGeluBGrad => {
                Some(AuxRole::Input)
            }
            _ => None,
        }
    }

    /// Whether the auxiliary buffer is a ReLU bit-mask (one bit per element)
    /// rather than a matrix of `aux_data_type` elements.
    pub fn aux_is_bitmask(self) -> bool {
        matches!(
            self,
            Epilogue::ReluAux | Epilogue::ReluAuxBias | Epilogue::DRelu | Epilogue::DReluBGrad
        )
    }
}

ffi_enum! {
    /// How cuBLASLt interprets a matrix scaling factor
    /// (`cublasLtMatmulMatrixScale_t`).
    ScaleMode {
        /// A single f32 applied to the whole matrix.
        Scalar32F = 0,
        Vec16UE4M3 = 1,
        Vec32UE8M0 = 2,
        OuterVec32F = 3,
        Vec128_32F = 4,
        Blk128x128_32F = 5,
        /// One f32 per batch entry.
        PerBatchScalar32F = 6,
        Vec128MnK4UE8M0 = 12,
        Vec32MnK4UE8M0 = 13,
    }
}

ffi_enum! {
    /// Where cuBLASLt reads alpha and beta from (`cublasLtPointerMode_t`).
    PointerMode {
        Host = 0,
        Device = 1,
        DeviceVector = 2,
        AlphaDeviceVectorBetaZero = 3,
        AlphaDeviceVectorBetaHost = 4,
    }
}

ffi_enum! {
    /// How split-K partial results are accumulated
    /// (`cublasLtReductionScheme_t`).
    ReductionScheme {
        None = 0,
        Inplace = 1,
        ComputeType = 2,
        OutputType = 4,
    }
}

/// A set of [`ReductionScheme`]s, as a cuBLASLt bitmask.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReductionSchemeMask(pub u32);

impl ReductionSchemeMask {
    /// Every reduction scheme (`CUBLASLT_REDUCTION_SCHEME_MASK`).
    pub const ALL: Self = Self(7);
    /// No split-K reduction at all.
    pub const NONE: Self = Self(0);

    pub fn contains(self, scheme: ReductionScheme) -> bool {
        scheme == ReductionScheme::None || self.0 & scheme.raw() == scheme.raw()
    }
}

impl Default for ReductionSchemeMask {
    fn default() -> Self {
        Self::ALL
    }
}

/// Bitmask over the numerical implementation details an algorithm may use
/// (`cublasLtNumericalImplFlags_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NumericalImplFlags(pub u64);

impl NumericalImplFlags {
    /// Allow every implementation. This is cuBLASLt's default.
    pub const ALL: Self = Self(u64::MAX);

    pub const FMA: Self = Self(0x01);
    pub const HMMA: Self = Self(0x02);
    pub const IMMA: Self = Self(0x04);
    pub const DMMA: Self = Self(0x08);
    pub const TENSOR_OP_MASK: Self = Self(0xfe);
    pub const OP_TYPE_MASK: Self = Self(0xff);

    pub const ACCUMULATOR_16F: Self = Self(0x01 << 8);
    pub const ACCUMULATOR_32F: Self = Self(0x02 << 8);
    pub const ACCUMULATOR_64F: Self = Self(0x04 << 8);
    pub const ACCUMULATOR_32I: Self = Self(0x08 << 8);
    pub const ACCUMULATOR_TYPE_MASK: Self = Self(0xff << 8);

    pub const INPUT_16F: Self = Self(0x01 << 16);
    pub const INPUT_16BF: Self = Self(0x02 << 16);
    pub const INPUT_TF32: Self = Self(0x04 << 16);
    pub const INPUT_32F: Self = Self(0x08 << 16);
    pub const INPUT_64F: Self = Self(0x10 << 16);
    pub const INPUT_8I: Self = Self(0x20 << 16);
    pub const INPUT_8F_E4M3: Self = Self(0x40 << 16);
    pub const INPUT_8F_E5M2: Self = Self(0x80 << 16);
    pub const OP_INPUT_TYPE_MASK: Self = Self(0xff << 16);

    pub const GAUSSIAN: Self = Self(0x01 << 32);

    pub fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

impl Default for NumericalImplFlags {
    fn default() -> Self {
        Self::ALL
    }
}

impl std::ops::BitOr for NumericalImplFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

/// Tile shape id of an algorithm configuration (`cublasLtMatmulTile_t`).
///
/// The named values are an open-ended list that grows with every cuBLAS
/// release, so they are left opaque; read them back with
/// [`Algorithm::config`](crate::Algorithm::config).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TileId(pub u32);

impl TileId {
    pub const UNDEFINED: Self = Self(0);
}

/// Pipeline-stages id of an algorithm configuration (`cublasLtMatmulStages_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StagesId(pub u32);

impl StagesId {
    pub const UNDEFINED: Self = Self(0);
}

/// Inner (MMA) shape id of an algorithm configuration
/// (`cublasLtMatmulInnerShape_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InnerShapeId(pub u16);

impl InnerShapeId {
    pub const UNDEFINED: Self = Self(0);
    pub const MMA884: Self = Self(1);
    pub const MMA1684: Self = Self(2);
    pub const MMA1688: Self = Self(3);
    pub const MMA16816: Self = Self(4);
}

/// Thread-block cluster shape id of an algorithm configuration
/// (`cublasLtClusterShape_t`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClusterShapeId(pub u16);

impl ClusterShapeId {
    pub const AUTO: Self = Self(0);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epilogue_predicates() {
        assert!(Epilogue::GeluAuxBias.needs_bias());
        assert!(!Epilogue::GeluAux.needs_bias());
        assert!(Epilogue::DGeluBGrad.writes_bias_grad());
        assert!(!Epilogue::DGelu.writes_bias_grad());
        assert_eq!(Epilogue::GeluAux.aux_role(), Some(AuxRole::Output));
        assert_eq!(Epilogue::DGeluBGrad.aux_role(), Some(AuxRole::Input));
        assert_eq!(Epilogue::Bias.aux_role(), None);
        assert!(Epilogue::ReluAux.aux_is_bitmask());
        assert!(!Epilogue::GeluAux.aux_is_bitmask());
    }

    #[test]
    fn enum_round_trips() {
        assert_eq!(
            DataType::from_raw(DataType::BF16.raw()),
            Some(DataType::BF16)
        );
        assert_eq!(Epilogue::from_raw(512), Some(Epilogue::BGradB));
        assert_eq!(Order::from_raw(9), None);
        assert_eq!(ComputeType::F32.default_scale_type(), DataType::F32);
        assert_eq!(ComputeType::F16.default_scale_type(), DataType::F16);
    }
}
