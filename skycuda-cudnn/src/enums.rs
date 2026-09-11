//! Rust enums mirroring the shim's C enums. Values match the underlying
//! `cudnn_frontend` scoped enums (parity is `static_assert`'d in the shim), so
//! each converts to the FFI type by a plain cast.

macro_rules! ffi_enum {
    ($(#[$m:meta])* $name:ident : $ffi:ty { $($variant:ident = $val:expr),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(i32)]
        pub enum $name {
            $($variant = $val),+
        }
        impl $name {
            #[inline]
            pub(crate) fn to_ffi(self) -> $ffi {
                self as i32 as $ffi
            }
        }
    };
}

use cudnn_frontend_sys as sys;

ffi_enum! {
    /// Element data type (`DataType_t`).
    DataType: sys::cudnnFeDataType {
        NotSet = 0, Float = 1, Double = 2, Half = 3, Int8 = 4, Int32 = 5,
        Int8x4 = 6, Uint8 = 7, Uint8x4 = 8, Int8x32 = 9, Bfloat16 = 10,
        Int64 = 11, Boolean = 12, ByteBoolean = 13, Fp8E4m3 = 14, Fp8E5m2 = 15,
        FastFloatForFp8 = 16, Fp8E8m0 = 17, Fp8E5m3 = 18, Fp4E2m1 = 19,
        Int4 = 20, ComplexFp32 = 21, ComplexFp64 = 22,
    }
}

ffi_enum! {
    /// Heuristic mode (`HeurMode_t`; `OPENSOURCE` is intentionally excluded).
    HeurMode: sys::cudnnFeHeurMode {
        A = 0, B = 1, Fallback = 2,
    }
}

ffi_enum! {
    /// Plan build policy (`BuildPlanPolicy_t`).
    BuildPlanPolicy: sys::cudnnFeBuildPlanPolicy {
        HeuristicsChoice = 0, All = 1,
    }
}

ffi_enum! {
    /// Pointwise op mode (`PointwiseMode_t`).
    PointwiseMode: sys::cudnnFePointwiseMode {
        NotSet = 0, Add = 1, Mul = 2, Sqrt = 3, Max = 4, Min = 5, ReluFwd = 6,
        TanhFwd = 7, SigmoidFwd = 8, EluFwd = 9, GeluFwd = 10, SoftplusFwd = 11,
        SwishFwd = 12, ReluBwd = 13, TanhBwd = 14, SigmoidBwd = 15, EluBwd = 16,
        GeluBwd = 17, SoftplusBwd = 18, SwishBwd = 19, Erf = 20, Identity = 21,
        GeluApproxTanhBwd = 22, GeluApproxTanhFwd = 23, GenIndex = 24,
        BinarySelect = 25, Exp = 26, Log = 27, Neg = 28, Mod = 29, Pow = 30,
        Abs = 31, Ceil = 32, Cos = 33, Floor = 34, Rsqrt = 35, Sin = 36,
        LogicalNot = 37, Tan = 38, Sub = 39, AddSquare = 40, Div = 41,
        CmpEq = 42, CmpNeq = 43, CmpGt = 44, CmpGe = 45, CmpLt = 46, CmpLe = 47,
        LogicalAnd = 48, LogicalOr = 49, Reciprocal = 50,
    }
}

ffi_enum! {
    /// Reduction mode (`ReductionMode_t`).
    ReductionMode: sys::cudnnFeReductionMode {
        NotSet = 0, Add = 1, Mul = 2, Min = 3, Max = 4, Amax = 5, Avg = 6,
        Norm1 = 7, Norm2 = 8, MulNoZeros = 9,
    }
}

ffi_enum! {
    /// Normalization forward phase (`NormFwdPhase_t`).
    NormFwdPhase: sys::cudnnFeNormFwdPhase {
        NotSet = 0, Inference = 1, Training = 2,
    }
}

ffi_enum! {
    /// Convolution mode (`ConvolutionMode_t`).
    ConvolutionMode: sys::cudnnFeConvolutionMode {
        NotSet = 0, Convolution = 1, CrossCorrelation = 2,
    }
}

ffi_enum! {
    /// Resample mode (`ResampleMode_t`).
    ResampleMode: sys::cudnnFeResampleMode {
        NotSet = 0, AvgpoolExcludePadding = 1, AvgpoolIncludePadding = 2,
        Bilinear = 3, Nearest = 4, Maxpool = 5,
    }
}

ffi_enum! {
    /// Resample padding mode (`PaddingMode_t`).
    PaddingMode: sys::cudnnFePaddingMode {
        NotSet = 0, EdgeValPad = 1, NegInfPad = 2, ZeroPad = 3,
    }
}

ffi_enum! {
    /// Reshape mode (`ReshapeMode_t`).
    ReshapeMode: sys::cudnnFeReshapeMode {
        NotSet = 0, ViewOnly = 1, Logical = 2,
    }
}

ffi_enum! {
    /// Tensor reordering (`TensorReordering_t`).
    TensorReordering: sys::cudnnFeTensorReordering {
        None = 0, Int8x32 = 1, F16x16 = 2, F8_128x4 = 3,
    }
}

ffi_enum! {
    /// SDPA diagonal alignment (`DiagonalAlignment_t`).
    DiagonalAlignment: sys::cudnnFeDiagonalAlignment {
        TopLeft = 0, BottomRight = 1,
    }
}

ffi_enum! {
    /// RNG distribution (`RngDistribution_t`).
    RngDistribution: sys::cudnnFeRngDistribution {
        NotSet = 0, Bernoulli = 1, Uniform = 2, Normal = 3,
    }
}

ffi_enum! {
    /// MoE grouped-matmul mode (`MoeGroupedMatmulMode_t`).
    MoeGroupedMatmulMode: sys::cudnnFeMoeGroupedMatmulMode {
        NotSet = 0, None = 1, Gather = 2, Scatter = 3,
    }
}

ffi_enum! {
    /// Attention implementation selector (`AttentionImplementation_t`).
    AttentionImplementation: sys::cudnnFeAttentionImplementation {
        Auto = 0, Composite = 1, Unified = 2,
    }
}
