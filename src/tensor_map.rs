// --- LLM-generated --- //

//! Tiled TMA tensor maps (`CUtensorMap`) bound to device-visible memory.

use crate::{
    device_memory::{
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
    error::Result,
    launch::KernelArg,
    stream::Stream,
};
use bytemuck::{Pod, Zeroable};
use cudarc::driver::sys::{
    CUdeviceptr_v2, CUtensorMap, CUtensorMapDataType_enum, CUtensorMapFloatOOBfill_enum,
    CUtensorMapInterleave_enum, CUtensorMapL2promotion_enum, CUtensorMapSwizzle_enum,
    cuTensorMapEncodeTiled,
};

pub const MAX_RANK: usize = 5;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TensorMapDataType {
    U8,
    U16,
    U32,
    I32,
    U64,
    I64,
    F16,
    F32,
    F64,
    BF16,
    F32Ftz,
    Tf32,
    Tf32Ftz,
    /// 16 packed 4-bit values per 8 bytes, no gaps.
    U4x16Align8B,
    /// 16 packed 4-bit values per 16 bytes (8 data bytes, 8 gap bytes).
    U4x16Align16B,
    /// 16 packed 6-bit values per 16 bytes (12 data bytes, 4 gap bytes).
    U6x16Align16B,
}

impl TensorMapDataType {
    /// Size of one element in bytes, or `None` for the packed sub-byte types.
    pub fn size_bytes(self) -> Option<u64> {
        use TensorMapDataType::*;
        match self {
            U8 => Some(1),
            U16 | F16 | BF16 => Some(2),
            U32 | I32 | F32 | F32Ftz | Tf32 | Tf32Ftz => Some(4),
            U64 | I64 | F64 => Some(8),
            U4x16Align8B | U4x16Align16B | U6x16Align16B => None,
        }
    }

    fn is_packed_align16(self) -> bool {
        matches!(
            self,
            TensorMapDataType::U4x16Align16B | TensorMapDataType::U6x16Align16B
        )
    }

    fn to_ffi(self) -> CUtensorMapDataType_enum {
        use CUtensorMapDataType_enum::*;
        use TensorMapDataType::*;
        match self {
            U8 => CU_TENSOR_MAP_DATA_TYPE_UINT8,
            U16 => CU_TENSOR_MAP_DATA_TYPE_UINT16,
            U32 => CU_TENSOR_MAP_DATA_TYPE_UINT32,
            I32 => CU_TENSOR_MAP_DATA_TYPE_INT32,
            U64 => CU_TENSOR_MAP_DATA_TYPE_UINT64,
            I64 => CU_TENSOR_MAP_DATA_TYPE_INT64,
            F16 => CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
            F32 => CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            F64 => CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
            BF16 => CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            F32Ftz => CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ,
            Tf32 => CU_TENSOR_MAP_DATA_TYPE_TFLOAT32,
            Tf32Ftz => CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ,
            U4x16Align8B => CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
            U4x16Align16B => CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B,
            U6x16Align16B => CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum TensorMapInterleave {
    #[default]
    None,
    B16,
    B32,
}

impl TensorMapInterleave {
    fn to_ffi(self) -> CUtensorMapInterleave_enum {
        use CUtensorMapInterleave_enum::*;
        match self {
            TensorMapInterleave::None => CU_TENSOR_MAP_INTERLEAVE_NONE,
            TensorMapInterleave::B16 => CU_TENSOR_MAP_INTERLEAVE_16B,
            TensorMapInterleave::B32 => CU_TENSOR_MAP_INTERLEAVE_32B,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum TensorMapSwizzle {
    #[default]
    None,
    B32,
    B64,
    B128,
    B128Atom32B,
    B128Atom32BFlip8B,
    B128Atom64B,
}

impl TensorMapSwizzle {
    /// Swizzle span in bytes, which bounds the inner box dimension.
    pub fn span_bytes(self) -> Option<u64> {
        use TensorMapSwizzle::*;
        match self {
            None => Option::None,
            B32 => Some(32),
            B64 => Some(64),
            B128 | B128Atom32B | B128Atom32BFlip8B | B128Atom64B => Some(128),
        }
    }

    fn to_ffi(self) -> CUtensorMapSwizzle_enum {
        use CUtensorMapSwizzle_enum::*;
        use TensorMapSwizzle::*;
        match self {
            None => CU_TENSOR_MAP_SWIZZLE_NONE,
            B32 => CU_TENSOR_MAP_SWIZZLE_32B,
            B64 => CU_TENSOR_MAP_SWIZZLE_64B,
            B128 => CU_TENSOR_MAP_SWIZZLE_128B,
            B128Atom32B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
            B128Atom32BFlip8B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
            B128Atom64B => CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum TensorMapL2Promotion {
    #[default]
    None,
    B64,
    B128,
    B256,
}

impl TensorMapL2Promotion {
    fn to_ffi(self) -> CUtensorMapL2promotion_enum {
        use CUtensorMapL2promotion_enum::*;
        match self {
            TensorMapL2Promotion::None => CU_TENSOR_MAP_L2_PROMOTION_NONE,
            TensorMapL2Promotion::B64 => CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
            TensorMapL2Promotion::B128 => CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            TensorMapL2Promotion::B256 => CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum TensorMapOobFill {
    #[default]
    Zero,
    /// Fill with a NaN that requests zero in FMA. Floating-point, non-packed types only.
    NanRequestZeroFma,
}

impl TensorMapOobFill {
    fn to_ffi(self) -> CUtensorMapFloatOOBfill_enum {
        use CUtensorMapFloatOOBfill_enum::*;
        match self {
            TensorMapOobFill::Zero => CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            TensorMapOobFill::NanRequestZeroFma => {
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
            }
        }
    }
}

/// Description of a tiled tensor map, independent of the memory it addresses.
///
/// Dimension 0 is the innermost (contiguous) dimension. `global_strides[i]` is
/// the byte stride of dimension `i + 1`.
#[derive(Debug, Clone)]
pub struct TensorMapDesc {
    data_type: TensorMapDataType,
    rank: usize,
    global_dims: [u64; MAX_RANK],
    global_strides: [u64; MAX_RANK - 1],
    box_dims: [u32; MAX_RANK],
    element_strides: [u32; MAX_RANK],
    interleave: TensorMapInterleave,
    swizzle: TensorMapSwizzle,
    l2_promotion: TensorMapL2Promotion,
    oob_fill: TensorMapOobFill,
}

impl TensorMapDesc {
    pub fn new(
        data_type: TensorMapDataType,
        global_dims: &[u64],
        global_strides: &[u64],
        box_dims: &[u32],
    ) -> Self {
        let rank = global_dims.len();
        assert!(
            (1..=MAX_RANK).contains(&rank),
            "tensor map rank must be in 1..={MAX_RANK}, got {rank}"
        );
        assert_eq!(
            global_strides.len(),
            rank - 1,
            "tensor map of rank {rank} needs exactly {} global strides",
            rank - 1
        );
        assert_eq!(
            box_dims.len(),
            rank,
            "tensor map of rank {rank} needs exactly {rank} box dims"
        );

        let mut desc = Self {
            data_type,
            rank,
            global_dims: [1; MAX_RANK],
            global_strides: [0; MAX_RANK - 1],
            box_dims: [1; MAX_RANK],
            element_strides: [1; MAX_RANK],
            interleave: TensorMapInterleave::default(),
            swizzle: TensorMapSwizzle::default(),
            l2_promotion: TensorMapL2Promotion::default(),
            oob_fill: TensorMapOobFill::default(),
        };
        desc.global_dims[..rank].copy_from_slice(global_dims);
        desc.global_strides[..rank - 1].copy_from_slice(global_strides);
        desc.box_dims[..rank].copy_from_slice(box_dims);
        desc
    }

    #[must_use]
    pub fn element_strides(mut self, element_strides: &[u32]) -> Self {
        assert_eq!(
            element_strides.len(),
            self.rank,
            "tensor map of rank {} needs exactly {} element strides",
            self.rank,
            self.rank
        );
        self.element_strides[..self.rank].copy_from_slice(element_strides);
        self
    }

    #[must_use]
    pub fn interleave(mut self, interleave: TensorMapInterleave) -> Self {
        self.interleave = interleave;
        self
    }

    #[must_use]
    pub fn swizzle(mut self, swizzle: TensorMapSwizzle) -> Self {
        self.swizzle = swizzle;
        self
    }

    #[must_use]
    pub fn l2_promotion(mut self, l2_promotion: TensorMapL2Promotion) -> Self {
        self.l2_promotion = l2_promotion;
        self
    }

    #[must_use]
    pub fn oob_fill(mut self, oob_fill: TensorMapOobFill) -> Self {
        self.oob_fill = oob_fill;
        self
    }

    pub fn data_type(&self) -> TensorMapDataType {
        self.data_type
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn global_dims(&self) -> &[u64] {
        &self.global_dims[..self.rank]
    }

    pub fn global_strides(&self) -> &[u64] {
        &self.global_strides[..self.rank - 1]
    }

    pub fn box_dims(&self) -> &[u32] {
        &self.box_dims[..self.rank]
    }

    /// Required alignment of the global address and the global strides.
    pub fn alignment(&self) -> u64 {
        if self.interleave == TensorMapInterleave::B32 || self.data_type.is_packed_align16() {
            32
        } else {
            16
        }
    }

    /// Number of bytes from the base address to one past the last byte
    /// the tensor map can address.
    pub fn footprint_bytes(&self) -> usize {
        let dim0 = self.global_dims[0];
        let mut row0_bytes = match self.data_type {
            TensorMapDataType::U4x16Align8B => dim0.div_ceil(2),
            TensorMapDataType::U4x16Align16B | TensorMapDataType::U6x16Align16B => {
                dim0.div_ceil(16) * 16
            }
            ty => dim0
                .checked_mul(ty.size_bytes().unwrap())
                .expect("tensor map footprint overflow"),
        };
        match self.interleave {
            TensorMapInterleave::None => {}
            TensorMapInterleave::B16 => row0_bytes = row0_bytes.next_multiple_of(16),
            TensorMapInterleave::B32 => row0_bytes = row0_bytes.next_multiple_of(32),
        }

        let footprint = self
            .global_dims()
            .iter()
            .skip(1)
            .zip(self.global_strides())
            .try_fold(row0_bytes, |acc, (&dim, &stride)| {
                acc.checked_add(dim.checked_sub(1)?.checked_mul(stride)?)
            })
            .expect("tensor map footprint overflow");
        footprint.try_into().expect("tensor map footprint overflow")
    }

    fn validate(&self) {
        let rank = self.rank;
        if self.interleave != TensorMapInterleave::None {
            assert!(rank >= 3, "interleaved tensor maps must have rank >= 3");
        }
        for (i, &dim) in self.global_dims().iter().enumerate() {
            assert!(
                (1..=1 << 32).contains(&dim),
                "tensor map global dim {i} must be in 1..=2^32, got {dim}"
            );
        }
        let alignment = self.alignment();
        for (i, &stride) in self.global_strides().iter().enumerate() {
            assert!(
                stride.is_multiple_of(alignment) && stride < 1 << 40,
                "tensor map global stride {i} must be a multiple of {alignment} and less than 2^40, got {stride}"
            );
        }
        for (i, &dim) in self.box_dims().iter().enumerate() {
            assert!(
                (1..=256).contains(&dim),
                "tensor map box dim {i} must be in 1..=256, got {dim}"
            );
        }
        for (i, &stride) in self.element_strides[..rank].iter().enumerate() {
            assert!(
                (1..=8).contains(&stride),
                "tensor map element stride {i} must be in 1..=8, got {stride}"
            );
        }
        if self.interleave == TensorMapInterleave::B32 {
            assert_eq!(
                self.swizzle,
                TensorMapSwizzle::B32,
                "32B-interleaved tensor maps require 32B swizzle"
            );
        }

        match self.data_type {
            TensorMapDataType::U4x16Align8B => assert!(
                self.global_dims[0].is_multiple_of(2),
                "U4x16Align8B tensor maps require global dim 0 to be a multiple of 2"
            ),
            TensorMapDataType::U4x16Align16B | TensorMapDataType::U6x16Align16B => {
                assert!(
                    self.global_dims[0].is_multiple_of(128),
                    "{:?} tensor maps require global dim 0 to be a multiple of 128",
                    self.data_type
                );
                assert_eq!(
                    self.box_dims[0], 128,
                    "{:?} tensor maps require box dim 0 to be 128",
                    self.data_type
                );
            }
            _ => {}
        }

        if let Some(elem_bytes) = self.data_type.size_bytes()
            && self.interleave == TensorMapInterleave::None
        {
            let inner_bytes = u64::from(self.box_dims[0]) * elem_bytes;
            assert!(
                inner_bytes.is_multiple_of(16),
                "tensor map inner box dimension must be a multiple of 16 bytes, got {inner_bytes}"
            );
            if let Some(span) = self.swizzle.span_bytes() {
                assert!(
                    inner_bytes <= span,
                    "tensor map inner box dimension ({inner_bytes} bytes) exceeds swizzle span ({span} bytes)"
                );
            }
        }
    }

    /// Encodes a tensor map for the given global address without checking
    /// it against any memory allocation. Prefer [`TensorMap`].
    pub fn encode(&self, address: CUdeviceptr_v2) -> Result<TensorMapRepr> {
        let mut repr = TensorMapRepr::zeroed();
        unsafe {
            cuTensorMapEncodeTiled(
                &mut repr.0,
                self.data_type.to_ffi(),
                self.rank as u32,
                address as *mut _,
                self.global_dims.as_ptr(),
                self.global_strides.as_ptr(),
                self.box_dims.as_ptr(),
                self.element_strides.as_ptr(),
                self.interleave.to_ffi(),
                self.swizzle.to_ffi(),
                self.l2_promotion.to_ffi(),
                self.oob_fill.to_ffi(),
            )
            .result()?;
        }
        Ok(repr)
    }

    fn encode_checked(&self, address: CUdeviceptr_v2) -> Result<TensorMapRepr> {
        let alignment = self.alignment();
        assert!(
            address.is_multiple_of(alignment),
            "tensor map global address {address:#x} is not {alignment}-byte aligned"
        );
        self.encode(address)
    }
}

/// Kernel-parameter representation of a tensor map.
#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct TensorMapRepr(pub CUtensorMap);

const _: () = assert!(size_of::<TensorMapRepr>() == 128);

// SAFETY: `CUtensorMap` is a `[u64; 16]` with no padding.
unsafe impl Zeroable for TensorMapRepr {}
unsafe impl Pod for TensorMapRepr {}

/// A tiled tensor map bound to device-visible memory.
///
/// Passing `&TensorMap` to [`LaunchBuilder::arg`](crate::launch::LaunchBuilder::arg)
/// grants read access (TMA loads); passing `&mut TensorMap` grants write access
/// (TMA stores and reductions). The descriptor is re-encoded with the
/// memory's address on every launch. The kernel parameter must be declared as
/// `const __grid_constant__ CUtensorMap`.
#[derive(Debug)]
pub struct TensorMap<M> {
    memory: M,
    desc: TensorMapDesc,
}

impl<M> TensorMap<M>
where
    M: DeviceVisibleMemory,
{
    /// Panics if the descriptor is invalid or addresses bytes beyond `memory`.
    pub fn new(memory: M, desc: TensorMapDesc) -> Self {
        desc.validate();
        let footprint = desc.footprint_bytes();
        assert!(
            footprint <= memory.len(),
            "tensor map addresses {footprint} bytes, but memory is only {} bytes",
            memory.len()
        );
        Self { memory, desc }
    }

    pub fn desc(&self) -> &TensorMapDesc {
        &self.desc
    }

    pub fn into_inner(self) -> M {
        self.memory
    }

    pub fn inner(&self) -> &M {
        &self.memory
    }

    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.memory
    }
}

impl<'a, M> KernelArg<'a> for &'a TensorMap<M>
where
    M: DeviceVisibleMemory + 'a,
{
    type Repr = TensorMapRepr;
    type Guard = M::Guard<'a>;

    unsafe fn into_arg(self, stream: &'a Stream) -> Result<(Self::Repr, Self::Guard)> {
        let guard = self.memory.access_on(stream)?;
        let repr = self.desc.encode_checked(unsafe { guard.ptr() })?;
        Ok((repr, guard))
    }
}

impl<'a, M> KernelArg<'a> for &'a mut TensorMap<M>
where
    M: DeviceVisibleMemoryMut + 'a,
{
    type Repr = TensorMapRepr;
    type Guard = M::GuardMut<'a>;

    unsafe fn into_arg(self, stream: &'a Stream) -> Result<(Self::Repr, Self::Guard)> {
        let mut guard = self.memory.access_mut_on(stream)?;
        let repr = self.desc.encode_checked(unsafe { guard.ptr() })?;
        Ok((repr, guard))
    }
}
