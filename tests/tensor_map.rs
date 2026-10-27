// --- LLM-generated --- //

//! Host-side tests for tensor maps. No TMA kernels are launched, but the
//! driver refuses to encode tensor maps on pre-Hopper GPUs, so tests that
//! encode return early there. The `TensorMapLayout` tests never touch the
//! driver.

use skycuda::{
    cudarc::driver::{DriverError, sys::CUresult},
    device::Device,
    device_memory::DeviceMemory,
    device_memory_pool::{DeviceMemoryPool, DeviceMemoryPoolOptions},
    error::Error,
    launch::KernelArg,
    stream::Stream,
    tensor_map::{
        TensorMap, TensorMapDataType, TensorMapDesc, TensorMapInterleave, TensorMapL2Promotion,
        TensorMapLayout, TensorMapLayoutError, TensorMapMut, TensorMapOobFill, TensorMapRepr,
        TensorMapSwizzle,
    },
};
use std::collections::HashSet;

/// Kernel with a single `CUtensorMap`-sized parameter and no TMA instructions.
const TAKES_TENSOR_MAP_PTX: &str = r#"
.version 7.8
.target sm_80
.address_size 64

.visible .entry takes_tensor_map(.param .align 64 .b8 tensor_map[128])
{
    ret;
}
"#;

struct Fixture {
    device: Device,
    stream: Stream,
    pool: DeviceMemoryPool,
}

impl Fixture {
    fn new() -> Self {
        let device = Device::from_index(0).unwrap();
        let stream = device.create_stream().unwrap();
        let pool = device
            .create_device_memory_pool(DeviceMemoryPoolOptions::default())
            .unwrap();
        Self {
            device,
            stream,
            pool,
        }
    }

    fn alloc(&self, len: usize) -> DeviceMemory {
        self.pool.alloc_zeroed_on(len, &self.stream).unwrap()
    }

    fn supports_tma(&self) -> bool {
        self.device.compute_capability_major() >= 9
    }
}

/// 50 rows of 100 f16 values, rows padded to 256 bytes.
fn f16_matrix_desc() -> TensorMapDesc {
    TensorMapDesc::new(TensorMapDataType::F16, &[100, 50], &[256], &[64, 32])
        .swizzle(TensorMapSwizzle::B128)
}

const F16_MATRIX_FOOTPRINT: usize = 100 * 2 + 49 * 256;

fn is_nonzero(repr: &TensorMapRepr) -> bool {
    repr.0.opaque.iter().any(|&q| q != 0)
}

#[test]
fn repr_layout() {
    assert_eq!(size_of::<TensorMapRepr>(), 128);
    assert_eq!(align_of::<TensorMapRepr>(), 128);
}

#[test]
fn footprint() {
    assert_eq!(f16_matrix_desc().footprint_bytes(), F16_MATRIX_FOOTPRINT);

    let rank3 = TensorMapDesc::new(TensorMapDataType::F32, &[8, 4, 3], &[64, 512], &[8, 4, 3]);
    assert_eq!(rank3.footprint_bytes(), 8 * 4 + 3 * 64 + 2 * 512);

    let packed16 = TensorMapDesc::new(
        TensorMapDataType::U4x16Align16B,
        &[256, 10],
        &[512],
        &[128, 10],
    );
    assert_eq!(packed16.footprint_bytes(), 256 + 9 * 512);

    let packed8 = TensorMapDesc::new(
        TensorMapDataType::U4x16Align8B,
        &[256, 2],
        &[128],
        &[128, 2],
    );
    assert_eq!(packed8.footprint_bytes(), 128 + 128);
}

#[test]
fn encode_read() {
    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let memory = fx.alloc(F16_MATRIX_FOOTPRINT);
    let map = TensorMap::new(&memory, f16_matrix_desc());
    let (repr, _guard) = unsafe { (&map).into_arg(&fx.stream) }.unwrap();
    assert!(is_nonzero(&repr));
}

#[test]
fn encode_write() {
    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let mut memory = fx.alloc(F16_MATRIX_FOOTPRINT);
    let mut map = TensorMapMut::new(&mut memory, f16_matrix_desc());
    let (repr, _guard) = unsafe { (&mut map).into_arg(&fx.stream) }.unwrap();
    assert!(is_nonzero(&repr));
}

#[test]
fn encode_owned_read() {
    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let map = TensorMap::new(fx.alloc(F16_MATRIX_FOOTPRINT), f16_matrix_desc());
    let (repr, _guard) = unsafe { map.into_arg(&fx.stream) }.unwrap();
    assert!(is_nonzero(&repr));
}

#[test]
fn encode_owned_write() {
    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let map = TensorMapMut::new(fx.alloc(F16_MATRIX_FOOTPRINT), f16_matrix_desc());
    let (repr, _guard) = unsafe { map.into_arg(&fx.stream) }.unwrap();
    assert!(is_nonzero(&repr));
}

#[test]
fn encode_owned_slice_mut() {
    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let mut memory = fx.alloc(F16_MATRIX_FOOTPRINT + 256);
    let map = TensorMapMut::new(memory.slice_mut(256..), f16_matrix_desc());
    let (repr, _guard) = unsafe { map.into_arg(&fx.stream) }.unwrap();
    assert!(is_nonzero(&repr));
}

#[test]
fn encode_slice() {
    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let memory = fx.alloc(F16_MATRIX_FOOTPRINT + 256);
    let map = TensorMap::new(memory.slice(256..), f16_matrix_desc());
    let (repr, _guard) = unsafe { (&map).into_arg(&fx.stream) }.unwrap();
    assert!(is_nonzero(&repr));
}

#[test]
fn launch_builder_accepts_tensor_map() {
    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let module = fx.device.load_module_ptx(TAKES_TENSOR_MAP_PTX).unwrap();
    let function = module.get_function("takes_tensor_map").unwrap();
    assert_eq!(function.param_sizes(), &[128]);

    let memory = fx.alloc(F16_MATRIX_FOOTPRINT);
    let map = TensorMap::new(&memory, f16_matrix_desc());
    fx.stream.launch_builder(&function).arg(&map).unwrap();
    fx.stream
        .launch_builder(&function)
        .arg(TensorMap::new(memory.slice(..), f16_matrix_desc()))
        .unwrap();
}

#[test]
fn driver_rejection_is_error() {
    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let memory = fx.alloc(4096);
    // NaN OOB fill is only valid for floating-point types, which we leave to the driver.
    let desc = TensorMapDesc::new(TensorMapDataType::U8, &[64, 64], &[64], &[64, 8])
        .oob_fill(TensorMapOobFill::NanRequestZeroFma);
    let map = TensorMap::new(&memory, desc);
    assert!(unsafe { (&map).into_arg(&fx.stream) }.is_err());
}

#[test]
fn encode_unsupported_before_hopper() {
    let fx = Fixture::new();
    if fx.supports_tma() {
        return;
    }
    let memory = fx.alloc(F16_MATRIX_FOOTPRINT);
    let map = TensorMap::new(&memory, f16_matrix_desc());
    let err = unsafe { (&map).into_arg(&fx.stream) }.err().unwrap();
    assert!(matches!(
        err,
        Error::CudaDriver(DriverError(CUresult::CUDA_ERROR_NOT_SUPPORTED))
    ));
}

#[test]
#[should_panic(expected = "tensor map addresses")]
fn footprint_exceeds_memory() {
    let fx = Fixture::new();
    let memory = fx.alloc(F16_MATRIX_FOOTPRINT - 1);
    TensorMap::new(&memory, f16_matrix_desc());
}

#[test]
#[should_panic(expected = "global stride 0 must be a multiple of 16")]
fn misaligned_stride() {
    let fx = Fixture::new();
    let memory = fx.alloc(1 << 16);
    let desc = TensorMapDesc::new(TensorMapDataType::F16, &[100, 50], &[200], &[64, 32]);
    TensorMap::new(&memory, desc);
}

#[test]
#[should_panic(expected = "box dim 1 must be in 1..=256")]
fn box_too_large() {
    let fx = Fixture::new();
    let memory = fx.alloc(1 << 20);
    let desc = TensorMapDesc::new(TensorMapDataType::F16, &[64, 1024], &[128], &[64, 512]);
    TensorMap::new(&memory, desc);
}

#[test]
#[should_panic(expected = "exceeds swizzle span")]
fn box_exceeds_swizzle_span() {
    let fx = Fixture::new();
    let memory = fx.alloc(1 << 16);
    let desc = TensorMapDesc::new(TensorMapDataType::F32, &[64, 64], &[256], &[64, 8])
        .swizzle(TensorMapSwizzle::B128);
    TensorMap::new(&memory, desc);
}

#[test]
#[should_panic(expected = "is not 16-byte aligned")]
fn misaligned_address() {
    let fx = Fixture::new();
    let memory = fx.alloc(F16_MATRIX_FOOTPRINT + 8);
    let map = TensorMap::new(memory.slice(8..), f16_matrix_desc());
    let _ = unsafe { (&map).into_arg(&fx.stream) };
}

fn f16_layout() -> TensorMapLayout {
    TensorMapLayout::new(TensorMapDataType::F16, &[64, 32]).swizzle(TensorMapSwizzle::B128)
}

#[test]
fn from_layout_matches_new() {
    let desc = f16_matrix_desc();
    let from_layout = TensorMapDesc::from_layout(&f16_layout(), &[100, 50], &[256]);
    assert_eq!(desc.layout(), from_layout.layout());
    assert_eq!(desc.layout(), &f16_layout());
    assert_eq!(desc.global_dims(), from_layout.global_dims());
    assert_eq!(desc.global_strides(), from_layout.global_strides());
    assert_eq!(desc.footprint_bytes(), from_layout.footprint_bytes());
    assert_eq!(desc.alignment(), from_layout.alignment());
    assert_eq!(from_layout.rank(), 2);
    assert_eq!(from_layout.box_dims(), &[64, 32]);
    assert_eq!(from_layout.get_swizzle(), TensorMapSwizzle::B128);

    let fx = Fixture::new();
    if !fx.supports_tma() {
        return;
    }
    let memory = fx.alloc(F16_MATRIX_FOOTPRINT);
    let map_a = TensorMap::new(&memory, desc);
    let map_b = TensorMap::new(&memory, from_layout);
    let (a, _guard_a) = unsafe { (&map_a).into_arg(&fx.stream) }.unwrap();
    let (b, _guard_b) = unsafe { (&map_b).into_arg(&fx.stream) }.unwrap();
    assert!(is_nonzero(&a));
    assert_eq!(a.0.opaque, b.0.opaque);
}

#[test]
fn desc_builders_forward_to_layout() {
    let desc = TensorMapDesc::new(TensorMapDataType::F32, &[8, 4, 3], &[64, 512], &[8, 4, 3])
        .element_strides(&[1, 2, 1])
        .interleave(TensorMapInterleave::B16)
        .swizzle(TensorMapSwizzle::B64)
        .l2_promotion(TensorMapL2Promotion::B128)
        .oob_fill(TensorMapOobFill::NanRequestZeroFma);
    let layout = TensorMapLayout::new(TensorMapDataType::F32, &[8, 4, 3])
        .element_strides(&[1, 2, 1])
        .interleave(TensorMapInterleave::B16)
        .swizzle(TensorMapSwizzle::B64)
        .l2_promotion(TensorMapL2Promotion::B128)
        .oob_fill(TensorMapOobFill::NanRequestZeroFma);
    assert_eq!(desc.layout(), &layout);
    assert_eq!(desc.data_type(), TensorMapDataType::F32);
    assert_eq!(desc.get_element_strides(), &[1, 2, 1]);
    assert_eq!(desc.get_interleave(), TensorMapInterleave::B16);
    assert_eq!(desc.get_swizzle(), TensorMapSwizzle::B64);
    assert_eq!(desc.get_l2_promotion(), TensorMapL2Promotion::B128);
    assert_eq!(desc.get_oob_fill(), TensorMapOobFill::NanRequestZeroFma);
}

#[test]
#[should_panic(expected = "needs exactly 2 global dims")]
fn from_layout_rank_mismatch() {
    TensorMapDesc::from_layout(&f16_layout(), &[100, 50, 2], &[256, 1024]);
}

#[test]
fn layout_eq_and_hash() {
    let a = f16_layout();
    let b = TensorMapLayout::new(TensorMapDataType::F16, &[64, 32]).swizzle(TensorMapSwizzle::B128);
    assert_eq!(a, b);
    // Unused trailing slots must not leak into equality.
    let rank1 = TensorMapLayout::new(TensorMapDataType::U8, &[64]);
    assert_eq!(rank1, TensorMapLayout::new(TensorMapDataType::U8, &[64]));
    assert_eq!(rank1.box_dims(), &[64]);
    assert_eq!(rank1.get_element_strides(), &[1]);

    let set: HashSet<_> = [
        a.clone(),
        b,
        a.clone().swizzle(TensorMapSwizzle::B64),
        a.clone().element_strides(&[1, 2]),
        a.clone().l2_promotion(TensorMapL2Promotion::B256),
        a.oob_fill(TensorMapOobFill::NanRequestZeroFma),
    ]
    .into_iter()
    .collect();
    assert_eq!(set.len(), 5);
}

#[test]
fn box_bytes_plain() {
    assert_eq!(f16_layout().box_bytes(), 64 * 32 * 2);
    assert_eq!(
        TensorMapLayout::new(TensorMapDataType::U8, &[16]).box_bytes(),
        16
    );
    assert_eq!(
        TensorMapLayout::new(TensorMapDataType::F64, &[4, 3, 2]).box_bytes(),
        4 * 3 * 2 * 8
    );
    // Swizzle, L2 promotion and OOB fill don't change the size.
    let layout = TensorMapLayout::new(TensorMapDataType::F32, &[32, 8])
        .swizzle(TensorMapSwizzle::B128Atom32B)
        .l2_promotion(TensorMapL2Promotion::B256)
        .oob_fill(TensorMapOobFill::NanRequestZeroFma);
    assert_eq!(layout.box_bytes(), 32 * 8 * 4);
}

#[test]
fn box_bytes_element_strides() {
    // ceil(box / stride) elements per dimension.
    let layout = TensorMapLayout::new(TensorMapDataType::F16, &[64, 32]);
    assert_eq!(
        layout.clone().element_strides(&[1, 2]).box_bytes(),
        64 * 16 * 2
    );
    assert_eq!(
        layout.clone().element_strides(&[1, 3]).box_bytes(),
        64 * 11 * 2
    );
    assert_eq!(
        layout.clone().element_strides(&[1, 8]).box_bytes(),
        64 * 4 * 2
    );
    // Dimension 0's stride is ignored without interleave.
    assert_eq!(layout.element_strides(&[4, 1]).box_bytes(), 64 * 32 * 2);

    // With interleave it applies, and the row rounds up to the 16-byte slice.
    let interleaved = TensorMapLayout::new(TensorMapDataType::F16, &[16, 4, 2])
        .interleave(TensorMapInterleave::B16);
    assert_eq!(interleaved.clone().box_bytes(), 32 * 4 * 2);
    assert_eq!(
        interleaved.clone().element_strides(&[2, 1, 1]).box_bytes(),
        16 * 4 * 2
    );
    assert_eq!(
        interleaved.element_strides(&[4, 2, 1]).box_bytes(),
        16 * 2 * 2
    );
}

#[test]
fn box_bytes_interleave_rounds_slice() {
    let b32 = TensorMapLayout::new(TensorMapDataType::F16, &[8, 4, 2])
        .interleave(TensorMapInterleave::B32)
        .swizzle(TensorMapSwizzle::B32);
    assert_eq!(b32.box_bytes(), 32 * 4 * 2);
    let b16 = TensorMapLayout::new(TensorMapDataType::F32, &[4, 4, 2])
        .interleave(TensorMapInterleave::B16);
    assert_eq!(b16.box_bytes(), 16 * 4 * 2);
}

#[test]
fn box_bytes_packed() {
    // 16 values per 8 bytes, no gaps.
    let u4 = TensorMapLayout::new(TensorMapDataType::U4x16Align8B, &[128, 2]);
    assert_eq!(u4.box_bytes(), 64 * 2);
    assert_eq!(
        TensorMapLayout::new(TensorMapDataType::U4x16Align8B, &[64, 4]).box_bytes(),
        32 * 4
    );
    // 16 values per 16-byte slot.
    let u4_16 = TensorMapLayout::new(TensorMapDataType::U4x16Align16B, &[128, 10]);
    assert_eq!(u4_16.box_bytes(), 128 * 10);
    let u6 = TensorMapLayout::new(TensorMapDataType::U6x16Align16B, &[128, 4, 2]);
    assert_eq!(u6.box_bytes(), 128 * 4 * 2);
    // Element strides apply to the outer dimensions.
    assert_eq!(u6.element_strides(&[1, 2, 1]).box_bytes(), 128 * 2 * 2);
    assert_eq!(
        TensorMapLayout::new(TensorMapDataType::U4x16Align8B, &[128, 8])
            .element_strides(&[1, 4])
            .box_bytes(),
        64 * 2
    );
}

#[test]
fn smem_alignment() {
    use TensorMapSwizzle::*;
    let layout = TensorMapLayout::new(TensorMapDataType::U8, &[16, 8]);
    let expected = [
        (None, 128),
        (B32, 256),
        (B64, 512),
        (B128, 1024),
        (B128Atom32B, 512),
        (B128Atom32BFlip8B, 512),
        (B128Atom64B, 256),
    ];
    for (swizzle, alignment) in expected {
        assert_eq!(
            layout.clone().swizzle(swizzle).smem_alignment(),
            alignment,
            "{swizzle:?}"
        );
    }
    // Other fields don't matter.
    assert_eq!(f16_layout().smem_alignment(), 1024);
    assert_eq!(
        TensorMapLayout::new(TensorMapDataType::F64, &[2, 2, 2])
            .interleave(TensorMapInterleave::B16)
            .smem_alignment(),
        128
    );
}

#[test]
fn check_accepts_valid_layouts() {
    assert_eq!(f16_layout().check(), Ok(()));
    let packed = TensorMapLayout::new(TensorMapDataType::U4x16Align16B, &[128, 4])
        .swizzle(TensorMapSwizzle::B128);
    assert_eq!(packed.check(), Ok(()));
    let packed8 = TensorMapLayout::new(TensorMapDataType::U4x16Align8B, &[64, 4])
        .swizzle(TensorMapSwizzle::B64);
    assert_eq!(packed8.check(), Ok(()));
    let interleaved = TensorMapLayout::new(TensorMapDataType::F16, &[16, 4, 2])
        .interleave(TensorMapInterleave::B32)
        .swizzle(TensorMapSwizzle::B32);
    assert_eq!(interleaved.check(), Ok(()));
}

#[test]
fn check_errors() {
    use TensorMapLayoutError::*;
    let f16 = |box_dims: &[u32]| TensorMapLayout::new(TensorMapDataType::F16, box_dims);

    let layout = f16(&[8, 4]).interleave(TensorMapInterleave::B16);
    assert_eq!(layout.check(), Err(InterleaveRank { rank: 2 }));

    assert_eq!(
        f16(&[64, 257]).check(),
        Err(BoxDimOutOfRange { dim: 1, value: 257 })
    );
    assert_eq!(
        f16(&[64, 0]).check(),
        Err(BoxDimOutOfRange { dim: 1, value: 0 })
    );

    assert_eq!(
        f16(&[64, 4]).element_strides(&[1, 9]).check(),
        Err(ElementStrideOutOfRange { dim: 1, value: 9 })
    );
    assert_eq!(
        f16(&[64, 4]).element_strides(&[0, 1]).check(),
        Err(ElementStrideOutOfRange { dim: 0, value: 0 })
    );

    let layout = f16(&[16, 4, 2])
        .interleave(TensorMapInterleave::B32)
        .swizzle(TensorMapSwizzle::B64);
    assert_eq!(
        layout.check(),
        Err(Interleave32RequiresSwizzle32 {
            swizzle: TensorMapSwizzle::B64
        })
    );

    let layout = TensorMapLayout::new(TensorMapDataType::U6x16Align16B, &[128, 4, 2])
        .interleave(TensorMapInterleave::B16);
    assert_eq!(
        layout.check(),
        Err(PackedInterleave {
            data_type: TensorMapDataType::U6x16Align16B,
            interleave: TensorMapInterleave::B16
        })
    );

    for data_type in [
        TensorMapDataType::U4x16Align16B,
        TensorMapDataType::U6x16Align16B,
    ] {
        assert_eq!(
            TensorMapLayout::new(data_type, &[64, 4]).check(),
            Err(PackedBoxDim0 {
                data_type,
                value: 64
            })
        );
    }

    // 15 elements * 2 bytes.
    assert_eq!(
        f16(&[15, 4]).check(),
        Err(InnerBoxNotMultipleOf16 { bytes: 30 })
    );
    // 16 values per 8 bytes: 16 values is half a 16-byte row.
    assert_eq!(
        TensorMapLayout::new(TensorMapDataType::U4x16Align8B, &[16]).check(),
        Err(InnerBoxNotMultipleOf16 { bytes: 8 })
    );
    // The element stride does not change the inner box rule.
    assert_eq!(f16(&[64, 4]).element_strides(&[8, 1]).check(), Ok(()));

    let layout =
        TensorMapLayout::new(TensorMapDataType::F32, &[64, 8]).swizzle(TensorMapSwizzle::B128);
    assert_eq!(
        layout.check(),
        Err(InnerBoxExceedsSwizzleSpan {
            bytes: 256,
            span: 128
        })
    );
    let layout = f16(&[64, 4]).swizzle(TensorMapSwizzle::B64);
    assert_eq!(
        layout.check(),
        Err(InnerBoxExceedsSwizzleSpan {
            bytes: 128,
            span: 64
        })
    );
    // 128 bytes per packed row fit only the 128-byte swizzle family.
    let packed = TensorMapLayout::new(TensorMapDataType::U4x16Align16B, &[128, 4])
        .swizzle(TensorMapSwizzle::B32);
    assert_eq!(
        packed.check(),
        Err(InnerBoxExceedsSwizzleSpan {
            bytes: 128,
            span: 32
        })
    );
}

#[test]
fn check_error_messages() {
    let err = TensorMapLayout::new(TensorMapDataType::F16, &[64, 512])
        .check()
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "tensor map box dim 1 must be in 1..=256, got 512"
    );
}

#[test]
#[should_panic(expected = "32B-interleaved tensor maps require 32B swizzle")]
fn validate_panics_with_layout_message() {
    let fx = Fixture::new();
    let memory = fx.alloc(1 << 16);
    let desc = TensorMapDesc::new(TensorMapDataType::F16, &[16, 4, 2], &[64, 256], &[16, 4, 2])
        .interleave(TensorMapInterleave::B32);
    TensorMap::new(&memory, desc);
}

#[test]
#[should_panic(expected = "require global dim 0 to be a multiple of 128")]
fn validate_packed_global_dim() {
    let fx = Fixture::new();
    let memory = fx.alloc(1 << 16);
    let desc = TensorMapDesc::new(
        TensorMapDataType::U4x16Align16B,
        &[192, 4],
        &[512],
        &[128, 4],
    );
    TensorMap::new(&memory, desc);
}
