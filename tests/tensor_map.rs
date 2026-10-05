// --- LLM-generated --- //

//! Host-side tests for tensor maps. No TMA kernels are launched, but the
//! driver refuses to encode tensor maps on pre-Hopper GPUs, so tests that
//! encode return early there.

use skycuda::{
    cudarc::driver::{DriverError, sys::CUresult},
    device::Device,
    device_memory::DeviceMemory,
    device_memory_pool::{DeviceMemoryPool, DeviceMemoryPoolOptions},
    error::Error,
    launch::KernelArg,
    stream::Stream,
    tensor_map::{
        TensorMap, TensorMapDataType, TensorMapDesc, TensorMapMut, TensorMapOobFill, TensorMapRepr,
        TensorMapSwizzle,
    },
};

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
            .create_device_memory_pool(DeviceMemoryPoolOptions {
                max_reserved_free_bytes: 0,
                max_total_bytes: None,
            })
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
