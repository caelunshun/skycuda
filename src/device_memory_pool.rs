use crate::{device::Device, device_memory::DeviceMemory, error::Result, stream::Stream};
use cudarc::driver::sys::{
    CUmemAllocationHandleType, CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
    CUmemLocation_st, CUmemLocation_st__bindgen_ty_1,
    CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, CUmemPoolProps_st, CUmemoryPool,
    cuMemPoolCreate, cuMemPoolDestroy, cuMemPoolSetAttribute,
};
use std::{
    ffi::c_void,
    ptr,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

#[derive(Debug, Clone)]
pub struct DeviceMemoryPool(Arc<Inner>);

impl DeviceMemoryPool {
    pub(crate) fn new(options: DeviceMemoryPoolOptions, device: &Device) -> Result<Self> {
        device.set_as_current()?;

        let mut cu_mempool = ptr::null_mut();
        let props = CUmemPoolProps_st {
            allocType: CU_MEM_ALLOCATION_TYPE_PINNED,
            handleTypes: CUmemAllocationHandleType(0),
            location: CUmemLocation_st {
                type_: CU_MEM_LOCATION_TYPE_DEVICE,
                __bindgen_anon_1: CUmemLocation_st__bindgen_ty_1 {
                    id: device.ordinal().try_into()?,
                },
            },
            win32SecurityAttributes: ptr::null_mut(),
            maxSize: 1 << 40,
            usage: 0,
            gpuDirectRDMACapable: 0,
            reserved: [0; _],
        };

        let mut max_bytes = options.max_reserved_free_bytes as u64;
        unsafe {
            cuMemPoolCreate(&mut cu_mempool, &props).result()?;
            cuMemPoolSetAttribute(
                cu_mempool,
                CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut max_bytes as *mut u64 as *mut c_void,
            )
            .result()?;
        }

        Ok(Self(Arc::new(Inner {
            device: device.clone(),
            cu_mempool,
            allocated_bytes: AtomicUsize::new(0),
            peak_allocated_bytes: AtomicUsize::new(0),
        })))
    }

    pub fn alloc_zeroed_on(&self, len: usize, stream: &Stream) -> Result<DeviceMemory> {
        DeviceMemory::new_zeroed(len, self, stream)
    }

    pub fn cu_mempool(&self) -> CUmemoryPool {
        self.0.cu_mempool
    }

    pub(crate) fn record_allocation(&self, bytes: usize) {
        let prev_allocated = self.0.allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
        let new_allocated = prev_allocated + bytes;
        loop {
            let peak = self.0.peak_allocated_bytes.load(Ordering::Relaxed);
            if peak > new_allocated {
                break;
            }
            if self
                .0
                .peak_allocated_bytes
                .compare_exchange_weak(peak, new_allocated, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }

    pub(crate) fn record_deallocation(&self, bytes: usize) {
        self.0.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    /// Current number of live allocated bytes (not including fragmentation / reserved
    /// memory overhead).
    pub fn current_allocated_bytes(&self) -> usize {
        self.0.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Peak number of live allocated bytes at any point during this pool's lifetime.
    pub fn peak_allocated_bytes(&self) -> usize {
        self.0.peak_allocated_bytes.load(Ordering::Relaxed)
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }
}

#[derive(Debug)]
struct Inner {
    device: Device,
    cu_mempool: CUmemoryPool,
    allocated_bytes: AtomicUsize,
    peak_allocated_bytes: AtomicUsize,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemPoolDestroy(self.cu_mempool).result();
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeviceMemoryPoolOptions {
    pub max_reserved_free_bytes: usize,
}
