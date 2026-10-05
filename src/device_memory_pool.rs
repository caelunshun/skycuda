use crate::{device::Device, device_memory::DeviceMemory, error::Result, stream::Stream};
use bytesize::ByteSize;
use cudarc::driver::sys::{
    CUmemAllocationHandleType,
    CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
    CUmemLocation_st, CUmemLocation_st__bindgen_ty_1,
    CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
    CUmemPool_attribute::{
        CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
    },
    CUmemPoolProps_st, CUmemoryPool,
    CUresult::CUDA_ERROR_OUT_OF_MEMORY,
    cuMemPoolCreate, cuMemPoolDestroy, cuMemPoolGetAttribute, cuMemPoolSetAttribute,
};
use std::{
    collections::{BTreeMap, HashMap},
    ffi::c_void,
    panic::Location,
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
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
            maxSize: options.max_total_bytes,
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
            enable_profiling: options.enable_profiling,
            profiled_allocations: Mutex::new(HashMap::new()),
        })))
    }

    #[track_caller]
    pub fn alloc_zeroed_on(&self, len: usize, stream: &Stream) -> Result<DeviceMemory> {
        let mut memory = unsafe { self.alloc_uninit_on(len, stream)? };
        stream.fill(&mut memory, 0u8)?;
        Ok(memory)
    }

    /// Allocate uninitialized device memory.
    ///
    /// # Safety
    /// Reading or creating Rust references to the underlying buffer contents before they are written to is undefined
    /// behavior.
    #[track_caller]
    pub unsafe fn alloc_uninit_on(&self, len: usize, stream: &Stream) -> Result<DeviceMemory> {
        let id = self.new_allocation_id();
        let res = unsafe { DeviceMemory::new_uninit(len, self, stream, id) };

        match res {
            Ok(mem) => {
                self.record_allocation(id, len);
                Ok(mem)
            }
            Err(e @ crate::error::Error::CudaDriver(de)) if de.0 == CUDA_ERROR_OUT_OF_MEMORY => {
                if self.0.enable_profiling {
                    let mut allocations_by_size_and_site: BTreeMap<
                        (usize, &'static Location<'static>),
                        usize,
                    > = Default::default();
                    for allocation in self.0.profiled_allocations.lock().unwrap().values() {
                        *allocations_by_size_and_site
                            .entry((allocation.size, allocation.allocation_site))
                            .or_default() += 1;
                    }

                    let mut actual_bytes = 0u64;
                    unsafe {
                        cuMemPoolGetAttribute(
                            self.cu_mempool(),
                            CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT,
                            &mut actual_bytes as *mut u64 as *mut c_void,
                        );
                    }

                    tracing::error!(
                        "out of memory on allocation of {}. currently, {} is used from this pool (excluding fragmentation overhead), and the driver has reserved a total of {} in this pool. summary of live allocations in this pool:",
                        ByteSize(len as u64),
                        ByteSize(self.0.allocated_bytes.load(Ordering::Relaxed) as u64),
                        ByteSize(actual_bytes)
                    );
                    for ((size, site), count) in allocations_by_size_and_site.into_iter().rev() {
                        let total_bytes = size * count;
                        tracing::error!(
                            "* {} from {count} allocation{} of {} at {site}",
                            ByteSize(total_bytes as u64),
                            if count != 1 { "s" } else { "" },
                            ByteSize(size as u64)
                        );
                    }
                }

                Err(e)
            }
            Err(e) => Err(e),
        }
    }

    fn new_allocation_id(&self) -> u64 {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        NEXT.fetch_add(1, Ordering::Relaxed)
    }

    pub fn cu_mempool(&self) -> CUmemoryPool {
        self.0.cu_mempool
    }

    #[track_caller]
    fn record_allocation(&self, id: u64, bytes: usize) {
        if self.0.enable_profiling {
            self.0.profiled_allocations.lock().unwrap().insert(
                id,
                ProfiledAllocation {
                    allocation_site: Location::caller(),
                    size: bytes,
                },
            );
        }

        let prev_allocated = self.0.allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
        let new_allocated = prev_allocated + bytes;
        loop {
            let peak = self.0.peak_allocated_bytes.load(Ordering::Relaxed);
            if peak >= new_allocated {
                break;
            }
            if self
                .0
                .peak_allocated_bytes
                .compare_exchange_weak(peak, new_allocated, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                tracing::debug!(
                    "new allocated device memory peak: {}",
                    ByteSize::b(new_allocated as u64)
                );
                break;
            }
        }
    }

    pub(crate) fn record_deallocation(&self, id: u64, bytes: usize) {
        self.0.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);
        if self.0.enable_profiling {
            self.0.profiled_allocations.lock().unwrap().remove(&id);
        }
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
    enable_profiling: bool,
    profiled_allocations: Mutex<HashMap<u64, ProfiledAllocation>>,
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
    pub enable_profiling: bool,
    pub max_reserved_free_bytes: usize,
    pub max_total_bytes: usize,
}

impl Default for DeviceMemoryPoolOptions {
    fn default() -> Self {
        Self {
            enable_profiling: false,
            max_reserved_free_bytes: 1 << 30,
            max_total_bytes: 1 << 40,
        }
    }
}

#[derive(Debug, Clone)]
struct ProfiledAllocation {
    allocation_site: &'static Location<'static>,
    size: usize,
}
