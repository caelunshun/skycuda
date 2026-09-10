use std::{
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

use cudarc::driver::sys::{
    CUcontext,
    CUctx_flags::CU_CTX_MAP_HOST,
    CUctxCreateParams_st,
    CUdevice_attribute::{
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
        CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    },
    CUevent,
    CUevent_flags::CU_EVENT_DISABLE_TIMING,
    cuCtxCreate_v4, cuCtxDestroy_v2, cuCtxSetCurrent, cuDeviceGetAttribute, cuEventCreate,
    cuEventDestroy_v2,
};

use crate::{
    device_memory_pool::{DeviceMemoryPool, DeviceMemoryPoolOptions},
    error::Result,
    host_pinned_memory_pool::{HostPinnedMemoryPool, HostPinnedMemoryPoolOptions},
    stream::Stream,
};

#[derive(Debug, Clone)]
pub struct Device(Arc<Inner>);

impl Device {
    pub fn from_index(ordinal: u32) -> Result<Self> {
        let ordinal = i32::try_from(ordinal)?;

        let mut cu_context = ptr::null_mut();
        let mut params = CUctxCreateParams_st {
            execAffinityParams: ptr::null_mut(),
            numExecAffinityParams: 0,
            cigParams: ptr::null_mut(),
        };
        unsafe {
            cuCtxCreate_v4(
                &mut cu_context,
                &mut params,
                CU_CTX_MAP_HOST as u32,
                ordinal,
            )
            .result()?;
        }

        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        let cached_attributes = CachedAttributes::get(ordinal)?;

        Ok(Self(Arc::new(Inner {
            id,
            device_ordinal: ordinal,
            cu_context,
            cached_attributes,
            cu_event_pool: Mutex::new(Vec::new()),
        })))
    }

    pub fn create_stream(&self) -> Result<Stream> {
        Stream::new(self)
    }

    pub fn create_device_memory_pool(
        &self,
        options: DeviceMemoryPoolOptions,
    ) -> Result<DeviceMemoryPool> {
        DeviceMemoryPool::new(options, self)
    }

    pub fn create_host_pinned_memory_pool(
        &self,
        options: HostPinnedMemoryPoolOptions,
    ) -> Result<HostPinnedMemoryPool> {
        HostPinnedMemoryPool::new(options, self)
    }

    pub fn ordinal(&self) -> u32 {
        self.0.device_ordinal.try_into().unwrap()
    }

    pub fn context_id(&self) -> u64 {
        self.0.id
    }

    pub fn cu_context(&self) -> CUcontext {
        self.0.cu_context
    }

    pub fn set_as_current(&self) -> Result<()> {
        unsafe {
            cuCtxSetCurrent(self.0.cu_context).result()?;
        }
        Ok(())
    }

    pub fn compute_capability_major(&self) -> u32 {
        self.0.cached_attributes.compute_capability_major
    }

    pub fn compute_capability_minor(&self) -> u32 {
        self.0.cached_attributes.compute_capability_minor
    }

    pub fn l2_bytes(&self) -> usize {
        self.0.cached_attributes.l2_bytes
    }

    pub fn sm_count(&self) -> usize {
        self.0.cached_attributes.sm_count
    }

    pub fn max_dynamic_smem_per_cta(&self) -> usize {
        self.0.cached_attributes.max_dynamic_smem
    }

    pub(crate) fn take_pooled_cu_event(&self) -> Result<CUevent> {
        match self.0.cu_event_pool.lock().unwrap().pop() {
            Some(event) => Ok(event),
            None => {
                let mut event = ptr::null_mut();
                unsafe {
                    cuEventCreate(&mut event, CU_EVENT_DISABLE_TIMING as u32).result()?;
                }
                Ok(event)
            }
        }
    }

    pub(crate) fn release_pooled_cu_event(&self, cu_event: CUevent) -> Result<()> {
        self.0.cu_event_pool.lock().unwrap().push(cu_event);
        Ok(())
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Device {}

#[derive(Debug)]
struct Inner {
    id: u64,
    device_ordinal: i32,
    cu_context: CUcontext,
    cached_attributes: CachedAttributes,
    cu_event_pool: Mutex<Vec<CUevent>>,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            for cu_event in self.cu_event_pool.get_mut().unwrap().drain(..) {
                let _ = cuEventDestroy_v2(cu_event).result();
            }

            let _ = cuCtxDestroy_v2(self.cu_context).result();
        }
    }
}

#[derive(Debug)]
struct CachedAttributes {
    compute_capability_major: u32,
    compute_capability_minor: u32,
    l2_bytes: usize,
    sm_count: usize,
    max_dynamic_smem: usize,
}

impl CachedAttributes {
    pub fn get(device_idx: i32) -> Result<Self> {
        let mut compute_capability_major = 0;
        let mut compute_capability_minor = 0;
        let mut l2_bytes = 0;
        let mut sm_count = 0;
        let mut max_dynamic_smem = 0;

        unsafe {
            cuDeviceGetAttribute(
                &mut compute_capability_major,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                device_idx,
            )
            .result()?;
            cuDeviceGetAttribute(
                &mut compute_capability_minor,
                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                device_idx,
            )
            .result()?;
            cuDeviceGetAttribute(&mut l2_bytes, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, device_idx)
                .result()?;
            cuDeviceGetAttribute(
                &mut sm_count,
                CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                device_idx,
            )
            .result()?;
            cuDeviceGetAttribute(
                &mut max_dynamic_smem,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                device_idx,
            )
            .result()?;
        }

        Ok(CachedAttributes {
            compute_capability_major: compute_capability_major.try_into()?,
            compute_capability_minor: compute_capability_minor.try_into()?,
            l2_bytes: l2_bytes.try_into()?,
            sm_count: sm_count.try_into()?,
            max_dynamic_smem: max_dynamic_smem.try_into()?,
        })
    }
}
