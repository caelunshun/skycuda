use std::{ffi::c_void, ptr, sync::Arc};

use cudarc::driver::sys::{
    CUmemAllocationHandleType, CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
    CUmemLocation_st, CUmemLocation_st__bindgen_ty_1,
    CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
    CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD, CUmemPoolProps_st, CUmemoryPool,
    cuMemPoolCreate, cuMemPoolDestroy, cuMemPoolSetAttribute,
};

use crate::{device::Device, error::Result};

#[derive(Debug, Clone)]
pub struct DeviceMemoryPool(Arc<Inner>);

impl DeviceMemoryPool {
    pub(crate) fn new(device: &Device) -> Result<Self> {
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
            maxSize: 0,
            usage: 0,
            reserved: [0; _],
        };
        unsafe {
            cuMemPoolCreate(&mut cu_mempool, &props).result()?;
        }

        Ok(Self(Arc::new(Inner {
            device: device.clone(),
            cu_mempool,
        })))
    }

    pub fn set_max_reserved_free_bytes(&self, max_bytes: usize) -> Result<()> {
        let mut max_bytes = u64::try_from(max_bytes)?;
        unsafe {
            cuMemPoolSetAttribute(
                self.0.cu_mempool,
                CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut max_bytes as *mut u64 as *mut c_void,
            )
            .result()?;
        }
        Ok(())
    }

    pub fn cu_mempool(&self) -> CUmemoryPool {
        self.0.cu_mempool
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }
}

#[derive(Debug)]
struct Inner {
    device: Device,
    cu_mempool: CUmemoryPool,
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
