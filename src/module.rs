use std::{ptr, sync::Arc};

use cudarc::driver::sys::{CUmodule, cuModuleLoadData};

use crate::{device::Device, error::Result};

#[derive(Debug, Clone)]
pub struct Module(Arc<Inner>);

impl Module {
    /// # Safety
    /// - If `data` contains PTX, it must be null-terminated.
    /// - If `data` contains cubin, it must be well-formed.
    pub(crate) unsafe fn new(data: &[u8], device: &Device) -> Result<Self> {
        device.set_as_current()?;
        let mut cu_module = ptr::null_mut();
        unsafe {
            cuModuleLoadData(&mut cu_module, data.as_ptr().cast()).result()?;
        }

        Ok(Self(Arc::new(Inner {
            device: device.clone(),
            cu_module,
        })))
    }

    pub fn cu_module(&self) -> CUmodule {
        self.0.cu_module
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }
}

#[derive(Debug)]
struct Inner {
    device: Device,
    cu_module: CUmodule,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
