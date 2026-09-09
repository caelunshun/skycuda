use std::{
    fmt::Debug,
    mem, ptr,
    sync::{Arc, Mutex},
};

use cudarc::driver::sys::{
    CUevent,
    CUevent_flags::CU_EVENT_DISABLE_TIMING,
    CUresult::{CUDA_ERROR_NOT_READY, CUDA_SUCCESS},
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventSynchronize,
};

use crate::{device::Device, error::Result};

/// CUDA event with no timing support, used
/// only for synchronization.
#[derive(Debug, Clone)]
pub struct Event(Arc<Inner>);

impl Event {
    pub(crate) fn new(device: &Device) -> Result<Self> {
        let mut cu_event = ptr::null_mut();
        unsafe {
            cuEventCreate(&mut cu_event, CU_EVENT_DISABLE_TIMING as u32).result()?;
        };

        Ok(Self(Arc::new(Inner {
            device: device.clone(),
            cu_event,
            synchronize_callbacks: Mutex::new(Vec::new()),
        })))
    }

    pub fn is_completed(&self) -> Result<bool> {
        let res = unsafe { cuEventQuery(self.0.cu_event) };
        match res {
            CUDA_SUCCESS => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            e => e.result().map_err(From::from).map(|_| unreachable!()),
        }
    }

    pub fn block_on(&self) -> Result<()> {
        unsafe {
            cuEventSynchronize(self.0.cu_event).result()?;
        }
        self.completion_observed()?;
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    pub fn cu_event(&self) -> CUevent {
        self.0.cu_event
    } 

    pub(crate) fn on_complete(
        &self,
        callback: impl FnOnce() -> Result<()> + Send + Sync + 'static,
    ) {
        self.0
            .synchronize_callbacks
            .lock()
            .unwrap()
            .push(Box::new(callback));
    }

    pub(crate) fn completion_observed(&self) -> Result<()> {
        let callbacks = mem::take(&mut *self.0.synchronize_callbacks.lock().unwrap());
        for callback in callbacks {
            callback()?;
        }
        Ok(())
    }
}

struct Inner {
    device: Device,
    cu_event: CUevent,
    synchronize_callbacks: Mutex<Vec<Box<dyn FnOnce() -> Result<()> + Send + Sync>>>,
}

impl Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("device", &self.device)
            .field("cu_event", &self.cu_event)
            .finish()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = cuEventDestroy_v2(self.cu_event).result();
        }
    }
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
