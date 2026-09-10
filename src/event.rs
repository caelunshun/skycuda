use std::{
    fmt::Debug,
    mem,
    sync::{Arc, Mutex},
};

use cudarc::driver::sys::{
    CUevent,
    CUresult::{CUDA_ERROR_NOT_READY, CUDA_SUCCESS}, cuEventQuery, cuEventSynchronize,
};

use crate::{device::Device, error::Result};

/// CUDA event with no timing support, used
/// only for synchronization.
///
/// Unlike raw CUDA events, an event is recorded
/// only once. We pool raw events internally to facilitate
/// reuse.
#[derive(Debug, Clone)]
pub struct Event(Arc<Inner>);

impl Event {
    pub(crate) fn new(device: &Device) -> Result<Self> {
        let cu_event = device.take_pooled_cu_event()?;
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
        self.notify_completion_observed()?;
        Ok(())
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    pub fn cu_event(&self) -> CUevent {
        self.0.cu_event
    }

    pub(crate) fn on_completion_observed(
        &self,
        callback: impl FnOnce() -> Result<()> + Send + Sync + 'static,
    ) {
        self.0
            .synchronize_callbacks
            .lock()
            .unwrap()
            .push(Box::new(callback));
    }

    pub(crate) fn notify_completion_observed(&self) -> Result<()> {
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
    #[allow(clippy::type_complexity)]
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
        let _ = self.device.release_pooled_cu_event(self.cu_event);
    }
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
