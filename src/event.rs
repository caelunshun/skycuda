use std::{
    fmt::Debug,
    mem,
    sync::{Arc, Mutex},
};

use cudarc::driver::sys::{
    CUevent,
    CUresult::{CUDA_ERROR_NOT_READY, CUDA_SUCCESS},
    cuEventQuery, cuEventRecord, cuEventSynchronize,
};

use crate::{device::Device, error::Result, stream::Stream};

/// CUDA event with no timing support, used
/// only for synchronization.
///
/// Unlike raw CUDA events, an `Event` instance is recorded
/// exactly once. We pool raw events internally to enable
/// reuse.
#[derive(Debug, Clone)]
pub struct Event(Arc<Inner>);

impl Event {
    pub(crate) fn record(stream: &Stream) -> Result<Self> {
        let cu_event = stream.device().take_pooled_cu_event()?;

        unsafe {
            cuEventRecord(cu_event, stream.cu_stream()).result()?;
        }

        Ok(Self(Arc::new(Inner {
            recorded_on: stream.clone(),
            cu_event,
            synchronize_callbacks: Mutex::new(Vec::new()),
        })))
    }

    pub fn is_completed(&self) -> Result<bool> {
        let res = unsafe { cuEventQuery(self.0.cu_event) };
        match res {
            CUDA_SUCCESS => {
                self.recorded_on().notify_completion_observed_up_to(self)?;
                Ok(true)
            }
            CUDA_ERROR_NOT_READY => Ok(false),
            e => e.result().map_err(From::from).map(|_| unreachable!()),
        }
    }

    pub fn block_on(&self) -> Result<()> {
        unsafe {
            cuEventSynchronize(self.0.cu_event).result()?;
        }
        self.recorded_on().notify_completion_observed_up_to(self)?;
        Ok(())
    }

    pub fn device(&self) -> &Device {
        self.0.recorded_on.device()
    }

    pub fn cu_event(&self) -> CUevent {
        self.0.cu_event
    }

    pub fn recorded_on(&self) -> &Stream {
        &self.0.recorded_on
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
        for callback in mem::take(&mut *self.0.synchronize_callbacks.lock().unwrap()) {
            callback()?;
        }
        Ok(())
    }
}

impl PartialEq for Event {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Event {}

struct Inner {
    recorded_on: Stream,
    cu_event: CUevent,
    #[allow(clippy::type_complexity)]
    synchronize_callbacks: Mutex<Vec<Box<dyn FnOnce() -> Result<()> + Send + Sync>>>,
}

impl Debug for Inner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inner")
            .field("recorded_on", &self.recorded_on)
            .field("cu_event", &self.cu_event)
            .finish()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        let _ = self
            .recorded_on
            .device()
            .release_pooled_cu_event(self.cu_event);
    }
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
