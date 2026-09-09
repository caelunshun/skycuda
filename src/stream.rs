use std::{ptr, sync::Arc};

use cudarc::driver::sys::{
    CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
    CUresult::{CUDA_ERROR_NOT_READY, CUDA_SUCCESS},
    CUstream,
    CUstream_flags::CU_STREAM_NON_BLOCKING,
    cuEventRecord, cuStreamCreate, cuStreamDestroy_v2, cuStreamQuery, cuStreamSynchronize,
    cuStreamWaitEvent,
};

use crate::{device::Device, error::Result, event::Event};

#[derive(Debug, Clone)]
pub struct Stream(Arc<Inner>);

impl Stream {
    pub(crate) fn new(device: &Device) -> Result<Self> {
        device.set_as_current()?;
        let mut cu_stream = ptr::null_mut();
        unsafe {
            cuStreamCreate(&mut cu_stream, CU_STREAM_NON_BLOCKING as u32).result()?;
        }

        Ok(Self(Arc::new(Inner {
            device: device.clone(),
            cu_stream,
        })))
    }

    pub fn record_event(&self, event: &Event) -> Result<()> {
        unsafe {
            cuEventRecord(event.cu_event(), self.0.cu_stream).result()?;
        }
        Ok(())
    }

    pub fn wait_on_event(&self, event: &Event) -> Result<()> {
        unsafe {
            cuStreamWaitEvent(
                self.0.cu_stream,
                event.cu_event(),
                CU_EVENT_WAIT_DEFAULT as u32,
            )
            .result()?;
        }
        Ok(())
    }

    /// Blocks the current thread on waiting for all operations
    /// (at least those enqueued when this function
    /// is called) to be completed.
    pub fn block_on(&self) -> Result<()> {
        unsafe {
            cuStreamSynchronize(self.0.cu_stream).result()?;
        }
        Ok(())
    }

    /// Returns whether all operations enqueued to the stream
    /// so far have completed.
    pub fn have_all_ops_completed(&self) -> Result<bool> {
        let res = unsafe { cuStreamQuery(self.0.cu_stream) };

        match res {
            CUDA_SUCCESS => Ok(true),
            CUDA_ERROR_NOT_READY => Ok(false),
            e => e.result().map_err(From::from).map(|_| unreachable!()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.0.device
    }

    pub fn cu_stream(&self) -> CUstream {
        self.0.cu_stream
    }
}

#[derive(Debug)]
struct Inner {
    device: Device,
    cu_stream: CUstream,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = cuStreamDestroy_v2(self.cu_stream).result();
        }
    }
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
