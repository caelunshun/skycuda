use std::{
    fmt::Debug,
    mem, ptr,
    sync::{Arc, Mutex},
};

use bytemuck::Pod;
use cudarc::driver::sys::{
    CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
    CUresult::{CUDA_ERROR_NOT_READY, CUDA_SUCCESS},
    CUstream,
    CUstream_flags::CU_STREAM_NON_BLOCKING,
    cuMemcpyDtoDAsync_v2, cuMemsetD8Async, cuMemsetD16Async, cuMemsetD32Async, cuStreamCreate,
    cuStreamDestroy_v2, cuStreamQuery, cuStreamSynchronize, cuStreamWaitEvent,
};

use crate::{
    device::Device,
    device_memory::{
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
    error::Result,
    event::Event,
    function::Function,
    launch::LaunchBuilder,
};

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
            events_observed: Mutex::new(Vec::new()),
        })))
    }

    pub fn record_event(&self) -> Result<Event> {
        let event = Event::record(self)?;
        self.0.events_observed.lock().unwrap().push(event.clone());
        Ok(event)
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

        self.0.events_observed.lock().unwrap().push(event.clone());

        Ok(())
    }

    /// Requires that the size of `T` divides the memory size
    /// and is either 1, 2, or 4.
    pub fn fill<T: Pod>(&self, memory: &mut impl DeviceVisibleMemoryMut, value: T) -> Result<()> {
        const {
            assert!(matches!(size_of::<T>(), 1 | 2 | 4));
        }

        assert!(memory.len().is_multiple_of(size_of::<T>()));
        let num_elements = memory.len() / size_of::<T>();

        let mut guard = memory.access_mut_on(self)?;
        unsafe {
            match size_of::<T>() {
                1 => {
                    cuMemsetD8Async(
                        guard.ptr(),
                        bytemuck::cast(value),
                        num_elements,
                        self.cu_stream(),
                    )
                    .result()?;
                }
                2 => {
                    cuMemsetD16Async(
                        guard.ptr(),
                        bytemuck::cast(value),
                        num_elements,
                        self.cu_stream(),
                    )
                    .result()?;
                }
                4 => {
                    cuMemsetD32Async(
                        guard.ptr(),
                        bytemuck::cast(value),
                        num_elements,
                        self.cu_stream(),
                    )
                    .result()?;
                }
                _ => unreachable!(),
            }
        }
        Ok(())
    }

    /// Requires that source and destination memories have the same length.
    pub fn copy(
        &self,
        src: &impl DeviceVisibleMemory,
        dst: &mut impl DeviceVisibleMemoryMut,
    ) -> Result<()> {
        let len = src.len();
        assert_eq!(dst.len(), len);

        let src_guard = src.access_on(self)?;
        let mut dst_guard = dst.access_mut_on(self)?;

        unsafe {
            cuMemcpyDtoDAsync_v2(dst_guard.ptr(), src_guard.ptr(), len, self.cu_stream())
                .result()?;
        }

        Ok(())
    }

    pub fn launch_builder<'a>(&'a self, function: &'a Function) -> LaunchBuilder<'a> {
        LaunchBuilder::new(function, self)
    }

    /// Blocks the current thread on waiting for all operations
    /// (at least those enqueued when this function
    /// is called) to be completed.
    pub fn block_on(&self) -> Result<()> {
        unsafe {
            cuStreamSynchronize(self.0.cu_stream).result()?;
        }
        self.notify_completion_observed()?;
        Ok(())
    }

    /// Returns whether all operations enqueued to the stream
    /// so far have completed.
    pub fn have_all_ops_completed(&self) -> Result<bool> {
        let res = unsafe { cuStreamQuery(self.0.cu_stream) };

        match res {
            CUDA_SUCCESS => {
                self.notify_completion_observed()?;
                Ok(true)
            }
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

    pub(crate) fn notify_completion_observed(&self) -> Result<()> {
        let events = mem::take(&mut *self.0.events_observed.lock().unwrap());
        for event in events {
            event.notify_completion_observed()?;
        }
        Ok(())
    }

    pub(crate) fn notify_completion_observed_up_to(&self, event: &Event) -> Result<()> {
        let mut events = self.0.events_observed.lock().unwrap();
        let Some(i) = events.iter().position(|e| e == event) else {
            return Ok(());
        };

        let completed_events = events.drain(..=i).collect::<Vec<_>>();
        drop(events);

        for event in completed_events {
            event.notify_completion_observed()?;
        }

        Ok(())
    }
}

impl PartialEq for Stream {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Stream {}

#[derive(Debug)]
struct Inner {
    device: Device,
    cu_stream: CUstream,
    events_observed: Mutex<Vec<Event>>,
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
