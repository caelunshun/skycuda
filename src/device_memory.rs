use std::sync::Mutex;

use cudarc::driver::sys::{
    CUdeviceptr_v2, cuMemAllocFromPoolAsync, cuMemFreeAsync, cuMemsetD8Async,
};

use crate::{
    device::Device, device_memory_pool::DeviceMemoryPool, error::Result, event::Event,
    stream::Stream,
};

/// Raw, untyped wrapper over a device-local memory allocation.
/// This type of memory is only accessible to the GPU.
///
/// All device allocations come from CUDA memory pools and use stream-ordered
/// allocation semantics (similar to cudaMallocAsync).
///
/// This wrapper tracks read and write accesses to the memory
/// and the streams they occur on, automatically inserting events
/// where needed to enforce program-ordered access when reads/writes
/// would conflict.
#[derive(Debug)]
pub struct DeviceMemory {
    mem_pool: DeviceMemoryPool,
    ptr: CUdeviceptr_v2,
    len: usize,
    state: Mutex<State>,
}

impl DeviceMemory {
    pub(crate) fn new_zeroed(
        len: usize,
        mem_pool: &DeviceMemoryPool,
        stream: &Stream,
    ) -> Result<Self> {
        let mut ptr = 0;
        if len > 0 {
            unsafe {
                cuMemAllocFromPoolAsync(&mut ptr, len, mem_pool.cu_mempool(), stream.cu_stream())
                    .result()?;
                cuMemsetD8Async(ptr, 0, len, stream.cu_stream()).result()?;
            }
        }

        let last_written_event = mem_pool.device().create_event()?;
        stream.record_event(&last_written_event)?;

        Ok(Self {
            mem_pool: mem_pool.clone(),
            ptr,
            len,
            state: Mutex::new(State {
                last_written_stream: stream.clone(),
                reading_by_streams: Vec::new(),
            }),
        })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn mem_pool(&self) -> &DeviceMemoryPool {
        &self.mem_pool
    }

    pub fn device(&self) -> &Device {
        self.mem_pool.device()
    }

    unsafe fn free(&self) -> Result<()> {
        let state = self.state.lock().unwrap();

        for stream in &state.reading_by_streams {
            let event = self.device().create_event()?;
            stream.record_event(&event)?;
            state.last_written_stream.wait_on_event(&event)?;
        }

        unsafe {
            cuMemFreeAsync(self.ptr, state.last_written_stream.cu_stream()).result()?;
        }

        Ok(())
    }
}

impl Drop for DeviceMemory {
    fn drop(&mut self) {
        if self.len > 0 {
            let _ = unsafe { self.free() };
        }
    }
}

#[derive(Debug)]
struct State {
    last_written_stream: Stream,
    reading_by_streams: Vec<Stream>,
}

impl DeviceVisibleMemory for DeviceMemory {
    type Guard<'a> = DeviceMemoryGuard<'a>;

    fn len(&self) -> usize {
        self.len
    }

    fn access_on<'a>(&'a self, stream: &'a Stream) -> Result<DeviceMemoryGuard<'a>> {
        assert_eq!(self.device(), stream.device());

        let mut state = self.state.lock().unwrap();

        if stream != &state.last_written_stream && !state.reading_by_streams.contains(&stream) {
            let event = self.device().create_event()?;
            state.last_written_stream.record_event(&event)?;
            stream.wait_on_event(&event)?;
            state.reading_by_streams.push(stream.clone());
        }

        Ok(DeviceMemoryGuard { memory: self })
    }
}

impl DeviceVisibleMemoryMut for DeviceMemory {
    type GuardMut<'a> = DeviceMemoryGuardMut<'a>;

    fn access_mut_on<'a>(&'a mut self, stream: &'a Stream) -> Result<DeviceMemoryGuardMut<'a>> {
        assert_eq!(self.device(), stream.device());
        let mut state = self.state.lock().unwrap();

        if state.reading_by_streams.is_empty() {
            if stream != &state.last_written_stream {
                let event = self.device().create_event()?;
                state.last_written_stream.record_event(&event)?;
                stream.wait_on_event(&event)?;
            }
        } else {
            for reader_stream in state.reading_by_streams.drain(..) {
                let event = self.device().create_event()?;
                reader_stream.record_event(&event)?;
                stream.wait_on_event(&event)?;
            }
        }

        state.last_written_stream = stream.clone();

        drop(state);

        Ok(DeviceMemoryGuardMut { memory: self })
    }
}

#[derive(Debug)]
pub struct DeviceMemoryGuard<'a> {
    memory: &'a DeviceMemory,
}

impl<'a> DevicePtrGuard<'a> for DeviceMemoryGuard<'a> {
    unsafe fn ptr(&self) -> CUdeviceptr_v2 {
        self.memory.ptr
    }
}

#[derive(Debug)]
pub struct DeviceMemoryGuardMut<'a> {
    memory: &'a mut DeviceMemory,
}

impl<'a> DevicePtrGuard<'a> for DeviceMemoryGuardMut<'a> {
    unsafe fn ptr(&self) -> CUdeviceptr_v2 {
        self.memory.ptr
    }
}

impl<'a> DevicePtrGuardMut<'a> for DeviceMemoryGuardMut<'a> {
    unsafe fn ptr(&mut self) -> CUdeviceptr_v2 {
        self.memory.ptr
    }
}

pub trait DeviceVisibleMemory {
    type Guard<'a>: DevicePtrGuard<'a>
    where
        Self: 'a;

    fn len(&self) -> usize;

    /// Introduces synchronization to ensure that read-only access to the memory on the given
    /// stream will not cause data races, by waiting on any previous (in program order) write accesses.
    fn access_on<'a>(&'a self, stream: &'a Stream) -> Result<Self::Guard<'a>>;
}

pub trait DeviceVisibleMemoryMut: DeviceVisibleMemory {
    type GuardMut<'a>: DevicePtrGuardMut<'a>
    where
        Self: 'a;

    /// Introduces synchronization to ensure that read- and write-access to the
    /// memory on the given stream will not cause data races, by waiting on any previous (in program
    /// order) read accesses.
    fn access_mut_on<'a>(&'a mut self, stream: &'a Stream) -> Result<Self::GuardMut<'a>>;
}

pub trait DevicePtrGuard<'a> {
    /// # Safety
    /// The returned pointer is only valid for issues
    /// to the associated stream during the lifetime
    /// of this access handle, and must only be used
    /// for reads.
    unsafe fn ptr(&self) -> CUdeviceptr_v2;
}

pub trait DevicePtrGuardMut<'a> {
    /// # Safety
    /// The returned pointer is only valid for issues
    /// to the associated stream during the lifetime
    /// of this access handle
    unsafe fn ptr(&mut self) -> CUdeviceptr_v2;
}
