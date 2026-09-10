use std::{
    mem::MaybeUninit,
    ops::RangeBounds,
    slice,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::sys::CUdeviceptr_v2;

use crate::{
    device_memory::{
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
    error::Result,
    event::Event,
    host_pinned_memory_pool::{HostPinnedMemoryPool, RawHostPinnedAllocation},
    memory_slice::{MemorySlice, MemorySliceMut},
    stream::Stream,
};

#[derive(Debug)]
pub struct HostPinnedMemory(Arc<Inner>);

impl HostPinnedMemory {
    pub(crate) fn new_zeroed(len: usize, pool: &HostPinnedMemoryPool) -> Result<Self> {
        let allocation = unsafe { pool.alloc_raw(len)? };

        unsafe {
            let slice = slice::from_raw_parts_mut(
                allocation.host_ptr.as_ptr().cast::<MaybeUninit<u8>>(),
                allocation.len,
            );
            slice.fill(MaybeUninit::new(0));
        }

        Ok(Self(Arc::new(Inner {
            pool: pool.clone(),
            allocation,
            state: Mutex::new(State::HostExclusive),
        })))
    }

    fn make_host_exclusive_blocking(&self) -> Result<()> {
        let mut state = self.0.state.lock().unwrap();

        match &*state {
            State::HostExclusive => {}
            State::DeviceExclusive { done_event, .. } => {
                done_event.wait().block_on()?;
            }
            State::Shared {
                reader_done_events, ..
            } => {
                for done_event in reader_done_events {
                    done_event.wait().block_on()?;
                }
            }
        }

        *state = State::HostExclusive;

        Ok(())
    }

    fn make_shared_blocking(&self) -> Result<()> {
        let mut state = self.0.state.lock().unwrap();

        match &*state {
            State::HostExclusive => {}
            State::DeviceExclusive { done_event, .. } => {
                done_event.wait().block_on()?;
            }
            State::Shared { .. } => return Ok(()),
        }

        *state = State::Shared {
            prev_writer_done_event: None, // we already observed writer completion, so no need to keep
            prev_writer_stream: None,     // ^^^
            reader_done_events: Vec::new(),
            reader_streams: Vec::new(),
        };

        Ok(())
    }

    /// Waits for the device to no longer have exclusive access to
    /// the memory (if needed), then returns the data
    /// as a slice.
    pub fn view(&self) -> Result<&[u8]> {
        self.make_shared_blocking()?;
        unsafe {
            Ok(slice::from_raw_parts(
                self.0.allocation.host_ptr.as_ptr(),
                self.0.allocation.len,
            ))
        }
    }

    /// Waits for the device to no longer be accessing
    /// the memory (if needed), then returns the data
    /// as a slice.
    pub fn view_mut(&mut self) -> Result<&mut [u8]> {
        self.make_host_exclusive_blocking()?;
        unsafe {
            Ok(slice::from_raw_parts_mut(
                self.0.allocation.host_ptr.as_ptr(),
                self.0.allocation.len,
            ))
        }
    }

    #[must_use]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> MemorySlice<&'_ Self> {
        MemorySlice::new(self, range)
    }

    #[must_use]
    pub fn slice_mut(&mut self, range: impl RangeBounds<usize>) -> MemorySliceMut<&'_ mut Self> {
        MemorySliceMut::new(self, range)
    }
}

impl Drop for HostPinnedMemory {
    fn drop(&mut self) {
        let state = self.0.state.lock().unwrap();

        match &*state {
            State::HostExclusive => {
                // Can just deallocate immediately; allow Inner to drop.
            }
            State::DeviceExclusive { done_event, .. } => {
                // Need to keep Inner alive until
                // device is done using it.
                let inner = self.0.clone();
                done_event.wait().on_completion_observed(move || {
                    drop(inner);
                    Ok(())
                });
            }
            State::Shared {
                prev_writer_done_event,
                reader_done_events,
                ..
            } => {
                // Need to keep Inner alive until
                // device is done using it.
                if let Some(writer_done_event) = prev_writer_done_event {
                    let inner = self.0.clone();
                    writer_done_event.on_completion_observed(move || {
                        drop(inner);
                        Ok(())
                    });
                }
                for done_event in reader_done_events {
                    let inner: Arc<Inner> = self.0.clone();
                    done_event.wait().on_completion_observed(move || {
                        drop(inner);
                        Ok(())
                    });
                }
            }
        }
    }
}

impl DeviceVisibleMemory for HostPinnedMemory {
    type Guard<'a> = HostPinnedMemoryGuard<'a>;

    fn len(&self) -> usize {
        self.0.allocation.len
    }

    fn access_on<'a>(&'a self, stream: &'a Stream) -> Result<Self::Guard<'a>> {
        let mut state = self.0.state.lock().unwrap();

        let target_done_event = match &mut *state {
            State::HostExclusive => {
                let target_done_event = Arc::new(OnceLock::new());
                *state = State::Shared {
                    prev_writer_done_event: None,
                    reader_done_events: vec![target_done_event.clone()],
                    prev_writer_stream: None,
                    reader_streams: vec![stream.clone()],
                };
                target_done_event
            }
            State::DeviceExclusive {
                done_event,
                writer_stream,
            } => {
                if writer_stream != stream {
                    let done_event = Arc::new(OnceLock::<Event>::new());
                    *state = State::Shared {
                        prev_writer_stream: Some(writer_stream.clone()),
                        prev_writer_done_event: Some(done_event.wait().clone()),
                        reader_done_events: vec![done_event.clone()],
                        reader_streams: vec![stream.clone()],
                    };
                    done_event
                } else {
                    *done_event = Arc::new(OnceLock::new());
                    done_event.clone()
                }
            }
            State::Shared {
                prev_writer_stream,
                prev_writer_done_event,
                reader_done_events,
                reader_streams,
            } => {
                let done_event = Arc::new(OnceLock::new());
                reader_done_events.push(done_event.clone());
                if prev_writer_stream.as_ref() != Some(stream) && !reader_streams.contains(stream) {
                    if let Some(prev_writer_done_event) = prev_writer_done_event.as_ref() {
                        stream.wait_on_event(prev_writer_done_event)?;
                    }
                    reader_streams.push(stream.clone());
                }

                done_event
            }
        };

        Ok(HostPinnedMemoryGuard {
            memory: self,
            stream,
            done_event: target_done_event,
        })
    }
}

impl DeviceVisibleMemoryMut for HostPinnedMemory {
    type GuardMut<'a> = HostPinnedMemoryGuardMut<'a>;

    fn access_mut_on<'a>(&'a mut self, stream: &'a Stream) -> Result<Self::GuardMut<'a>> {
        let mut state = self.0.state.lock().unwrap();

        let target_done_event = Arc::new(OnceLock::new());
        match &mut *state {
            State::HostExclusive => {
                *state = State::DeviceExclusive {
                    writer_stream: stream.clone(),
                    done_event: target_done_event.clone(),
                };
            }
            State::DeviceExclusive {
                writer_stream,
                done_event,
            } => {
                if stream != writer_stream {
                    stream.wait_on_event(done_event.wait())?;
                    *writer_stream = stream.clone();
                }
                *done_event = target_done_event.clone();
            }
            State::Shared {
                prev_writer_done_event,
                reader_done_events,
                ..
            } => {
                if reader_done_events.is_empty() {
                    if let Some(prev_writer_done_event) = &*prev_writer_done_event {
                        stream.wait_on_event(prev_writer_done_event)?;
                    }
                } else {
                    for reader_done_event in reader_done_events {
                        stream.wait_on_event(reader_done_event.wait())?;
                    }
                }

                *state = State::DeviceExclusive {
                    writer_stream: stream.clone(),
                    done_event: target_done_event.clone(),
                };
            }
        }

        Ok(HostPinnedMemoryGuardMut {
            memory: self,
            stream,
            done_event: target_done_event,
        })
    }
}

#[derive(Debug)]
struct Inner {
    pool: HostPinnedMemoryPool,
    allocation: RawHostPinnedAllocation,
    state: Mutex<State>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.pool.dealloc_raw(self.allocation);
        }
    }
}

#[derive(Debug)]
enum State {
    HostExclusive,
    DeviceExclusive {
        writer_stream: Stream,
        done_event: Arc<OnceLock<Event>>,
    },
    Shared {
        prev_writer_stream: Option<Stream>,
        prev_writer_done_event: Option<Event>,
        reader_done_events: Vec<Arc<OnceLock<Event>>>,
        reader_streams: Vec<Stream>,
    },
}

/// Note that failing to invoke `Drop` for objects
/// of this type will cause an eventual hang.
#[derive(Debug)]
pub struct HostPinnedMemoryGuard<'a> {
    memory: &'a HostPinnedMemory,
    stream: &'a Stream,
    done_event: Arc<OnceLock<Event>>,
}

impl<'a> DevicePtrGuard<'a> for HostPinnedMemoryGuard<'a> {
    unsafe fn ptr(&self) -> CUdeviceptr_v2 {
        self.memory.0.allocation.device_ptr
    }
}

impl<'a> Drop for HostPinnedMemoryGuard<'a> {
    fn drop(&mut self) {
        if let Ok(event) = self.stream.record_event() {
            self.done_event.set(event).ok().unwrap();
        }
    }
}

/// Note that failing to invoke `Drop` for objects
/// of this type will cause an eventual hang.
#[derive(Debug)]
pub struct HostPinnedMemoryGuardMut<'a> {
    memory: &'a HostPinnedMemory,
    stream: &'a Stream,
    done_event: Arc<OnceLock<Event>>,
}

impl<'a> DevicePtrGuard<'a> for HostPinnedMemoryGuardMut<'a> {
    unsafe fn ptr(&self) -> CUdeviceptr_v2 {
        self.memory.0.allocation.device_ptr
    }
}

impl<'a> DevicePtrGuardMut<'a> for HostPinnedMemoryGuardMut<'a> {
    unsafe fn ptr(&mut self) -> CUdeviceptr_v2 {
        self.memory.0.allocation.device_ptr
    }
}

impl<'a> Drop for HostPinnedMemoryGuardMut<'a> {
    fn drop(&mut self) {
        if let Ok(event) = self.stream.record_event() {
            self.done_event.set(event).ok().unwrap();
        }
    }
}
