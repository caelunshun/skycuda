use std::{
    mem::MaybeUninit,
    slice,
    sync::{Arc, Mutex},
};

use crate::{
    device::Device,
    error::Result,
    event::Event,
    host_pinned_memory_pool::{HostPinnedAllocation, HostPinnedMemoryPool},
    stream::Stream,
};

#[derive(Debug)]
pub struct HostPinnedMemory(Arc<Inner>);

impl HostPinnedMemory {
    pub(crate) fn new_zeroed(len: usize, pool: &HostPinnedMemoryPool) -> Result<Self> {
        let allocation = unsafe { pool.alloc(len)? };

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

    fn make_host_exclusive(&self) -> Result<()> {
        let mut state = self.0.state.lock().unwrap();

        match &*state {
            State::HostExclusive => return Ok(()),
            State::DeviceExclusive { done_event } => {
                done_event.block_on()?;
            }
            State::Shared { done_events } => {
                for done_event in done_events {
                    done_event.block_on()?;
                }
            }
        }

        Ok(())
    }
}

impl Drop for HostPinnedMemory {
    fn drop(&mut self) {
        let state = self.0.state.lock().unwrap();

        match &*state {
            State::HostExclusive => {
                // Can just deallocate immediately; allow Inner to drop.
            }
            State::DeviceExclusive { done_event } => {
                // Need to keep Inner alive until
                // device is done using it.
                let inner = self.0.clone();
                done_event.on_complete(move || {
                    drop(inner);
                    Ok(())
                });
            }
            State::Shared { done_events } => {
                // Need to keep Inner alive until
                // device is done using it.
                for done_event in done_events {
                    let inner = self.0.clone();
                    done_event.on_complete(move || {
                        drop(inner);
                        Ok(())
                    });
                }
            }
        }
    }
}

#[derive(Debug)]
struct Inner {
    pool: HostPinnedMemoryPool,
    allocation: HostPinnedAllocation,
    state: Mutex<State>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            let _ = self.pool.dealloc(self.allocation);
        }
    }
}

#[derive(Debug)]
enum State {
    HostExclusive,
    DeviceExclusive { done_event: Event },
    Shared { done_events: Vec<Event> },
}
