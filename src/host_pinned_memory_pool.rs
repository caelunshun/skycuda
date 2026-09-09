use std::{
    ptr::{self, NonNull},
    sync::{Arc, Mutex},
};

use cudarc::driver::sys::{
    CUdeviceptr_v2, cuMemAllocHost_v2, cuMemFreeHost, cuMemHostGetDevicePointer_v2,
};
use range_alloc::RangeAllocator;

use crate::{device::Device, error::Result};

#[derive(Debug, Clone)]
pub struct HostPinnedMemoryPool(Arc<Mutex<Inner>>);

impl HostPinnedMemoryPool {
    pub const GUARANTEED_ALIGNMENT: usize = 128;
    const MIN_CHUNK_SIZE: usize = 4 * 1024 * 1024;

    pub(crate) fn new(options: HostPinnedMemoryPoolOptions, device: &Device) -> Result<Self> {
        Ok(Self(Arc::new(Mutex::new(Inner {
            device: device.clone(),
            options,
            chunks: Vec::new(),
            free_bytes: 0,
        }))))
    }

    pub(crate) unsafe fn alloc(&self, len: usize) -> Result<HostPinnedAllocation> {
        assert!(len > 0);
        let mut inner = self.0.lock().unwrap();

        for chunk in &mut inner.chunks {
            if let Ok(range) = chunk
                .allocator
                .allocate_range_aligned(len, Self::GUARANTEED_ALIGNMENT)
            {
                let allocation = HostPinnedAllocation {
                    host_ptr: unsafe { chunk.host_ptr.add(range.start) },
                    device_ptr: chunk.device_ptr + range.start as u64,
                    len,
                };
                inner.free_bytes -= len;
                return Ok(allocation);
            }
        }

        let new_chunk_size = len.max(Self::MIN_CHUNK_SIZE).next_power_of_two();

        let mut host_ptr = ptr::null_mut();
        unsafe {
            cuMemAllocHost_v2(&mut host_ptr, new_chunk_size).result()?;
        }

        let mut device_ptr = 0;
        unsafe {
            cuMemHostGetDevicePointer_v2(&mut device_ptr, host_ptr, 0).result()?;
        }

        let mut chunk = Chunk {
            host_ptr: NonNull::new(host_ptr.cast()).unwrap(),
            device_ptr,
            len,
            allocator: RangeAllocator::new(0..new_chunk_size),
        };
        let range = chunk
            .allocator
            .allocate_range_aligned(len, Self::GUARANTEED_ALIGNMENT)
            .unwrap();
        inner.free_bytes += new_chunk_size - len;

        let allocation = HostPinnedAllocation {
            host_ptr: unsafe { chunk.host_ptr.add(range.start) },
            device_ptr: chunk.device_ptr + range.start as u64,
            len,
        };

        inner.chunks.push(chunk);

        Ok(allocation)
    }

    pub(crate) unsafe fn dealloc(&self, allocation: HostPinnedAllocation) -> Result<()> {
        let mut inner = self.0.lock().unwrap();

        let chunk = inner
            .chunks
            .iter_mut()
            .find(|chunk| {
                allocation.device_ptr >= chunk.device_ptr
                    && allocation.device_ptr < (chunk.device_ptr + chunk.len as u64)
            })
            .unwrap();

        let start_offset = (allocation.device_ptr - chunk.device_ptr) as usize;
        let end_offset = start_offset + allocation.len;
        chunk.allocator.free_range(start_offset..end_offset);
        inner.free_bytes += allocation.len;

        while inner.free_bytes > inner.options.max_reserved_free_bytes {
            let free_chunk_index = inner.chunks.iter().position(|c| c.allocator.is_empty());
            match free_chunk_index {
                Some(i) => {
                    let chunk = inner.chunks.swap_remove(i);
                    inner.free_bytes -= chunk.len;
                }
                None => break, // can't free due to fragmentation
            }
        }

        Ok(())
    }
}

#[derive(Debug)]
struct Inner {
    device: Device,
    options: HostPinnedMemoryPoolOptions,
    chunks: Vec<Chunk>,
    free_bytes: usize,
}

#[derive(Debug, Clone)]
pub struct HostPinnedMemoryPoolOptions {
    pub max_reserved_free_bytes: usize,
}

#[derive(Debug)]
struct Chunk {
    host_ptr: NonNull<u8>,
    device_ptr: CUdeviceptr_v2,
    len: usize,
    allocator: RangeAllocator<usize>,
}

impl Drop for Chunk {
    fn drop(&mut self) {
        unsafe {
            let _ = cuMemFreeHost(self.host_ptr.as_ptr().cast()).result();
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct HostPinnedAllocation {
    pub host_ptr: NonNull<u8>,
    pub device_ptr: CUdeviceptr_v2,
    pub len: usize,
}
