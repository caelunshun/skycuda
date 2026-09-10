use std::ops::RangeBounds;

use cudarc::driver::sys::CUdeviceptr_v2;

use crate::device_memory::{
    DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
};

/// Read-only slice of device-visible memory.
pub struct MemorySlice<T> {
    memory: T,
    offset: usize,
    len: usize,
}

impl<T> MemorySlice<T>
where
    T: DeviceVisibleMemory,
{
    pub fn new(memory: T, range: impl RangeBounds<usize>) -> Self {
        let offset = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        assert!(offset < memory.len());
        let end = match range.end_bound() {
            std::ops::Bound::Included(x) => *x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => memory.len(),
        };
        let len = end.checked_sub(offset).unwrap();

        Self {
            memory,
            offset,
            len,
        }
    }
}

impl<T> DeviceVisibleMemory for MemorySlice<T>
where
    T: DeviceVisibleMemory,
{
    type Guard<'b>
        = MemorySliceGuard<T::Guard<'b>>
    where
        T: 'b;

    fn len(&self) -> usize {
        self.len
    }

    fn access_on<'b>(
        &'b self,
        stream: &'b crate::stream::Stream,
    ) -> crate::error::Result<Self::Guard<'b>> {
        self.memory.access_on(stream).map(|inner| MemorySliceGuard {
            inner,
            offset: self.offset,
        })
    }
}

pub struct MemorySliceGuard<T> {
    inner: T,
    offset: usize,
}

/// Mutable slice of device-visible memory.
pub struct MemorySliceMut<T> {
    memory: T,
    offset: usize,
    len: usize,
}

impl<T> MemorySliceMut<T>
where
    T: DeviceVisibleMemoryMut,
{
    pub(crate) fn new(memory: T, range: impl RangeBounds<usize>) -> Self {
        let offset = match range.start_bound() {
            std::ops::Bound::Included(x) => *x,
            std::ops::Bound::Excluded(x) => *x + 1,
            std::ops::Bound::Unbounded => 0,
        };
        assert!(offset < memory.len());
        let end = match range.end_bound() {
            std::ops::Bound::Included(x) => *x + 1,
            std::ops::Bound::Excluded(x) => *x,
            std::ops::Bound::Unbounded => memory.len(),
        };
        let len = end.checked_sub(offset).unwrap();

        Self {
            memory,
            offset,
            len,
        }
    }
}

impl<T> DeviceVisibleMemory for MemorySliceMut<T>
where
    T: DeviceVisibleMemoryMut,
{
    type Guard<'b>
        = MemorySliceGuard<T::Guard<'b>>
    where
        T: 'b;

    fn len(&self) -> usize {
        self.len
    }

    fn access_on<'b>(
        &'b self,
        stream: &'b crate::stream::Stream,
    ) -> crate::error::Result<Self::Guard<'b>> {
        self.memory.access_on(stream).map(|inner| MemorySliceGuard {
            inner,
            offset: self.offset,
        })
    }
}

impl<T> DeviceVisibleMemoryMut for MemorySliceMut<T>
where
    T: DeviceVisibleMemoryMut,
{
    type GuardMut<'b>
        = MemorySliceGuardMut<T::GuardMut<'b>>
    where
        T: 'b;

    fn access_mut_on<'b>(
        &'b mut self,
        stream: &'b crate::stream::Stream,
    ) -> crate::error::Result<Self::GuardMut<'b>> {
        self.memory
            .access_mut_on(stream)
            .map(|inner| MemorySliceGuardMut {
                inner,
                offset: self.offset,
            })
    }
}

pub struct MemorySliceGuardMut<T> {
    inner: T,
    offset: usize,
}

impl<'a, T> DevicePtrGuard<'a> for MemorySliceGuard<T>
where
    T: DevicePtrGuard<'a>,
{
    unsafe fn ptr(&self) -> CUdeviceptr_v2 {
        let base = unsafe { self.inner.ptr() };
        base + self.offset as u64
    }
}

impl<'a, T> DevicePtrGuard<'a> for MemorySliceGuardMut<T>
where
    T: DevicePtrGuard<'a>,
{
    unsafe fn ptr(&self) -> CUdeviceptr_v2 {
        let base = unsafe { self.inner.ptr() };
        base + self.offset as u64
    }
}

impl<'a, T> DevicePtrGuardMut<'a> for MemorySliceGuardMut<T>
where
    T: DevicePtrGuardMut<'a>,
{
    unsafe fn ptr(&mut self) -> CUdeviceptr_v2 {
        let base = unsafe { self.inner.ptr() };
        base + self.offset as u64
    }
}
