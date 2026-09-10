use std::{
    any::type_name_of_val,
    ptr::{self, NonNull},
};

use bumpalo::Bump;
use bytemuck::Pod;
use cudarc::driver::sys::{CUdeviceptr_v2, cuLaunchKernel};
use half::{bf16, f16};

use crate::{
    device_memory::{
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
    error::Result,
    function::Function,
    stream::Stream,
};

pub struct LaunchBuilder<'a> {
    stream: &'a Stream,
    function: &'a Function,
    bump: Bump,
    /// Each argument is a pointer into a bump allocator chunk.
    /// Note that these pointers are stable even when the Bump is moved.
    args: Vec<NonNull<u8>>,
    arg_guards: Vec<Box<dyn Anything + 'a>>,
    dynamic_smem: usize,
    block_size: (u32, u32, u32),
}

trait Anything {}
impl<T> Anything for T {}

impl<'a> LaunchBuilder<'a> {
    pub(crate) fn new(function: &'a Function, stream: &'a Stream) -> Self {
        Self {
            stream,
            function,
            bump: Bump::new(),
            args: Vec::new(),
            arg_guards: Vec::new(),
            dynamic_smem: 0,
            block_size: (1, 1, 1),
        }
    }

    pub fn arg(&mut self, arg: impl KernelArg<'a>) -> Result<&mut Self> {
        let (repr, guard) = unsafe { arg.into_arg(self.stream)? };

        let arg_index = self.arg_guards.len();
        assert!(
            arg_index < self.function.num_params(),
            "too many arguments to kernel function (expected exactly {})",
            self.function.num_params()
        );

        assert_eq!(
            size_of_val(&repr),
            self.function.param_sizes()[arg_index],
            "kernel function argument size mismatch at index {arg_index}: provided value (type {}) has size {}, but function expects size {}",
            type_name_of_val(&repr),
            size_of_val(&repr),
            self.function.param_sizes()[arg_index]
        );

        self.args
            .push(NonNull::from_mut(self.bump.alloc(repr)).cast());
        self.arg_guards.push(Box::new(guard));
        Ok(self)
    }

    pub fn dynamic_smem(&mut self, dynamic_smem: usize) -> Result<&mut Self> {
        self.dynamic_smem = dynamic_smem;
        Ok(self)
    }

    pub fn block_size(&mut self, block_x: u32, block_y: u32, block_z: u32) -> Result<&mut Self> {
        self.block_size = (block_x, block_y, block_z);
        Ok(self)
    }

    /// # Safety
    /// This executes arbitrary code on the GPU, which may also be
    /// able to access host memory.
    pub unsafe fn launch(&mut self, grid_x: u32, grid_y: u32, grid_z: u32) -> Result<()> {
        unsafe {
            cuLaunchKernel(
                self.function.cu_function(),
                grid_x,
                grid_y,
                grid_z,
                self.block_size.0,
                self.block_size.1,
                self.block_size.2,
                self.dynamic_smem.try_into()?,
                self.stream.cu_stream(),
                self.args.as_mut_ptr().cast(),
                ptr::null_mut(),
            )
            .result()?;
        }
        Ok(())
    }
}

pub trait KernelArg<'a> {
    type Repr: Pod;
    type Guard: 'a;

    /// # Safety
    /// May return raw device pointers, which can only
    /// be used by issues onto the specified stream with restricted
    /// types of access, and only issues made before the guard is dropped.
    unsafe fn into_arg(self, stream: &'a Stream) -> Result<(Self::Repr, Self::Guard)>;
}

// impl<'a, T> KernelArg<'a> for T
// where
//     T: Pod,
// {
//     type Repr = Self;
//     type Guard = ();

//     unsafe fn into_arg(self, _stream: &'a Stream) -> Result<(Self::Repr, Self::Guard)> {
//         Ok((self, ()))
//     }
// }

impl<'a, T> KernelArg<'a> for &'a T
where
    T: DeviceVisibleMemory + 'a,
{
    type Repr = CUdeviceptr_v2;
    type Guard = T::Guard<'a>;

    unsafe fn into_arg(self, stream: &'a Stream) -> Result<(Self::Repr, Self::Guard)> {
        let guard = self.access_on(stream)?;
        unsafe { Ok((guard.ptr(), guard)) }
    }
}

impl<'a, T> KernelArg<'a> for &'a mut T
where
    T: DeviceVisibleMemoryMut + 'a,
{
    type Repr = CUdeviceptr_v2;
    type Guard = T::GuardMut<'a>;

    unsafe fn into_arg(self, stream: &'a Stream) -> Result<(Self::Repr, Self::Guard)> {
        let mut guard = self.access_mut_on(stream)?;
        unsafe { Ok((guard.ptr(), guard)) }
    }
}

macro_rules! trivial_kernel_arg_impl {
    ($t:ty) => {
        impl<'a> KernelArg<'a> for $t {
            type Repr = Self;
            type Guard = ();

            unsafe fn into_arg(self, _stream: &'a Stream) -> Result<(Self::Repr, Self::Guard)> {
                Ok((self, ()))
            }
        }
    };
}

trivial_kernel_arg_impl!(u8);
trivial_kernel_arg_impl!(u16);
trivial_kernel_arg_impl!(u32);
trivial_kernel_arg_impl!(u64);

trivial_kernel_arg_impl!(i8);
trivial_kernel_arg_impl!(i16);
trivial_kernel_arg_impl!(i32);
trivial_kernel_arg_impl!(i64);

trivial_kernel_arg_impl!(bf16);
trivial_kernel_arg_impl!(f16);
trivial_kernel_arg_impl!(f32);
trivial_kernel_arg_impl!(f64);
