use crate::{device::Device, error::Result, module::Module};
use cudarc::driver::sys::{
    CUfunction, CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
    cuFuncGetParamCount, cuFuncGetParamInfo, cuFuncLoad, cuFuncSetAttribute, cuModuleGetFunction,
    cuOccupancyMaxActiveBlocksPerMultiprocessor,
};
use std::{ffi::CString, ptr, sync::Arc};

#[derive(Debug, Clone)]
pub struct Function(Arc<Inner>);

impl Function {
    pub(crate) fn new(symbol_name: &str, module: &Module) -> Result<Self> {
        let mut cu_function = ptr::null_mut();
        let symbol_name = CString::new(symbol_name)?;
        unsafe {
            cuModuleGetFunction(&mut cu_function, module.cu_module(), symbol_name.as_ptr())
                .result()?;
        }
        assert!(!cu_function.is_null());

        module.device().set_as_current()?;
        unsafe {
            cuFuncLoad(cu_function).result()?;
        }

        let mut param_count = 0;
        unsafe {
            cuFuncGetParamCount(cu_function, &mut param_count).result()?;
        }

        let mut param_sizes = Vec::new();
        for param_idx in 0..param_count {
            let mut param_size = 0;
            let mut param_offset = 0;
            unsafe {
                cuFuncGetParamInfo(cu_function, param_idx, &mut param_offset, &mut param_size)
                    .result()?;
            }
            assert!(param_size > 0);
            param_sizes.push(param_size);
        }

        Ok(Self(Arc::new(Inner {
            module: module.clone(),
            cu_function,
            param_sizes,
        })))
    }

    pub fn module(&self) -> &Module {
        &self.0.module
    }

    pub fn device(&self) -> &Device {
        self.0.module.device()
    }

    pub fn cu_function(&self) -> CUfunction {
        self.0.cu_function
    }

    pub fn set_max_dynamic_smem(&self, max_bytes: usize) -> Result<()> {
        unsafe {
            cuFuncSetAttribute(
                self.cu_function(),
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                max_bytes.try_into()?,
            )
            .result()?;
        }
        Ok(())
    }

    pub fn max_active_blocks_per_sm(&self, block_size: u32, dynamic_smem: usize) -> Result<usize> {
        self.device().set_as_current()?;
        let mut num_blocks = 0;
        unsafe {
            cuOccupancyMaxActiveBlocksPerMultiprocessor(
                &mut num_blocks,
                self.0.cu_function,
                block_size.try_into()?,
                dynamic_smem,
            )
            .result()?;
        }
        Ok(num_blocks.try_into()?)
    }

    pub fn num_params(&self) -> usize {
        self.param_sizes().len()
    }

    pub fn param_sizes(&self) -> &[usize] {
        &self.0.param_sizes
    }
}

#[derive(Debug)]
struct Inner {
    module: Module,
    cu_function: CUfunction,
    param_sizes: Vec<usize>,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
