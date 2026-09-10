use std::{ffi::CString, ptr, sync::Arc};

use cudarc::driver::sys::{CUfunction, cuFuncLoad, cuModuleGetFunction};

use crate::{device::Device, error::Result, module::Module};

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

        Ok(Self(Arc::new(Inner {
            module: module.clone(),
            cu_function,
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
}

#[derive(Debug)]
struct Inner {
    module: Module,
    cu_function: CUfunction,
}

unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}
