use std::sync::Arc;
use std::{ffi::CString, sync::Once};
use std::{mem, ptr};

use cudarc::driver::CudaStream;
use cudarc::driver::sys::CUdeviceptr;
use cudnn_frontend_sys as sys;

mod enums;
mod error;

pub use enums::*;
pub use error::{Error, ErrorCode, Result};

use error::check;

pub fn load_cudnn_library(path: Option<&str>) -> Result<()> {
    static LOADED: Once = Once::new();
    let mut result = Ok(());
    LOADED.call_once(|| {
        let c = path.map(|p| CString::new(p).unwrap());
        let ptr = c.as_ref().map_or(ptr::null(), |c| c.as_ptr());
        result = check(unsafe { sys::cudnn_fe_load_library(ptr) });
    });
    result
}

pub fn backend_version() -> i64 {
    unsafe { sys::cudnn_fe_backend_version() }
}

pub struct CudnnHandle {
    stream: Arc<CudaStream>,
    raw: *mut sys::cudnnFeHandle,
}

impl CudnnHandle {
    pub fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_handle_create(stream.cu_stream().cast(), &mut raw) })?;
        Ok(CudnnHandle {
            raw,
            stream: stream.clone(),
        })
    }

    pub fn set_stream(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        check(unsafe { sys::cudnn_fe_handle_set_stream(self.raw, stream.cu_stream().cast()) })?;
        self.stream = stream.clone();
        Ok(())
    }

    pub(crate) fn as_ptr(&self) -> *mut sys::cudnnFeHandle {
        self.raw
    }
}

impl Drop for CudnnHandle {
    fn drop(&mut self) {
        unsafe { sys::cudnn_fe_handle_destroy(self.raw) };
    }
}

unsafe impl Send for CudnnHandle {}

/// Abstract tensor in a graph.
#[derive(Clone, Copy)]
pub struct Tensor<'g, 'h> {
    raw: *mut sys::cudnnFeTensor,
    graph: &'g GraphBuilder<'h>,
}

impl<'g, 'h> Tensor<'g, 'h> {
    pub(crate) unsafe fn from_raw(
        raw: *mut sys::cudnnFeTensor,
        graph: &'g GraphBuilder<'h>,
    ) -> Self {
        Tensor { raw, graph }
    }

    pub(crate) fn as_ptr(&self) -> *const sys::cudnnFeTensor {
        self.raw as *const _
    }

    pub fn uid(&self) -> Result<i64> {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_tensor_get_uid(self.as_ptr(), &mut out) })?;
        Ok(out)
    }

    pub fn set_uid(&self, uid: i64) -> Result<()> {
        check(unsafe { sys::cudnn_fe_tensor_set_uid(self.raw, uid) })
    }

    pub fn set_output(&self, is_output: bool) -> Result<()> {
        check(unsafe { sys::cudnn_fe_tensor_set_output(self.raw, is_output as i32) })
    }

    pub fn set_data_type(&self, dt: DataType) -> Result<()> {
        check(unsafe { sys::cudnn_fe_tensor_set_data_type(self.raw, dt.to_ffi()) })
    }

    pub fn set_dim(&self, dim: &[i64]) -> Result<()> {
        check(unsafe { sys::cudnn_fe_tensor_set_dim(self.raw, dim.as_ptr(), dim.len()) })
    }

    pub fn set_stride(&self, stride: &[i64]) -> Result<()> {
        check(unsafe { sys::cudnn_fe_tensor_set_stride(self.raw, stride.as_ptr(), stride.len()) })
    }

    /// Reads the (possibly inferred) dimensions.
    pub fn dim(&self) -> Result<Vec<i64>> {
        read_i64_vec(|out, len| unsafe { sys::cudnn_fe_tensor_get_dim(self.as_ptr(), out, len) })
    }

    /// Reads the (possibly inferred) strides.
    pub fn stride(&self) -> Result<Vec<i64>> {
        read_i64_vec(|out, len| unsafe { sys::cudnn_fe_tensor_get_stride(self.as_ptr(), out, len) })
    }

    pub fn is_in_graph(&self, graph: &GraphBuilder) -> bool {
        ptr::eq(self.graph, graph)
    }
}

fn read_i64_vec(mut f: impl FnMut(*mut i64, *mut usize) -> i32) -> Result<Vec<i64>> {
    let mut len: usize = 0;
    check(f(ptr::null_mut(), &mut len))?;
    let mut v = vec![0i64; len];
    check(f(v.as_mut_ptr(), &mut len))?;
    v.truncate(len);
    Ok(v)
}

#[derive(Clone)]
pub struct TensorAttributes {
    dim: Vec<i64>,
    stride: Vec<i64>,
    data_type: DataType,
    is_virtual: bool,
    is_pass_by_value: bool,
    uid: Option<i64>,
    alignment: i64,
    reordering: TensorReordering,
    name: Option<String>,
}

impl TensorAttributes {
    pub fn new(dim: &[i64], stride: &[i64]) -> Self {
        TensorAttributes {
            dim: dim.to_vec(),
            stride: stride.to_vec(),
            data_type: DataType::NotSet,
            is_virtual: false,
            is_pass_by_value: false,
            uid: None,
            alignment: 0,
            reordering: TensorReordering::None,
            name: None,
        }
    }

    pub fn data_type(mut self, dt: DataType) -> Self {
        self.data_type = dt;
        self
    }
    pub fn set_virtual(mut self, v: bool) -> Self {
        self.is_virtual = v;
        self
    }
    pub fn pass_by_value(mut self, v: bool) -> Self {
        self.is_pass_by_value = v;
        self
    }
    pub fn uid(mut self, uid: i64) -> Self {
        self.uid = Some(uid);
        self
    }
    pub fn alignment(mut self, a: i64) -> Self {
        self.alignment = a;
        self
    }
    pub fn reordering(mut self, r: TensorReordering) -> Self {
        self.reordering = r;
        self
    }
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
}

pub struct GraphBuilder<'h> {
    pub(crate) raw: *mut sys::cudnnFeGraph,
    handle: &'h CudnnHandle,
}

impl<'h> GraphBuilder<'h> {
    /// Creates an empty graph.
    pub fn new(handle: &'h CudnnHandle) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_graph_create(&mut raw) })?;
        Ok(Self { raw, handle })
    }

    pub fn set_io_data_type(&self, dt: DataType) -> Result<&Self> {
        check(unsafe { sys::cudnn_fe_graph_set_io_data_type(self.raw, dt.to_ffi()) })?;
        Ok(self)
    }

    pub fn set_intermediate_data_type(&self, dt: DataType) -> Result<&Self> {
        check(unsafe { sys::cudnn_fe_graph_set_intermediate_data_type(self.raw, dt.to_ffi()) })?;
        Ok(self)
    }

    pub fn set_compute_data_type(&self, dt: DataType) -> Result<&Self> {
        check(unsafe { sys::cudnn_fe_graph_set_compute_data_type(self.raw, dt.to_ffi()) })?;
        Ok(self)
    }

    pub fn set_name(&self, name: &str) -> Result<&Self> {
        let c = CString::new(name).unwrap();
        check(unsafe { sys::cudnn_fe_graph_set_name(self.raw, c.as_ptr()) })?;
        Ok(self)
    }

    pub fn set_dynamic_shape_enabled(&self, enabled: bool) -> Result<&Self> {
        check(unsafe { sys::cudnn_fe_graph_set_dynamic_shape_enabled(self.raw, enabled as i32) })?;
        Ok(self)
    }

    pub fn set_sm_count(&self, sm_count: i32) -> Result<&Self> {
        check(unsafe { sys::cudnn_fe_graph_set_sm_count(self.raw, sm_count) })?;
        Ok(self)
    }

    pub fn set_sm_version(&self, sm_version: i32) -> Result<&Self> {
        check(unsafe { sys::cudnn_fe_graph_set_sm_version(self.raw, sm_version) })?;
        Ok(self)
    }

    /// Adds an input/constant tensor to the graph.
    pub fn tensor<'g>(&'g self, attrs: &TensorAttributes) -> Result<Tensor<'g, 'h>> {
        let name_c = attrs
            .name
            .as_ref()
            .map(|s| CString::new(s.as_str()).unwrap());
        let desc = sys::cudnnFeTensorDesc {
            dim: attrs.dim.as_ptr(),
            dim_len: attrs.dim.len(),
            stride: attrs.stride.as_ptr(),
            stride_len: attrs.stride.len(),
            data_type: attrs.data_type.to_ffi(),
            is_virtual: attrs.is_virtual as i32,
            is_pass_by_value: attrs.is_pass_by_value as i32,
            has_uid: attrs.uid.is_some() as i32,
            uid: attrs.uid.unwrap_or(0),
            alignment: attrs.alignment,
            reordering: attrs.reordering.to_ffi(),
            ragged_offset: ptr::null(),
            ragged_offset_multiplier: 0,
            name: name_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
        };
        let mut out = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_graph_tensor(self.raw, &desc, &mut out) })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }

    pub fn tensor_like<'g>(
        &'g self,
        like: Tensor<'g, 'h>,
        name: Option<&str>,
    ) -> Result<Tensor<'g, 'h>> {
        assert!(like.is_in_graph(self));
        let name_c = name.map(|s| CString::new(s).unwrap());
        let mut out = ptr::null_mut();
        check(unsafe {
            sys::cudnn_fe_graph_tensor_like(
                self.raw,
                like.as_ptr(),
                name_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
                &mut out,
            )
        })?;
        Ok(unsafe { Tensor::from_raw(out, self) })
    }

    pub fn validate(&self) -> Result<()> {
        check(unsafe { sys::cudnn_fe_graph_validate(self.raw) })
    }

    pub fn build_operation_graph(&self) -> Result<()> {
        check(unsafe { sys::cudnn_fe_graph_build_operation_graph(self.raw, self.handle.as_ptr()) })
    }

    pub fn create_execution_plans(&self, modes: &[HeurMode]) -> Result<()> {
        let raw: Vec<sys::cudnnFeHeurMode> = modes.iter().map(|m| m.to_ffi()).collect();
        check(unsafe {
            sys::cudnn_fe_graph_create_execution_plans(self.raw, raw.as_ptr(), raw.len())
        })
    }

    pub fn check_support(&self) -> Result<()> {
        check(unsafe { sys::cudnn_fe_graph_check_support(self.raw, self.handle.as_ptr()) })
    }

    pub fn build_plans(&self, policy: BuildPlanPolicy) -> Result<()> {
        check(unsafe {
            sys::cudnn_fe_graph_build_plans(self.raw, self.handle.as_ptr(), policy.to_ffi())
        })
    }

    pub fn deselect_workspace_greater_than(&self, workspace: i64) -> Result<()> {
        check(unsafe { sys::cudnn_fe_graph_deselect_workspace_greater_than(self.raw, workspace) })
    }

    pub fn deselect_engines(&self, names: &[&str]) -> Result<()> {
        let cstrs: Vec<CString> = names.iter().map(|s| CString::new(*s).unwrap()).collect();
        let ptrs: Vec<*const std::os::raw::c_char> = cstrs.iter().map(|c| c.as_ptr()).collect();
        check(unsafe { sys::cudnn_fe_graph_deselect_engines(self.raw, ptrs.as_ptr(), ptrs.len()) })
    }

    pub fn execution_plan_count(&self) -> Result<i64> {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_execution_plan_count(self.raw, &mut out) })?;
        Ok(out)
    }

    pub fn engine_count(&self) -> Result<i64> {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_engine_count(self.raw, &mut out) })?;
        Ok(out)
    }

    pub fn plan_name_at_index(&self, index: i64) -> Result<String> {
        let mut buf = vec![0u8; 256];
        check(unsafe {
            sys::cudnn_fe_graph_get_plan_name_at_index(
                self.raw,
                index,
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len(),
            )
        })?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    }

    pub fn build(self, modes: &[HeurMode]) -> Result<BuiltGraph> {
        let raw: Vec<sys::cudnnFeHeurMode> = modes.iter().map(|m| m.to_ffi()).collect();
        check(unsafe {
            sys::cudnn_fe_graph_build(self.raw, (*self.handle).as_ptr(), raw.as_ptr(), raw.len())
        })?;
        let raw = self.raw;
        mem::forget(self);
        Ok(BuiltGraph { raw })
    }
}

impl Drop for GraphBuilder<'_> {
    fn drop(&mut self) {
        unsafe { sys::cudnn_fe_graph_destroy(self.raw) };
    }
}

pub struct BuiltGraph {
    raw: *mut sys::cudnnFeGraph,
}

impl BuiltGraph {
    pub fn workspace_size(&self) -> Result<i64> {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_workspace_size(self.raw, &mut out) })?;
        Ok(out)
    }

    /// # Safety
    /// All provided device pointers must match cuDNN expectations.
    pub unsafe fn execute(
        &self,
        handle: &CudnnHandle,
        variant_pack: &[(i64, CUdeviceptr)],
        workspace: CUdeviceptr,
    ) -> Result<()> {
        let (uids, ptrs) = split_varpack(variant_pack);
        check(unsafe {
            sys::cudnn_fe_graph_execute(
                self.raw,
                handle.as_ptr(),
                uids.as_ptr(),
                ptrs.as_ptr(),
                uids.len(),
                workspace,
            )
        })
    }

    /// # Safety
    /// All provided device pointers must match cuDNN expectations.
    pub unsafe fn execute_plan_at_index(
        &self,
        handle: &CudnnHandle,
        variant_pack: &[(i64, CUdeviceptr)],
        workspace: CUdeviceptr,
        index: i64,
    ) -> Result<()> {
        let (uids, ptrs) = split_varpack(variant_pack);
        check(unsafe {
            sys::cudnn_fe_graph_execute_plan_at_index(
                self.raw,
                handle.as_ptr(),
                uids.as_ptr(),
                ptrs.as_ptr(),
                uids.len(),
                workspace,
                index,
            )
        })
    }

    pub fn serialize(&self) -> Result<Vec<u8>> {
        let mut data = ptr::null_mut();
        let mut len = 0usize;
        check(unsafe { sys::cudnn_fe_graph_serialize(self.raw, &mut data, &mut len) })?;
        let out = unsafe { std::slice::from_raw_parts(data, len) }.to_vec();
        unsafe { sys::cudnn_fe_buffer_free(data, len) };
        Ok(out)
    }
}

unsafe impl Send for BuiltGraph {}
unsafe impl Sync for BuiltGraph {}

impl Drop for BuiltGraph {
    fn drop(&mut self) {
        unsafe { sys::cudnn_fe_graph_destroy(self.raw) };
    }
}

fn split_varpack(vp: &[(i64, CUdeviceptr)]) -> (Vec<i64>, Vec<u64>) {
    let uids = vp.iter().map(|(u, _)| *u).collect();
    let ptrs = vp.iter().map(|(_, p)| *p).collect();
    (uids, ptrs)
}

pub(crate) fn opt_ptr(t: Option<Tensor<'_, '_>>) -> *const sys::cudnnFeTensor {
    t.map_or(ptr::null(), |t| t.as_ptr())
}

/// Keeps a CString alive and yields its pointer (or null for None).
pub(crate) fn cstr(s: Option<&str>) -> (Option<CString>, *const std::os::raw::c_char) {
    match s {
        Some(s) => {
            let c = CString::new(s).unwrap();
            let p = c.as_ptr();
            (Some(c), p)
        }
        None => (None, ptr::null()),
    }
}

mod ops;
pub use ops::*;
