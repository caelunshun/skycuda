use skycuda::{
    cudarc::driver::sys::CUdeviceptr_v2,
    device_memory::{
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
    stream::Stream,
};
use skycuda_cudnn_sys as sys;
use std::{
    cell::RefCell,
    collections::HashMap,
    ffi::CString,
    mem, ptr,
    sync::{
        Arc, Mutex, Once,
        atomic::{AtomicU64, Ordering},
    },
};

mod enums;
mod error;
mod ops;
pub use enums::*;
use error::check;
pub use error::{Error, ErrorCode, Result};
pub use ops::*;

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

#[derive(Debug, Clone)]
pub struct CudnnHandle(Arc<Mutex<HandleInner>>, Stream);

#[derive(Debug, Clone)]
struct HandleInner {
    raw: *mut sys::cudnnFeHandle,
}

impl CudnnHandle {
    pub fn new(stream: &Stream) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_handle_create(stream.cu_stream().cast(), &mut raw) })?;
        Ok(CudnnHandle(
            Arc::new(Mutex::new(HandleInner { raw })),
            stream.clone(),
        ))
    }

    pub fn with_ptr<R>(&self, callback: impl FnOnce(*mut sys::cudnnFeHandle) -> R) -> R {
        let guard = self.0.lock().unwrap();
        let res = callback(guard.raw);
        drop(guard);
        res
    }
}

impl Drop for HandleInner {
    fn drop(&mut self) {
        unsafe { sys::cudnn_fe_handle_destroy(self.raw) };
    }
}

unsafe impl Send for HandleInner {}

/// Identifier binding a graph tensor to an allocation in a [`VariantPack`]
/// (cuDNN's tensor "uid").
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(pub i64);

impl TensorId {
    pub fn get(self) -> i64 {
        self.0
    }
}

impl From<i64> for TensorId {
    fn from(id: i64) -> Self {
        TensorId(id)
    }
}

impl std::fmt::Display for TensorId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Abstract tensor in a graph.
#[derive(Clone, Copy)]
pub struct Tensor<'g, 'h> {
    raw: *mut sys::cudnnFeTensor,
    graph: &'g GraphBuilder<'h>,
}

impl<'g, 'h> Tensor<'g, 'h> {
    /// Wraps a tensor that is read by the graph (created via `tensor`/`tensor_like`).
    pub(crate) unsafe fn from_raw_input(
        raw: *mut sys::cudnnFeTensor,
        graph: &'g GraphBuilder<'h>,
    ) -> Self {
        graph.tensors.borrow_mut().push((raw, false));
        Tensor { raw, graph }
    }

    /// Wraps a tensor that is written by the graph (produced by an operation).
    pub(crate) unsafe fn from_raw_output(
        raw: *mut sys::cudnnFeTensor,
        graph: &'g GraphBuilder<'h>,
    ) -> Self {
        graph.tensors.borrow_mut().push((raw, true));
        Tensor { raw, graph }
    }

    pub(crate) fn as_ptr(&self) -> *const sys::cudnnFeTensor {
        self.raw as *const _
    }

    pub fn uid(&self) -> TensorId {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_tensor_get_uid(self.as_ptr(), &mut out) }).unwrap();
        TensorId(out)
    }

    pub fn set_uid(&self, uid: TensorId) {
        check(unsafe { sys::cudnn_fe_tensor_set_uid(self.raw, uid.0) }).unwrap()
    }

    pub fn set_output(&self, is_output: bool) {
        check(unsafe { sys::cudnn_fe_tensor_set_output(self.raw, is_output as i32) }).unwrap()
    }

    pub fn set_data_type(&self, dt: DataType) {
        check(unsafe { sys::cudnn_fe_tensor_set_data_type(self.raw, dt.to_ffi()) }).unwrap()
    }

    pub fn set_dim(&self, dim: &[usize]) {
        let dim = to_ffi_dims(dim);
        check(unsafe { sys::cudnn_fe_tensor_set_dim(self.raw, dim.as_ptr(), dim.len()) }).unwrap()
    }

    pub fn set_stride(&self, stride: &[usize]) {
        let stride = to_ffi_dims(stride);
        check(unsafe { sys::cudnn_fe_tensor_set_stride(self.raw, stride.as_ptr(), stride.len()) })
            .unwrap()
    }

    /// Reads the (possibly inferred) dimensions.
    pub fn dim(&self) -> Vec<usize> {
        from_ffi_dims(read_i64_vec(|out, len| unsafe {
            sys::cudnn_fe_tensor_get_dim(self.as_ptr(), out, len)
        }))
    }

    /// Reads the (possibly inferred) strides.
    pub fn stride(&self) -> Vec<usize> {
        from_ffi_dims(read_i64_vec(|out, len| unsafe {
            sys::cudnn_fe_tensor_get_stride(self.as_ptr(), out, len)
        }))
    }

    pub fn is_in_graph(&self, graph: &GraphBuilder) -> bool {
        ptr::eq(self.graph, graph)
    }

    pub fn data_type(&self) -> DataType {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_tensor_get_data_type(self.as_ptr(), &mut out) }).unwrap();
        DataType::from_ffi(out).expect("cuDNN returned an unknown data type")
    }

    pub fn is_virtual(&self) -> bool {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_tensor_get_is_virtual(self.as_ptr(), &mut out) }).unwrap();
        out != 0
    }

    pub fn is_pass_by_value(&self) -> bool {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_tensor_get_is_pass_by_value(self.as_ptr(), &mut out) })
            .unwrap();
        out != 0
    }

    pub fn reordering(&self) -> TensorReordering {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_tensor_get_reordering(self.as_ptr(), &mut out) }).unwrap();
        TensorReordering::from_ffi(out).expect("cuDNN returned an unknown reordering")
    }
}

impl DataType {
    pub fn size_bits(self) -> Option<usize> {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_data_type_size_bits(self.to_ffi(), &mut out) }).unwrap();
        usize::try_from(out).ok().filter(|&b| b > 0)
    }
}

#[derive(Debug, Clone)]
pub struct TensorRequirements {
    uid: TensorId,
    data_type: DataType,
    dim: Vec<usize>,
    stride: Vec<usize>,
    is_output: bool,
    is_pass_by_value: bool,
    reordering: TensorReordering,
}

impl TensorRequirements {
    pub fn uid(&self) -> TensorId {
        self.uid
    }
    pub fn data_type(&self) -> DataType {
        self.data_type
    }
    pub fn dim(&self) -> &[usize] {
        &self.dim
    }
    pub fn stride(&self) -> &[usize] {
        &self.stride
    }
    /// Whether the graph writes to this tensor.
    pub fn is_output(&self) -> bool {
        self.is_output
    }
    pub fn is_pass_by_value(&self) -> bool {
        self.is_pass_by_value
    }

    /// Minimum allocation length in bytes, or `None` if it cannot be determined
    /// (unknown element size, vectorized data type, or reordered layout).
    pub fn min_len(&self) -> Option<usize> {
        if self.reordering != TensorReordering::None
            || matches!(
                self.data_type,
                DataType::Int8x4 | DataType::Uint8x4 | DataType::Int8x32
            )
            || self.dim.len() != self.stride.len()
        {
            return None;
        }
        let bits = self.data_type.size_bits()?;
        if self.dim.contains(&0) {
            return Some(0);
        }
        let span = self
            .dim
            .iter()
            .zip(&self.stride)
            .try_fold(1usize, |acc, (&d, &s)| {
                acc.checked_add((d - 1).checked_mul(s)?)
            })
            .expect("tensor extent overflows usize");
        Some(
            span.checked_mul(bits)
                .expect("tensor extent overflows usize")
                .div_ceil(8),
        )
    }
}

fn read_i64_vec(mut f: impl FnMut(*mut i64, *mut usize) -> i32) -> Vec<i64> {
    let mut len: usize = 0;
    check(f(ptr::null_mut(), &mut len)).unwrap();
    let mut v = vec![0i64; len];
    check(f(v.as_mut_ptr(), &mut len)).unwrap();
    v.truncate(len);
    v
}

/// Converts dimensions/strides to the signed representation used by cuDNN.
pub(crate) fn to_ffi_dims(v: &[usize]) -> Vec<i64> {
    v.iter()
        .map(|&x| i64::try_from(x).expect("dimension does not fit in i64"))
        .collect()
}

fn from_ffi_dims(v: Vec<i64>) -> Vec<usize> {
    v.into_iter()
        .map(|x| usize::try_from(x).expect("cuDNN returned a negative dimension"))
        .collect()
}

#[derive(Clone)]
pub struct TensorAttributes {
    dim: Vec<i64>,
    stride: Vec<i64>,
    data_type: DataType,
    is_virtual: bool,
    is_pass_by_value: bool,
    uid: Option<TensorId>,
    alignment: usize,
    reordering: TensorReordering,
    name: Option<String>,
}

impl TensorAttributes {
    pub fn new(dim: &[usize], stride: &[usize]) -> Self {
        TensorAttributes {
            dim: to_ffi_dims(dim),
            stride: to_ffi_dims(stride),
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
    pub fn uid(mut self, uid: TensorId) -> Self {
        self.uid = Some(uid);
        self
    }
    pub fn alignment(mut self, a: usize) -> Self {
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
    /// Every tensor handed out by this builder, with whether the graph writes it.
    tensors: RefCell<Vec<(*mut sys::cudnnFeTensor, bool)>>,
}

impl<'h> GraphBuilder<'h> {
    /// Creates an empty graph.
    pub fn new(handle: &'h CudnnHandle) -> Result<Self> {
        let mut raw = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_graph_create(&mut raw) })?;
        Ok(Self {
            raw,
            handle,
            tensors: RefCell::new(Vec::new()),
        })
    }

    pub fn set_io_data_type(&self, dt: DataType) -> &Self {
        check(unsafe { sys::cudnn_fe_graph_set_io_data_type(self.raw, dt.to_ffi()) }).unwrap();
        self
    }

    pub fn set_intermediate_data_type(&self, dt: DataType) -> &Self {
        check(unsafe { sys::cudnn_fe_graph_set_intermediate_data_type(self.raw, dt.to_ffi()) })
            .unwrap();
        self
    }

    pub fn set_compute_data_type(&self, dt: DataType) -> &Self {
        check(unsafe { sys::cudnn_fe_graph_set_compute_data_type(self.raw, dt.to_ffi()) }).unwrap();
        self
    }

    pub fn set_name(&self, name: &str) -> &Self {
        let c = CString::new(name).unwrap();
        check(unsafe { sys::cudnn_fe_graph_set_name(self.raw, c.as_ptr()) }).unwrap();
        self
    }

    pub fn set_dynamic_shape_enabled(&self, enabled: bool) -> &Self {
        check(unsafe { sys::cudnn_fe_graph_set_dynamic_shape_enabled(self.raw, enabled as i32) })
            .unwrap();
        self
    }

    pub fn set_sm_count(&self, sm_count: i32) -> &Self {
        check(unsafe { sys::cudnn_fe_graph_set_sm_count(self.raw, sm_count) }).unwrap();
        self
    }

    pub fn set_sm_version(&self, sm_version: i32) -> &Self {
        check(unsafe { sys::cudnn_fe_graph_set_sm_version(self.raw, sm_version) }).unwrap();
        self
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
            uid: attrs.uid.map_or(0, TensorId::get),
            alignment: i64::try_from(attrs.alignment).expect("alignment does not fit in i64"),
            reordering: attrs.reordering.to_ffi(),
            ragged_offset: ptr::null(),
            ragged_offset_multiplier: 0,
            name: name_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
        };
        let mut out = ptr::null_mut();
        check(unsafe { sys::cudnn_fe_graph_tensor(self.raw, &desc, &mut out) })?;
        Ok(unsafe { Tensor::from_raw_input(out, self) })
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
        Ok(unsafe { Tensor::from_raw_input(out, self) })
    }

    pub fn validate(&self) -> Result<()> {
        check(unsafe { sys::cudnn_fe_graph_validate(self.raw) })
    }

    pub fn build_operation_graph(&self) -> Result<()> {
        self.handle.with_ptr(|handle| {
            check(unsafe { sys::cudnn_fe_graph_build_operation_graph(self.raw, handle) })
        })
    }

    pub fn create_execution_plans(&self, modes: &[HeurMode]) -> Result<()> {
        let raw: Vec<sys::cudnnFeHeurMode> = modes.iter().map(|m| m.to_ffi()).collect();
        check(unsafe {
            sys::cudnn_fe_graph_create_execution_plans(self.raw, raw.as_ptr(), raw.len())
        })
    }

    pub fn check_support(&self) -> Result<()> {
        self.handle.with_ptr(|handle| {
            check(unsafe { sys::cudnn_fe_graph_check_support(self.raw, handle) })
        })
    }

    pub fn build_plans(&self, policy: BuildPlanPolicy) -> Result<()> {
        self.handle.with_ptr(|handle| {
            check(unsafe { sys::cudnn_fe_graph_build_plans(self.raw, handle, policy.to_ffi()) })
        })
    }

    pub fn deselect_workspace_greater_than(&self, workspace: usize) -> &Self {
        let workspace = i64::try_from(workspace).expect("workspace size does not fit in i64");
        check(unsafe { sys::cudnn_fe_graph_deselect_workspace_greater_than(self.raw, workspace) })
            .unwrap();
        self
    }

    pub fn deselect_engines(&self, names: &[&str]) -> &Self {
        let cstrs: Vec<CString> = names.iter().map(|s| CString::new(*s).unwrap()).collect();
        let ptrs: Vec<*const std::os::raw::c_char> = cstrs.iter().map(|c| c.as_ptr()).collect();
        check(unsafe { sys::cudnn_fe_graph_deselect_engines(self.raw, ptrs.as_ptr(), ptrs.len()) })
            .unwrap();
        self
    }

    pub fn execution_plan_count(&self) -> usize {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_execution_plan_count(self.raw, &mut out) }).unwrap();
        usize::try_from(out).unwrap()
    }

    pub fn engine_count(&self) -> usize {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_engine_count(self.raw, &mut out) }).unwrap();
        usize::try_from(out).unwrap()
    }

    pub fn plan_name_at_index(&self, index: usize) -> Result<String> {
        let mut buf = vec![0u8; 256];
        check(unsafe {
            sys::cudnn_fe_graph_get_plan_name_at_index(
                self.raw,
                i64::try_from(index).unwrap(),
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len(),
            )
        })?;
        let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
        Ok(String::from_utf8_lossy(&buf[..end]).into_owned())
    }

    pub fn build(self, modes: &[HeurMode]) -> Result<BuiltGraph> {
        let raw: Vec<sys::cudnnFeHeurMode> = modes.iter().map(|m| m.to_ffi()).collect();
        self.handle.with_ptr(|handle| {
            check(unsafe { sys::cudnn_fe_graph_build(self.raw, handle, raw.as_ptr(), raw.len()) })
        })?;

        let tensors = self.collect_tensor_requirements();
        let raw = self.raw;
        let builder_handle = self.handle.clone();
        mem::forget(self);

        static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(0);
        Ok(BuiltGraph {
            raw,
            builder_handle,
            id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            tensors,
        })
    }

    /// Reads back the (now inferred) properties of every non-virtual tensor.
    /// Must run after the cuDNN build so shape inference and uid assignment are done.
    fn collect_tensor_requirements(&self) -> HashMap<TensorId, TensorRequirements> {
        let mut map: HashMap<TensorId, TensorRequirements> = HashMap::new();
        for &(raw, is_output) in self.tensors.borrow().iter() {
            let t = Tensor { raw, graph: self };
            if t.is_virtual() {
                continue;
            }
            let req = TensorRequirements {
                uid: t.uid(),
                data_type: t.data_type(),
                dim: t.dim(),
                stride: t.stride(),
                is_output,
                is_pass_by_value: t.is_pass_by_value(),
                reordering: t.reordering(),
            };
            match map.get_mut(&req.uid) {
                Some(existing) => existing.is_output |= is_output,
                None => {
                    map.insert(req.uid, req);
                }
            }
        }
        map
    }
}

impl Drop for GraphBuilder<'_> {
    fn drop(&mut self) {
        unsafe { sys::cudnn_fe_graph_destroy(self.raw) };
    }
}

pub struct BuiltGraph {
    id: u64,
    #[expect(unused)]
    builder_handle: CudnnHandle,
    raw: *mut sys::cudnnFeGraph,
    tensors: HashMap<TensorId, TensorRequirements>,
}

impl BuiltGraph {
    /// Requirements for the allocation bound to `uid`, if it is a non-virtual tensor of this graph.
    pub fn tensor_requirements(&self, uid: TensorId) -> Option<&TensorRequirements> {
        self.tensors.get(&uid)
    }

    pub fn all_tensor_requirements(&self) -> impl Iterator<Item = &TensorRequirements> {
        self.tensors.values()
    }

    pub fn workspace_size(&self) -> usize {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_workspace_size(self.raw, &mut out) }).unwrap();
        usize::try_from(out).expect("cuDNN returned a negative workspace size")
    }

    pub fn new_variant_pack<'a>(&'a self, handle: &'a CudnnHandle) -> VariantPack<'a> {
        VariantPack::new(self, handle)
    }

    /// # Safety
    /// All provided device pointers must match cuDNN expectations.
    pub unsafe fn execute(
        &self,
        variant_pack: &VariantPack,
        workspace: &mut impl DeviceVisibleMemoryMut,
    ) -> Result<()> {
        assert_eq!(variant_pack.graph.id, self.id);
        let mut workspace_guard = workspace.access_mut_on(&variant_pack.handle.1)?;
        variant_pack.handle.with_ptr(|handle| {
            check(unsafe {
                sys::cudnn_fe_graph_execute(
                    self.raw,
                    handle,
                    // `TensorId` is `repr(transparent)` over `i64`.
                    variant_pack.entry_ids.as_ptr().cast::<i64>(),
                    variant_pack.entry_ptrs.as_ptr(),
                    variant_pack.entry_ids.len(),
                    workspace_guard.ptr(),
                )
            })
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

/// Device allocations bound to the non-virtual tensors of a [`BuiltGraph`].
///
/// Adding an entry panics if the uid is not a non-virtual tensor of the graph,
/// was already added, is written by the graph but added via [`add`](Self::add),
/// or the allocation is shorter than the tensor's extent (when determinable).
pub struct VariantPack<'a> {
    graph: &'a BuiltGraph,
    handle: &'a CudnnHandle,
    entry_ids: Vec<TensorId>,
    entry_ptrs: Vec<CUdeviceptr_v2>,
    entry_guards: Vec<Box<dyn Anything + 'a>>,
}

trait Anything {}
impl<T> Anything for T {}

impl<'a> VariantPack<'a> {
    fn new(graph: &'a BuiltGraph, handle: &'a CudnnHandle) -> Self {
        Self {
            graph,
            handle,
            entry_ids: Vec::new(),
            entry_ptrs: Vec::new(),
            entry_guards: Vec::new(),
        }
    }

    fn validate(&self, id: TensorId, len: usize, mutable: bool) {
        let req = self
            .graph
            .tensor_requirements(id)
            .unwrap_or_else(|| panic!("tensor uid {id} is not a non-virtual tensor of this graph"));
        assert!(
            !self.entry_ids.contains(&id),
            "tensor uid {id} was already added to this variant pack"
        );
        assert!(
            mutable || !req.is_output,
            "tensor uid {id} is written by the graph; use `add_mut`"
        );
        if let Some(min_len) = req.min_len() {
            assert!(
                len >= min_len,
                "tensor uid {id} (dim {:?}, stride {:?}, {:?}) needs at least {min_len} bytes, \
                 but the allocation is {len} bytes",
                req.dim,
                req.stride,
                req.data_type,
            );
        }
    }

    pub fn add(
        &mut self,
        id: TensorId,
        memory: &'a impl DeviceVisibleMemory,
    ) -> skycuda::error::Result<&mut Self> {
        self.validate(id, memory.len(), false);
        let guard = memory.access_on(&self.handle.1)?;
        let ptr = unsafe { guard.ptr() };

        self.entry_ids.push(id);
        self.entry_ptrs.push(ptr);
        self.entry_guards.push(Box::new(guard));

        Ok(self)
    }

    pub fn add_mut(
        &mut self,
        id: TensorId,
        memory: &'a mut impl DeviceVisibleMemoryMut,
    ) -> skycuda::error::Result<&mut Self> {
        self.validate(id, memory.len(), true);
        let mut guard = memory.access_mut_on(&self.handle.1)?;
        let ptr = unsafe { guard.ptr() };

        self.entry_ids.push(id);
        self.entry_ptrs.push(ptr);
        self.entry_guards.push(Box::new(guard));

        Ok(self)
    }
}
