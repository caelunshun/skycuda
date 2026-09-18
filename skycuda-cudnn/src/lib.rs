use skycuda::{
    cudarc::driver::sys::CUdeviceptr_v2,
    device_memory::{
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
    stream::Stream,
};
use skycuda_cudnn_sys as sys;
use std::{
    cell::{Cell, RefCell},
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

    pub fn set_ragged_offset(&self, offset: Tensor<'g, 'h>, multiplier: Option<usize>) {
        assert!(offset.is_in_graph(self.graph));
        let multiplier = multiplier
            .map(|m| i64::try_from(m).expect("ragged offset multiplier does not fit in i64"))
            .unwrap_or(0);
        check(unsafe {
            sys::cudnn_fe_tensor_set_ragged_offset(
                self.graph.raw,
                self.raw,
                offset.as_ptr(),
                multiplier,
            )
        })
        .unwrap()
    }

    pub fn has_ragged_offset(&self) -> bool {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_tensor_get_has_ragged_offset(self.as_ptr(), &mut out) })
            .unwrap();
        out != 0
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
    is_ragged: bool,
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
    /// Whether the tensor uses a ragged (packed) layout with a ragged offset tensor.
    pub fn is_ragged(&self) -> bool {
        self.is_ragged
    }

    /// Minimum allocation length in bytes, or `None` if validation unsupported
    pub fn min_len(&self) -> Option<usize> {
        min_len_for(
            self.data_type,
            &self.dim,
            &self.stride,
            self.is_ragged,
            self.reordering,
        )
    }
}

fn min_len_for(
    data_type: DataType,
    dim: &[usize],
    stride: &[usize],
    is_ragged: bool,
    reordering: TensorReordering,
) -> Option<usize> {
    if is_ragged
        || reordering != TensorReordering::None
        || matches!(
            data_type,
            DataType::Int8x4 | DataType::Uint8x4 | DataType::Int8x32
        )
        || dim.len() != stride.len()
    {
        return None;
    }
    let bits = data_type.size_bits()?;
    if dim.contains(&0) {
        return Some(0);
    }
    let span = dim
        .iter()
        .zip(stride)
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
    override_shape_enabled: Cell<bool>,
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
            override_shape_enabled: Cell::new(false),
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

    /// Allows the built graph to be executed with shapes/strides other than the
    /// ones it was built with (see [`VariantPack::add_with_shape`]).
    pub fn set_override_shape_enabled(&self, enabled: bool) -> &Self {
        check(unsafe { sys::cudnn_fe_graph_set_override_shape_enabled(self.raw, enabled as i32) })
            .unwrap();
        self.override_shape_enabled.set(enabled);
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

    pub fn build_plan_at_index(&self, index: usize) -> Result<()> {
        let index = i64::try_from(index).expect("plan index does not fit in i64");
        self.handle.with_ptr(|handle| {
            check(unsafe { sys::cudnn_fe_graph_build_plan_at_index(self.raw, handle, index) })
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
        plan_name_at_index(self.raw, index)
    }

    pub fn to_json(&self) -> Result<String> {
        graph_json(self.raw)
    }

    /// Validates, builds and selects execution plans, returning an executable graph.
    pub fn build(self, modes: &[HeurMode]) -> Result<BuiltGraph> {
        self.build_with_policy(modes, BuildPlanPolicy::HeuristicsChoice)
    }

    /// Like [`Self::build`] but with an explicit plan build policy.
    pub fn build_with_policy(
        self,
        modes: &[HeurMode],
        policy: BuildPlanPolicy,
    ) -> Result<BuiltGraph> {
        if self.override_shape_enabled.get() && backend_version() < 92100 {
            return Err(Error {
                code: ErrorCode::GraphNotSupported,
                message: format!(
                    "shape overrides require cuDNN backend 9.21.0 or newer, but the loaded \
                     backend is version {}",
                    backend_version()
                ),
            });
        }
        let raw: Vec<sys::cudnnFeHeurMode> = modes.iter().map(|m| m.to_ffi()).collect();
        self.handle.with_ptr(|handle| {
            check(unsafe {
                sys::cudnn_fe_graph_build(
                    self.raw,
                    handle,
                    raw.as_ptr(),
                    raw.len(),
                    policy.to_ffi(),
                )
            })
        })?;

        let tensors = self.collect_tensor_requirements();
        let raw = self.raw;
        let builder_handle = self.handle.clone();
        let override_shape_enabled = self.override_shape_enabled.get();
        mem::forget(self);

        static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(0);
        Ok(BuiltGraph {
            raw,
            builder_handle,
            id: NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed),
            tensors,
            override_shape_enabled,
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
                is_ragged: t.has_ragged_offset(),
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
    override_shape_enabled: bool,
}

impl BuiltGraph {
    /// Requirements for the allocation bound to `uid`, if it is a non-virtual tensor of this graph.
    pub fn tensor_requirements(&self, uid: TensorId) -> Option<&TensorRequirements> {
        self.tensors.get(&uid)
    }

    pub fn all_tensor_requirements(&self) -> impl Iterator<Item = &TensorRequirements> {
        self.tensors.values()
    }

    pub fn is_override_shape_enabled(&self) -> bool {
        self.override_shape_enabled
    }

    /// Workspace size of the candidate plan for the shapes the graph was built with.
    pub fn workspace_size(&self) -> usize {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_workspace_size(self.raw, &mut out) }).unwrap();
        usize::try_from(out).expect("cuDNN returned a negative workspace size")
    }

    /// Largest workspace size over all built plans, for the shapes the graph was
    /// built with. A workspace this large suffices for [`ExecutionPlan::execute`]
    /// on any built plan.
    pub fn max_workspace_size(&self) -> usize {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_max_workspace_size(self.raw, &mut out) }).unwrap();
        usize::try_from(out).expect("cuDNN returned a negative workspace size")
    }

    /// Number of execution plans the heuristics returned (built or not).
    pub fn execution_plan_count(&self) -> usize {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_execution_plan_count(self.raw, &mut out) }).unwrap();
        usize::try_from(out).unwrap()
    }

    /// Total number of engines cuDNN has for this operation graph, whether or not
    /// the heuristics proposed them. Valid engine ids are `0..engine_count()`.
    pub fn engine_count(&self) -> usize {
        let mut out = 0;
        check(unsafe { sys::cudnn_fe_graph_get_engine_count(self.raw, &mut out) }).unwrap();
        usize::try_from(out).unwrap()
    }

    /// Knobs (and their ranges) supported by a global engine id.
    pub fn knobs_for_engine(&self, engine_id: usize) -> Result<Vec<Knob>> {
        let engine_id = i64::try_from(engine_id).expect("engine id does not fit in i64");
        let raw = read_array(|buf, cap, len| unsafe {
            sys::cudnn_fe_graph_get_knobs_for_engine(self.raw, engine_id, buf, cap, len)
        })?;
        Ok(raw
            .into_iter()
            .map(|k| Knob {
                knob_type: knob_type_from_ffi(k.type_),
                min_value: k.min_value,
                max_value: k.max_value,
                stride: k.stride,
            })
            .collect())
    }

    /// Index of the plan [`Self::execute`] runs.
    pub fn candidate_plan_index(&self) -> usize {
        let mut out = -1;
        check(unsafe { sys::cudnn_fe_graph_get_candidate_plan_index(self.raw, &mut out) }).unwrap();
        usize::try_from(out).expect("a built graph always has a candidate plan")
    }

    /// The plan [`Self::execute`] runs.
    pub fn candidate_plan(&self) -> ExecutionPlan<'_> {
        ExecutionPlan {
            graph: self,
            index: self.candidate_plan_index(),
        }
    }

    pub fn plan(&self, index: usize) -> Option<ExecutionPlan<'_>> {
        (index < self.execution_plan_count()).then_some(ExecutionPlan { graph: self, index })
    }

    /// All plans in heuristics priority order.
    pub fn plans(&self) -> impl ExactSizeIterator<Item = ExecutionPlan<'_>> + '_ {
        (0..self.execution_plan_count()).map(move |index| ExecutionPlan { graph: self, index })
    }

    /// Workspace size for the shapes bound in `variant_pack`.
    pub fn workspace_size_for(&self, variant_pack: &VariantPack) -> Result<usize> {
        assert_eq!(variant_pack.graph.id, self.id);
        if !variant_pack.has_shape_overrides() {
            return Ok(self.workspace_size());
        }
        if backend_version() < 92300 {
            return Err(Error {
                code: ErrorCode::GraphNotSupported,
                message: format!(
                    "querying the workspace size for override shapes requires cuDNN backend \
                     9.23.0 or newer, but the loaded backend is version {}",
                    backend_version()
                ),
            });
        }
        let overrides = variant_pack.ffi_overrides();
        let mut out = 0;
        variant_pack.handle.with_ptr(|handle| {
            check(unsafe {
                sys::cudnn_fe_graph_get_workspace_size_with_overrides(
                    self.raw,
                    handle,
                    overrides.as_ptr(),
                    overrides.len(),
                    &mut out,
                )
            })
        })?;
        Ok(usize::try_from(out).expect("cuDNN returned a negative workspace size"))
    }

    /// Workspace size of the plan at `index`, for the shapes the graph was built
    /// with. Fails if the plan was not built.
    pub fn workspace_size_at_index(&self, index: usize) -> Result<usize> {
        let mut out = 0;
        check(unsafe {
            sys::cudnn_fe_graph_get_plan_workspace_size_at_index(
                self.raw,
                ptr::null_mut(),
                i64::try_from(index).expect("plan index does not fit in i64"),
                ptr::null(),
                0,
                &mut out,
            )
        })?;
        Ok(usize::try_from(out).expect("cuDNN returned a negative workspace size"))
    }

    /// Workspace size of the plan at `index`, for the shapes bound in
    /// `variant_pack`. Fails if the plan was not built.
    pub fn workspace_size_at_index_for(
        &self,
        index: usize,
        variant_pack: &VariantPack,
    ) -> Result<usize> {
        assert_eq!(variant_pack.graph.id, self.id);
        if !variant_pack.has_shape_overrides() {
            return self.workspace_size_at_index(index);
        }
        if backend_version() < 92300 {
            return Err(Error {
                code: ErrorCode::GraphNotSupported,
                message: format!(
                    "querying the workspace size for override shapes requires cuDNN backend \
                     9.23.0 or newer, but the loaded backend is version {}",
                    backend_version()
                ),
            });
        }
        let overrides = variant_pack.ffi_overrides();
        let mut out = 0;
        variant_pack.handle.with_ptr(|handle| {
            check(unsafe {
                sys::cudnn_fe_graph_get_plan_workspace_size_at_index(
                    self.raw,
                    handle,
                    i64::try_from(index).expect("plan index does not fit in i64"),
                    overrides.as_ptr(),
                    overrides.len(),
                    &mut out,
                )
            })
        })?;
        Ok(usize::try_from(out).expect("cuDNN returned a negative workspace size"))
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
        let overrides = variant_pack.ffi_overrides();
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
                    overrides.as_ptr(),
                    overrides.len(),
                )
            })
        })
    }

    /// Runs a specific built plan.
    ///
    /// # Safety
    /// All provided device pointers must match cuDNN expectations.
    pub unsafe fn execute_plan_at_index(
        &self,
        variant_pack: &VariantPack,
        workspace: &mut impl DeviceVisibleMemoryMut,
        index: usize,
    ) -> Result<()> {
        assert_eq!(variant_pack.graph.id, self.id);
        let overrides = variant_pack.ffi_overrides();
        let mut workspace_guard = workspace.access_mut_on(&variant_pack.handle.1)?;
        variant_pack.handle.with_ptr(|handle| {
            check(unsafe {
                sys::cudnn_fe_graph_execute_plan_at_index(
                    self.raw,
                    handle,
                    variant_pack.entry_ids.as_ptr().cast::<i64>(),
                    variant_pack.entry_ptrs.as_ptr(),
                    variant_pack.entry_ids.len(),
                    workspace_guard.ptr(),
                    i64::try_from(index).expect("plan index does not fit in i64"),
                    overrides.as_ptr(),
                    overrides.len(),
                )
            })
        })
    }

    /// A JSON description of the graph: its context, nodes and tensors.
    pub fn to_json(&self) -> Result<String> {
        graph_json(self.raw)
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

/// A tunable engine parameter and its allowed range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Knob {
    pub knob_type: KnobType,
    pub min_value: i64,
    pub max_value: i64,
    pub stride: i64,
}

/// A knob value chosen for a particular plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KnobChoice {
    pub knob_type: KnobType,
    pub value: i64,
}

/// The engine and knob choices a plan is made of.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    /// Global engine id; see [`BuiltGraph::knobs_for_engine`].
    pub engine_id: usize,
    pub knobs: Vec<KnobChoice>,
}

/// One of the execution plans of a [`BuiltGraph`].
#[derive(Clone, Copy)]
pub struct ExecutionPlan<'a> {
    graph: &'a BuiltGraph,
    index: usize,
}

impl<'a> ExecutionPlan<'a> {
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn graph(&self) -> &'a BuiltGraph {
        self.graph
    }

    fn ffi_index(&self) -> i64 {
        i64::try_from(self.index).expect("plan index does not fit in i64")
    }

    /// Engine tag such as `eng0_k2=3_k4=1`.
    pub fn name(&self) -> String {
        plan_name_at_index(self.graph.raw, self.index).unwrap()
    }

    /// Whether cuDNN built this plan. Only built plans can report a workspace
    /// size or execute.
    pub fn is_built(&self) -> bool {
        let mut out = 0;
        check(unsafe {
            sys::cudnn_fe_graph_is_plan_built_at_index(self.graph.raw, self.ffi_index(), &mut out)
        })
        .unwrap();
        out != 0
    }

    pub fn is_candidate(&self) -> bool {
        self.index == self.graph.candidate_plan_index()
    }

    pub fn engine_config(&self) -> Result<EngineConfig> {
        let mut engine_id = -1i64;
        let raw = read_array(|buf, cap, len| unsafe {
            sys::cudnn_fe_graph_get_plan_engine_and_knobs_at_index(
                self.graph.raw,
                self.ffi_index(),
                &mut engine_id,
                buf,
                cap,
                len,
            )
        })?;
        Ok(EngineConfig {
            engine_id: usize::try_from(engine_id).expect("cuDNN returned a negative engine id"),
            knobs: raw
                .into_iter()
                .map(|k| KnobChoice {
                    knob_type: knob_type_from_ffi(k.type_),
                    value: k.value,
                })
                .collect(),
        })
    }

    pub fn behavior_notes(&self) -> Vec<BehaviorNote> {
        read_array(|buf, cap, len| unsafe {
            sys::cudnn_fe_graph_get_plan_behavior_notes_at_index(
                self.graph.raw,
                self.ffi_index(),
                buf,
                cap,
                len,
            )
        })
        .unwrap()
        .into_iter()
        .map(|n| BehaviorNote::from_ffi(n).expect("unknown behavior note from cudnn-frontend"))
        .collect()
    }

    pub fn numerical_notes(&self) -> Vec<NumericalNote> {
        read_array(|buf, cap, len| unsafe {
            sys::cudnn_fe_graph_get_plan_numerical_notes_at_index(
                self.graph.raw,
                self.ffi_index(),
                buf,
                cap,
                len,
            )
        })
        .unwrap()
        .into_iter()
        .map(|n| NumericalNote::from_ffi(n).expect("unknown numerical note from cudnn-frontend"))
        .collect()
    }

    /// Workspace size for the shapes the graph was built with. Fails if the plan
    /// was not built.
    pub fn workspace_size(&self) -> Result<usize> {
        self.graph.workspace_size_at_index(self.index)
    }

    /// Workspace size for the shapes bound in `variant_pack`. Fails if the plan
    /// was not built.
    pub fn workspace_size_for(&self, variant_pack: &VariantPack) -> Result<usize> {
        self.graph
            .workspace_size_at_index_for(self.index, variant_pack)
    }

    /// Runs this plan. Fails if the plan was not built.
    ///
    /// # Safety
    /// All provided device pointers must match cuDNN expectations.
    pub unsafe fn execute(
        &self,
        variant_pack: &VariantPack,
        workspace: &mut impl DeviceVisibleMemoryMut,
    ) -> Result<()> {
        unsafe {
            self.graph
                .execute_plan_at_index(variant_pack, workspace, self.index)
        }
    }
}

impl std::fmt::Debug for ExecutionPlan<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionPlan")
            .field("index", &self.index)
            .field("name", &self.name())
            .field("is_built", &self.is_built())
            .field("behavior_notes", &self.behavior_notes())
            .field("numerical_notes", &self.numerical_notes())
            .finish()
    }
}

fn knob_type_from_ffi(raw: sys::cudnnFeKnobType) -> KnobType {
    KnobType::from_ffi(raw).expect("unknown knob type from cudnn-frontend")
}

fn graph_json(raw: *mut sys::cudnnFeGraph) -> Result<String> {
    let mut data = ptr::null_mut();
    let mut len = 0usize;
    check(unsafe { sys::cudnn_fe_graph_print_json(raw, &mut data, &mut len) })?;
    let bytes = unsafe { std::slice::from_raw_parts(data, len) };
    let json = String::from_utf8_lossy(bytes).into_owned();
    unsafe { sys::cudnn_fe_buffer_free(data, len) };
    Ok(json)
}

fn plan_name_at_index(raw: *mut sys::cudnnFeGraph, index: usize) -> Result<String> {
    let index = i64::try_from(index).expect("plan index does not fit in i64");
    let mut buf = vec![0u8; 256];
    let mut len = 0usize;
    loop {
        check(unsafe {
            sys::cudnn_fe_graph_get_plan_name_at_index(
                raw,
                index,
                buf.as_mut_ptr().cast::<std::os::raw::c_char>(),
                buf.len(),
                &mut len,
            )
        })?;
        if len < buf.len() {
            buf.truncate(len);
            return Ok(String::from_utf8_lossy(&buf).into_owned());
        }
        buf = vec![0u8; len + 1];
    }
}

fn read_array<T: Copy>(mut query: impl FnMut(*mut T, usize, *mut usize) -> i32) -> Result<Vec<T>> {
    // `T` is always a plain C struct or enum from the shim, for which zero is valid.
    let zeroed = |n| vec![unsafe { mem::zeroed::<T>() }; n];
    let mut buf = zeroed(16);
    loop {
        let mut len = 0usize;
        check(query(buf.as_mut_ptr(), buf.len(), &mut len))?;
        if len <= buf.len() {
            buf.truncate(len);
            return Ok(buf);
        }
        buf = zeroed(len);
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
    shape_overrides: Vec<(TensorId, Vec<i64>, Vec<i64>)>,
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
            shape_overrides: Vec::new(),
        }
    }

    pub fn has_shape_overrides(&self) -> bool {
        !self.shape_overrides.is_empty()
    }

    fn ffi_overrides(&self) -> Vec<sys::cudnnFeShapeOverride> {
        self.shape_overrides
            .iter()
            .map(|(uid, dim, stride)| sys::cudnnFeShapeOverride {
                uid: uid.get(),
                dim: dim.as_ptr(),
                dim_len: dim.len(),
                stride: stride.as_ptr(),
                stride_len: stride.len(),
            })
            .collect()
    }

    fn validate(
        &self,
        id: TensorId,
        len: usize,
        mutable: bool,
        shape: Option<(&[usize], &[usize])>,
    ) {
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
        if let Some((dim, stride)) = shape {
            assert!(
                self.graph.is_override_shape_enabled(),
                "tensor uid {id} was given a runtime shape, but the graph was not built with \
                 `set_override_shape_enabled(true)`"
            );
            assert_eq!(
                dim.len(),
                stride.len(),
                "tensor uid {id}: dim and stride must have the same length"
            );
            assert_eq!(
                dim.len(),
                req.dim.len(),
                "tensor uid {id}: a shape override cannot change the rank of the tensor"
            );
        }
        let (dim, stride) = shape.unwrap_or((&req.dim, &req.stride));
        if let Some(min_len) =
            min_len_for(req.data_type, dim, stride, req.is_ragged, req.reordering)
        {
            assert!(
                len >= min_len,
                "tensor uid {id} (dim {dim:?}, stride {stride:?}, {:?}) needs at least {min_len} \
                 bytes, but the allocation is {len} bytes",
                req.data_type,
            );
        }
    }

    pub fn add(
        &mut self,
        id: TensorId,
        memory: &'a impl DeviceVisibleMemory,
    ) -> skycuda::error::Result<&mut Self> {
        self.validate(id, memory.len(), false, None);
        let guard = memory.access_on(&self.handle.1)?;
        let ptr = unsafe { guard.ptr() };
        self.record(id, ptr, Box::new(guard), None);
        Ok(self)
    }

    pub fn add_mut(
        &mut self,
        id: TensorId,
        memory: &'a mut impl DeviceVisibleMemoryMut,
    ) -> skycuda::error::Result<&mut Self> {
        self.validate(id, memory.len(), true, None);
        let mut guard = memory.access_mut_on(&self.handle.1)?;
        let ptr = unsafe { guard.ptr() };
        self.record(id, ptr, Box::new(guard), None);
        Ok(self)
    }

    pub fn add_with_shape(
        &mut self,
        id: TensorId,
        memory: &'a impl DeviceVisibleMemory,
        dim: &[usize],
        stride: &[usize],
    ) -> skycuda::error::Result<&mut Self> {
        self.validate(id, memory.len(), false, Some((dim, stride)));
        let guard = memory.access_on(&self.handle.1)?;
        let ptr = unsafe { guard.ptr() };
        self.record(id, ptr, Box::new(guard), Some((dim, stride)));
        Ok(self)
    }

    pub fn add_mut_with_shape(
        &mut self,
        id: TensorId,
        memory: &'a mut impl DeviceVisibleMemoryMut,
        dim: &[usize],
        stride: &[usize],
    ) -> skycuda::error::Result<&mut Self> {
        self.validate(id, memory.len(), true, Some((dim, stride)));
        let mut guard = memory.access_mut_on(&self.handle.1)?;
        let ptr = unsafe { guard.ptr() };
        self.record(id, ptr, Box::new(guard), Some((dim, stride)));
        Ok(self)
    }

    fn record(
        &mut self,
        id: TensorId,
        ptr: CUdeviceptr_v2,
        guard: Box<dyn Anything + 'a>,
        shape: Option<(&[usize], &[usize])>,
    ) {
        self.entry_ids.push(id);
        self.entry_ptrs.push(ptr);
        self.entry_guards.push(guard);
        if let Some((dim, stride)) = shape {
            self.shape_overrides
                .push((id, to_ffi_dims(dim), to_ffi_dims(stride)));
        }
    }
}
