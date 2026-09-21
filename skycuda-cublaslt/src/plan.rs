// --- LLM-generated --- //

//! The executable plan: heuristics run once at build, operands bound per call.

use crate::{
    CublasLtHandle,
    desc::{MatmulDesc, MatmulPreference, PointerAttr, RawMatmulDesc},
    enums::{
        AuxRole, ClusterShapeId, DataType, InnerShapeId, NumericalImplFlags, ReductionScheme,
        ReductionSchemeMask, ScaleMode, StagesId, TileId,
    },
    error::{Error, Result, Status, check},
    layout::RawMatrixLayout,
    scalar::Scalar,
    sys,
};
use bytemuck::Pod;
use skycuda::{
    cudarc::driver::sys::CUdeviceptr_v2,
    device_memory::{
        DevicePtrGuard, DevicePtrGuardMut, DeviceVisibleMemory, DeviceVisibleMemoryMut,
    },
    stream::Stream,
};
use std::{cell::RefCell, ffi::c_void, fmt, mem::MaybeUninit, ptr, sync::Mutex};

/// A built matmul: descriptors created, heuristics run, ready to execute.
///
/// Build one with
/// [`CublasLtHandle::build_plan`](crate::CublasLtHandle::build_plan) and reuse
/// it for every call with the same shapes and types. [`Self::reshape`] adapts
/// it to a new shape without running the heuristic again.
pub struct MatmulPlan {
    handle: CublasLtHandle,
    desc: MatmulDesc,
    raw_desc: Mutex<RawMatmulDesc>,
    a: RawMatrixLayout,
    b: RawMatrixLayout,
    c: RawMatrixLayout,
    d: RawMatrixLayout,
    pref: MatmulPreference,
    algorithms: Vec<Algorithm>,
}

impl MatmulPlan {
    pub(crate) fn build(
        handle: &CublasLtHandle,
        desc: &MatmulDesc,
        pref: &MatmulPreference,
    ) -> Result<Self> {
        assert_eq!(
            desc.pointer_mode,
            crate::enums::PointerMode::Host,
            "only host pointer mode is supported"
        );
        assert!(
            pref.requested_algo_count >= 1,
            "at least one heuristic result must be requested"
        );
        assert!(
            pref.aux_alignment_bytes.is_power_of_two(),
            "aux_alignment_bytes must be a power of two, got {}",
            pref.aux_alignment_bytes
        );

        handle.device().set_as_current()?;

        let a = RawMatrixLayout::new(&desc.a)?;
        let b = RawMatrixLayout::new(&desc.b)?;
        let c = RawMatrixLayout::new(desc.c_layout())?;
        let d = RawMatrixLayout::new(&desc.d)?;
        let mut raw_desc = RawMatmulDesc::new(desc)?;

        // The heuristic never dereferences the descriptor's pointer attributes,
        // but it does read their value: null means "absent", and a non-null
        // value tells it what alignment to assume. Give every pointer the
        // configuration requires a placeholder equal to the alignment the
        // caller promised, which `execute` then asserts the real pointers meet.
        let placeholder = u64::from(pref.aux_alignment_bytes);
        for attr in required_pointer_attrs(desc) {
            raw_desc.set_pointer(attr, placeholder)?;
        }

        let preference = crate::desc::RawMatmulPreference::new(
            pref,
            handle.device().compute_capability_major(),
        )?;

        let mut results =
            vec![unsafe { MaybeUninit::zeroed().assume_init() }; pref.requested_algo_count];
        let mut returned = 0i32;
        check(
            unsafe {
                sys::cublasLtMatmulAlgoGetHeuristic(
                    handle.raw(),
                    raw_desc.raw(),
                    a.raw(),
                    b.raw(),
                    c.raw(),
                    d.raw(),
                    preference.raw(),
                    pref.requested_algo_count as i32,
                    results.as_mut_ptr(),
                    &mut returned,
                )
            },
            "cublasLtMatmulAlgoGetHeuristic",
        )?;
        drop(preference);

        let algorithms: Vec<Algorithm> = results[..returned.max(0) as usize]
            .iter()
            .filter(|r| r.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            .map(|r| Algorithm {
                raw: r.algo,
                workspace_size: r.workspaceSize,
                waves_count: r.wavesCount,
            })
            .collect();

        if algorithms.is_empty() {
            return Err(no_algorithm_error(
                "cublasLtMatmulAlgoGetHeuristic",
                desc,
                pref,
                handle.device().compute_capability_major(),
            ));
        }

        Ok(Self {
            handle: handle.clone(),
            desc: desc.clone(),
            raw_desc: Mutex::new(raw_desc),
            a,
            b,
            c,
            d,
            pref: pref.clone(),
            algorithms,
        })
    }

    /// Rebuilds the plan for a different problem shape, reusing the algorithms
    /// the heuristic already chose instead of running it again.
    ///
    /// Only the shape may change: the matrix dimensions, leading dimensions,
    /// batching, and the auxiliary/bias/scalar batch strides. Everything the
    /// algorithm was selected for — the data types, orders, transposes,
    /// epilogue, scaling modes and the preference — must match, and a mismatch
    /// panics.
    ///
    /// Every algorithm is re-checked against the new shape; those cuBLASLt
    /// rejects, or whose workspace no longer fits the preference's budget, are
    /// dropped, and the rest have their workspace size and wave count
    /// refreshed. Returns [`Status::NotSupported`] if none survive, in which
    /// case the shape needs its own
    /// [`build_plan`](crate::CublasLtHandle::build_plan).
    pub fn reshape(&self, desc: &MatmulDesc) -> Result<MatmulPlan> {
        if let Some(field) = incompatible_field(&self.desc, desc) {
            panic!("reshaping a plan cannot change {field}; build a new plan instead");
        }

        self.handle.device().set_as_current()?;

        let a = RawMatrixLayout::new(&desc.a)?;
        let b = RawMatrixLayout::new(&desc.b)?;
        let c = RawMatrixLayout::new(desc.c_layout())?;
        let d = RawMatrixLayout::new(&desc.d)?;
        let mut raw_desc = RawMatmulDesc::new(desc)?;

        // As in `build`: the check reads the pointer attributes' values to
        // infer alignment, so give them the alignment the preference promises.
        let placeholder = u64::from(self.pref.aux_alignment_bytes);
        for attr in required_pointer_attrs(desc) {
            raw_desc.set_pointer(attr, placeholder)?;
        }

        let budget = self
            .pref
            .resolved_max_workspace_bytes(self.handle.device().compute_capability_major());
        let mut algorithms = Vec::with_capacity(self.algorithms.len());
        for algo in &self.algorithms {
            let mut result: sys::cublasLtMatmulHeuristicResult_t =
                unsafe { MaybeUninit::zeroed().assume_init() };
            let status = unsafe {
                sys::cublasLtMatmulAlgoCheck(
                    self.handle.raw(),
                    raw_desc.raw(),
                    a.raw(),
                    b.raw(),
                    c.raw(),
                    d.raw(),
                    &algo.raw,
                    &mut result,
                )
            };
            // An algorithm that cannot run the new shape is reported as a
            // failing status rather than an error to propagate.
            if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || result.state != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS
                || result.workspaceSize > budget
            {
                continue;
            }
            algorithms.push(Algorithm {
                raw: algo.raw,
                workspace_size: result.workspaceSize,
                waves_count: result.wavesCount,
            });
        }

        if algorithms.is_empty() {
            return Err(no_algorithm_error(
                "cublasLtMatmulAlgoCheck",
                desc,
                &self.pref,
                self.handle.device().compute_capability_major(),
            ));
        }

        Ok(Self {
            handle: self.handle.clone(),
            desc: desc.clone(),
            raw_desc: Mutex::new(raw_desc),
            a,
            b,
            c,
            d,
            pref: self.pref.clone(),
            algorithms,
        })
    }

    pub fn handle(&self) -> &CublasLtHandle {
        &self.handle
    }

    pub fn desc(&self) -> &MatmulDesc {
        &self.desc
    }

    pub fn preference(&self) -> &MatmulPreference {
        &self.pref
    }

    /// Every algorithm the heuristic accepted, best first.
    pub fn algorithms(&self) -> &[Algorithm] {
        &self.algorithms
    }

    /// The algorithm [`Self::execute`] runs.
    pub fn candidate(&self) -> &Algorithm {
        &self.algorithms[0]
    }

    /// Workspace the candidate algorithm needs, in bytes.
    pub fn workspace_size(&self) -> usize {
        self.candidate().workspace_size
    }

    /// Workspace large enough for any algorithm in [`Self::algorithms`].
    pub fn max_workspace_size(&self) -> usize {
        self.algorithms
            .iter()
            .map(|a| a.workspace_size)
            .max()
            .unwrap_or(0)
    }

    /// Asks cuBLASLt whether `algo` can run this problem, and at what cost.
    pub fn check(&self, algo: &Algorithm) -> Result<AlgorithmCheck> {
        self.handle.device().set_as_current()?;
        let raw_desc = self.raw_desc.lock().unwrap();
        let mut result: sys::cublasLtMatmulHeuristicResult_t =
            unsafe { MaybeUninit::zeroed().assume_init() };
        check(
            unsafe {
                sys::cublasLtMatmulAlgoCheck(
                    self.handle.raw(),
                    raw_desc.raw(),
                    self.a.raw(),
                    self.b.raw(),
                    self.c.raw(),
                    self.d.raw(),
                    &algo.raw,
                    &mut result,
                )
            },
            "cublasLtMatmulAlgoCheck",
        )?;
        check(result.state, "cublasLtMatmulAlgoCheck")?;
        Ok(AlgorithmCheck {
            workspace_size: result.workspaceSize,
            waves_count: result.wavesCount,
        })
    }

    /// Starts binding operands for a call.
    pub fn operands(&self) -> Operands<'_> {
        Operands::new(self)
    }

    /// Runs the candidate algorithm.
    ///
    /// # Safety
    /// The device buffers must hold data laid out as the plan's
    /// [`MatrixLayout`](crate::MatrixLayout)s describe. Lengths and alignments are checked, but
    /// cuBLASLt cannot verify contents.
    pub unsafe fn execute(
        &self,
        stream: &Stream,
        ops: &Operands<'_>,
        workspace: &mut impl DeviceVisibleMemoryMut,
    ) -> Result<()> {
        unsafe { self.execute_with(self.candidate(), stream, ops, workspace) }
    }

    /// Runs a specific algorithm, e.g. one from [`Self::algorithms`] being
    /// timed against the others.
    ///
    /// # Safety
    /// As [`Self::execute`].
    pub unsafe fn execute_with(
        &self,
        algo: &Algorithm,
        stream: &Stream,
        ops: &Operands<'_>,
        workspace: &mut impl DeviceVisibleMemoryMut,
    ) -> Result<()> {
        assert!(
            ptr::eq(ops.plan, self),
            "these operands were created for a different plan"
        );
        assert_eq!(
            stream.device(),
            self.handle.device(),
            "the stream and the cuBLASLt handle are on different devices"
        );
        assert_eq!(ops.alpha.data_type(), self.desc.scale_type);
        assert_eq!(ops.beta.data_type(), self.desc.scale_type);
        ops.assert_slots_present();

        self.handle.device().set_as_current()?;

        let mut slots = ops.slots.borrow_mut();
        let mut ptrs = [0u64; SLOT_COUNT];
        let mut guards: Vec<Box<dyn Anything + '_>> = Vec::with_capacity(SLOT_COUNT + 1);
        for (i, slot) in slots.iter_mut().enumerate() {
            if let Some(slot) = slot {
                let (ptr, guard) = slot.access(stream)?;
                ptrs[i] = ptr;
                guards.push(guard);
            }
        }
        self.assert_alignments(&ptrs);

        assert!(
            workspace.len() >= algo.workspace_size,
            "this algorithm needs a {}-byte workspace, but {} bytes were given",
            algo.workspace_size,
            workspace.len()
        );
        let workspace_len = workspace.len();
        let mut workspace_guard = workspace.access_mut_on(stream)?;
        let workspace_ptr = unsafe { workspace_guard.ptr() };

        let mut raw_desc = self.raw_desc.lock().unwrap();
        for attr in PointerAttr::ALL {
            raw_desc.set_pointer(attr, pointer_attr_value(attr, &ptrs))?;
        }

        // C aliases D when the plan describes an in-place matmul.
        let c_ptr = if self.desc.c.is_some() {
            ptrs[Slot::C as usize]
        } else {
            ptrs[Slot::D as usize]
        };

        let status = unsafe {
            sys::cublasLtMatmul(
                self.handle.raw(),
                raw_desc.raw(),
                ops.alpha.as_ptr(),
                ptrs[Slot::A as usize] as *const c_void,
                self.a.raw(),
                ptrs[Slot::B as usize] as *const c_void,
                self.b.raw(),
                ops.beta.as_ptr(),
                c_ptr as *const c_void,
                self.c.raw(),
                ptrs[Slot::D as usize] as *mut c_void,
                self.d.raw(),
                &algo.raw,
                workspace_ptr as *mut c_void,
                workspace_len,
                stream.cu_stream().cast(),
            )
        };
        drop(raw_desc);
        check(status, "cublasLtMatmul")?;

        // All pointers were captured at enqueue time, so the guards have done
        // their job and stream ordering takes it from here.
        drop(guards);
        Ok(())
    }

    fn assert_alignments(&self, ptrs: &[u64; SLOT_COUNT]) {
        let matrix_align = self.pref.min_alignment_bytes;
        for (slot, align) in [
            (Slot::A, matrix_align[0]),
            (Slot::B, matrix_align[1]),
            (Slot::C, matrix_align[2]),
            (Slot::D, matrix_align[3]),
        ] {
            assert_aligned(slot, ptrs[slot as usize], align);
        }
        for slot in Slot::ALL {
            if !matches!(slot, Slot::A | Slot::B | Slot::C | Slot::D) {
                assert_aligned(slot, ptrs[slot as usize], self.pref.aux_alignment_bytes);
            }
        }
    }

    /// Minimum length in bytes for a slot's buffer, or `None` when the layout
    /// is one this crate does not try to reason about.
    fn min_len_bytes(&self, slot: Slot) -> Option<usize> {
        let desc = &self.desc;
        let d = &desc.d;
        match slot {
            Slot::A => desc.a.min_len_bytes(),
            Slot::B => desc.b.min_len_bytes(),
            Slot::C => desc.c_layout().min_len_bytes(),
            Slot::D => d.min_len_bytes(),
            Slot::Bias | Slot::BiasGrad => {
                let bits = desc.bias_data_type.unwrap_or(d.data_type).size_bits() as u64;
                let batch =
                    (d.batch_count.max(1) as u64 - 1) * desc.bias_batch_stride.max(0) as u64;
                Some(((d.rows + batch) * bits).div_ceil(8) as usize)
            }
            Slot::AuxIn | Slot::AuxOut => {
                if desc.aux_ld <= 0 || d.rows == 0 || d.cols == 0 {
                    return None;
                }
                let bits = if desc.epilogue.aux_is_bitmask() {
                    1
                } else {
                    desc.aux_data_type.unwrap_or(d.data_type).size_bits() as u64
                };
                let batch = (d.batch_count.max(1) as u64 - 1) * desc.aux_batch_stride.max(0) as u64;
                let elements = (d.cols - 1) * desc.aux_ld as u64 + d.rows + batch;
                Some((elements * bits).div_ceil(8) as usize)
            }
            Slot::AScale => scale_min_len(desc.scales.a, d.batch_count),
            Slot::BScale => scale_min_len(desc.scales.b, d.batch_count),
            Slot::CScale => scale_min_len(desc.scales.c, d.batch_count),
            Slot::DScale => scale_min_len(desc.scales.d, d.batch_count),
            Slot::AuxScale => scale_min_len(desc.scales.aux, d.batch_count),
            Slot::DOutScale => scale_min_len(desc.scales.d_out, d.batch_count),
            // amax is a single value in the compute type.
            Slot::AmaxD | Slot::AmaxAux => {
                Some(desc.compute_type.default_scale_type().size_bits() / 8)
            }
        }
    }

    /// Whether the slot must be bound before executing.
    fn slot_required(&self, slot: Slot) -> bool {
        let desc = &self.desc;
        match slot {
            Slot::A | Slot::B | Slot::D => true,
            Slot::C => desc.c.is_some(),
            Slot::Bias => desc.epilogue.needs_bias(),
            Slot::BiasGrad => desc.epilogue.writes_bias_grad(),
            Slot::AuxIn => desc.epilogue.aux_role() == Some(AuxRole::Input),
            Slot::AuxOut => desc.epilogue.aux_role() == Some(AuxRole::Output),
            Slot::AScale => desc.scales.a.is_some(),
            Slot::BScale => desc.scales.b.is_some(),
            Slot::CScale => desc.scales.c.is_some(),
            Slot::DScale => desc.scales.d.is_some(),
            Slot::AuxScale => desc.scales.aux.is_some(),
            Slot::DOutScale => desc.scales.d_out.is_some(),
            Slot::AmaxD | Slot::AmaxAux => false,
        }
    }

    /// Whether the slot may be bound at all. The amax outputs are the only
    /// optional ones.
    fn slot_allowed(&self, slot: Slot) -> bool {
        matches!(slot, Slot::AmaxD | Slot::AmaxAux) || self.slot_required(slot)
    }
}

impl fmt::Debug for MatmulPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatmulPlan")
            .field("desc", &self.desc)
            .field("preference", &self.pref)
            .field("algorithms", &self.algorithms)
            .finish()
    }
}

/// The first field of `desc` an algorithm was selected for that `other`
/// changes, if any. Shape — dimensions, leading dimensions, batching and the
/// strides that follow from them — is deliberately not compared.
fn incompatible_field(desc: &MatmulDesc, other: &MatmulDesc) -> Option<&'static str> {
    for (name_type, name_order, lhs, rhs) in [
        ("A's data type", "A's order", &desc.a, &other.a),
        ("B's data type", "B's order", &desc.b, &other.b),
        (
            "C's data type",
            "C's order",
            desc.c_layout(),
            other.c_layout(),
        ),
        ("D's data type", "D's order", &desc.d, &other.d),
    ] {
        if lhs.data_type != rhs.data_type {
            return Some(name_type);
        }
        if lhs.order != rhs.order {
            return Some(name_order);
        }
    }

    [
        ("the compute type", desc.compute_type == other.compute_type),
        ("the scale type", desc.scale_type == other.scale_type),
        ("A's transpose", desc.trans_a == other.trans_a),
        ("B's transpose", desc.trans_b == other.trans_b),
        (
            "whether C is separate from D",
            desc.c.is_some() == other.c.is_some(),
        ),
        ("the epilogue", desc.epilogue == other.epilogue),
        (
            "the bias data type",
            desc.bias_data_type == other.bias_data_type,
        ),
        (
            "the auxiliary data type",
            desc.aux_data_type == other.aux_data_type,
        ),
        ("the pointer mode", desc.pointer_mode == other.pointer_mode),
        ("fast accumulation", desc.fast_accum == other.fast_accum),
        (
            "the SM count target",
            desc.sm_count_target == other.sm_count_target,
        ),
        ("the scaling modes", desc.scales == other.scales),
    ]
    .into_iter()
    .find(|(_, same)| !same)
    .map(|(what, _)| what)
}

fn no_algorithm_error(
    call: &'static str,
    desc: &MatmulDesc,
    pref: &MatmulPreference,
    compute_capability_major: u32,
) -> Error {
    Error::with_message(
        Status::NotSupported,
        call,
        format!(
            "cuBLASLt has no algorithm for this problem: compute {:?}, scale {:?}, \
             A {:?}{:?}, B {:?}{:?}, C {:?}, D {:?}, epilogue {:?}, scales {:?}, \
             workspace <= {} bytes",
            desc.compute_type,
            desc.scale_type,
            desc.trans_a,
            desc.a,
            desc.trans_b,
            desc.b,
            desc.c_layout(),
            desc.d,
            desc.epilogue,
            desc.scales,
            pref.resolved_max_workspace_bytes(compute_capability_major),
        ),
    )
}

fn assert_aligned(slot: Slot, ptr: u64, align: u32) {
    if ptr == 0 || align == 0 {
        return;
    }
    assert!(
        ptr.is_multiple_of(u64::from(align)),
        "the buffer bound to {slot:?} is at {ptr:#x}, which is not {align}-byte aligned as the \
         preference promises"
    );
}

/// Bytes a scaling factor buffer must hold, for the modes whose layout this
/// crate implements. The block- and vector-scaled modes have layouts that
/// depend on the algorithm's tiling, so they go unchecked.
fn scale_min_len(mode: Option<ScaleMode>, batch_count: i32) -> Option<usize> {
    match mode? {
        ScaleMode::Scalar32F => Some(4),
        ScaleMode::PerBatchScalar32F => Some(4 * batch_count.max(1) as usize),
        _ => None,
    }
}

fn required_pointer_attrs(desc: &MatmulDesc) -> Vec<PointerAttr> {
    let mut attrs = Vec::new();
    if desc.epilogue.needs_bias() || desc.epilogue.writes_bias_grad() {
        attrs.push(PointerAttr::Bias);
    }
    if desc.epilogue.aux_role().is_some() {
        attrs.push(PointerAttr::Aux);
    }
    for (mode, attr) in [
        (desc.scales.a, PointerAttr::AScale),
        (desc.scales.b, PointerAttr::BScale),
        (desc.scales.c, PointerAttr::CScale),
        (desc.scales.d, PointerAttr::DScale),
        (desc.scales.aux, PointerAttr::AuxScale),
        (desc.scales.d_out, PointerAttr::DOutScale),
    ] {
        if mode.is_some() {
            attrs.push(attr);
        }
    }
    attrs
}

/// The pointer a descriptor attribute should be set to, given the bound slots.
///
/// Bias-in and bias-gradient-out share one descriptor attribute, as do the two
/// directions of the auxiliary buffer; the epilogue decides which it is, and
/// only one slot of each pair is ever bound, so taking whichever is non-null
/// picks the right one.
fn pointer_attr_value(attr: PointerAttr, ptrs: &[u64; SLOT_COUNT]) -> u64 {
    let at = |slot: Slot| ptrs[slot as usize];
    match attr {
        PointerAttr::Bias => at(Slot::Bias) | at(Slot::BiasGrad),
        PointerAttr::Aux => at(Slot::AuxIn) | at(Slot::AuxOut),
        PointerAttr::AScale => at(Slot::AScale),
        PointerAttr::BScale => at(Slot::BScale),
        PointerAttr::CScale => at(Slot::CScale),
        PointerAttr::DScale => at(Slot::DScale),
        PointerAttr::AuxScale => at(Slot::AuxScale),
        PointerAttr::DOutScale => at(Slot::DOutScale),
        PointerAttr::AmaxD => at(Slot::AmaxD),
        PointerAttr::AmaxAux => at(Slot::AmaxAux),
    }
}

const SLOT_COUNT: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Slot {
    A = 0,
    B,
    C,
    D,
    Bias,
    BiasGrad,
    AuxIn,
    AuxOut,
    AScale,
    BScale,
    CScale,
    DScale,
    AuxScale,
    DOutScale,
    AmaxD,
    AmaxAux,
}

impl Slot {
    const ALL: [Slot; SLOT_COUNT] = [
        Slot::A,
        Slot::B,
        Slot::C,
        Slot::D,
        Slot::Bias,
        Slot::BiasGrad,
        Slot::AuxIn,
        Slot::AuxOut,
        Slot::AScale,
        Slot::BScale,
        Slot::CScale,
        Slot::DScale,
        Slot::AuxScale,
        Slot::DOutScale,
        Slot::AmaxD,
        Slot::AmaxAux,
    ];
}

trait Anything {}
impl<T> Anything for T {}

/// A bound buffer, which defers acquiring its access guard until the executing
/// stream is known.
trait MemorySlot {
    fn access<'s>(
        &'s mut self,
        stream: &'s Stream,
    ) -> skycuda::error::Result<(CUdeviceptr_v2, Box<dyn Anything + 's>)>;
}

struct ReadSlot<'a, T>(&'a T);

impl<T: DeviceVisibleMemory> MemorySlot for ReadSlot<'_, T> {
    fn access<'s>(
        &'s mut self,
        stream: &'s Stream,
    ) -> skycuda::error::Result<(CUdeviceptr_v2, Box<dyn Anything + 's>)> {
        let guard = self.0.access_on(stream)?;
        let ptr = unsafe { guard.ptr() };
        Ok((ptr, Box::new(guard)))
    }
}

struct WriteSlot<'a, T>(&'a mut T);

impl<T: DeviceVisibleMemoryMut> MemorySlot for WriteSlot<'_, T> {
    fn access<'s>(
        &'s mut self,
        stream: &'s Stream,
    ) -> skycuda::error::Result<(CUdeviceptr_v2, Box<dyn Anything + 's>)> {
        let mut guard = self.0.access_mut_on(stream)?;
        let ptr = unsafe { guard.ptr() };
        Ok((ptr, Box::new(guard)))
    }
}

/// The buffers and scalars one matmul call reads and writes.
///
/// Binding a buffer panics if the plan does not use that slot, if the slot was
/// already bound, or if the buffer is shorter than the layout needs. The
/// mutable borrows are held for as long as the `Operands` lives, so it can be
/// reused across repeated executions of the same plan.
pub struct Operands<'a> {
    plan: &'a MatmulPlan,
    alpha: Scalar,
    beta: Scalar,
    slots: RefCell<[Option<Box<dyn MemorySlot + 'a>>; SLOT_COUNT]>,
}

impl<'a> Operands<'a> {
    fn new(plan: &'a MatmulPlan) -> Self {
        Self {
            plan,
            alpha: Scalar::one(plan.desc.scale_type),
            beta: Scalar::zero(plan.desc.scale_type),
            slots: RefCell::new([const { None }; SLOT_COUNT]),
        }
    }

    /// Defaults to `1` in the descriptor's scale type.
    pub fn alpha(&mut self, alpha: Scalar) -> &mut Self {
        assert_eq!(
            alpha.data_type(),
            self.plan.desc.scale_type,
            "alpha must have the descriptor's scale type"
        );
        self.alpha = alpha;
        self
    }

    /// Defaults to `0` in the descriptor's scale type.
    pub fn beta(&mut self, beta: Scalar) -> &mut Self {
        assert_eq!(
            beta.data_type(),
            self.plan.desc.scale_type,
            "beta must have the descriptor's scale type"
        );
        self.beta = beta;
        self
    }

    fn bind_read(&mut self, slot: Slot, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.validate(slot, memory.len());
        self.slots.get_mut()[slot as usize] = Some(Box::new(ReadSlot(memory)));
        self
    }

    fn bind_write(&mut self, slot: Slot, memory: &'a mut impl DeviceVisibleMemoryMut) -> &mut Self {
        self.validate(slot, DeviceVisibleMemory::len(memory));
        self.slots.get_mut()[slot as usize] = Some(Box::new(WriteSlot(memory)));
        self
    }

    fn validate(&self, slot: Slot, len: usize) {
        assert!(
            self.plan.slot_allowed(slot),
            "this plan does not use the {slot:?} operand"
        );
        assert!(
            self.slots.borrow()[slot as usize].is_none(),
            "the {slot:?} operand was already bound"
        );
        if let Some(min_len) = self.plan.min_len_bytes(slot) {
            assert!(
                len >= min_len,
                "the {slot:?} operand needs at least {min_len} bytes, but the buffer is {len} bytes"
            );
        }
    }

    fn assert_slots_present(&self) {
        let slots = self.slots.borrow();
        for slot in Slot::ALL {
            assert!(
                !self.plan.slot_required(slot) || slots[slot as usize].is_some(),
                "this plan requires the {slot:?} operand, which was not bound"
            );
        }
    }

    pub fn a(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::A, memory)
    }

    pub fn b(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::B, memory)
    }

    /// Only for plans whose descriptor sets an explicit C layout; an in-place
    /// plan reads C from D.
    pub fn c(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::C, memory)
    }

    pub fn d(&mut self, memory: &'a mut impl DeviceVisibleMemoryMut) -> &mut Self {
        self.bind_write(Slot::D, memory)
    }

    /// The bias vector read by the `*BIAS` epilogues.
    pub fn bias(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::Bias, memory)
    }

    /// The bias gradient written by the `*BGRAD` epilogues.
    pub fn bias_grad(&mut self, memory: &'a mut impl DeviceVisibleMemoryMut) -> &mut Self {
        self.bind_write(Slot::BiasGrad, memory)
    }

    /// The auxiliary buffer read by the backward epilogues (`DRELU*`,
    /// `DGELU*`).
    pub fn aux_in(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::AuxIn, memory)
    }

    /// The auxiliary buffer written by the `*_AUX` epilogues.
    pub fn aux_out(&mut self, memory: &'a mut impl DeviceVisibleMemoryMut) -> &mut Self {
        self.bind_write(Slot::AuxOut, memory)
    }

    pub fn a_scale(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::AScale, memory)
    }

    pub fn b_scale(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::BScale, memory)
    }

    pub fn c_scale(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::CScale, memory)
    }

    pub fn d_scale(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::DScale, memory)
    }

    pub fn aux_scale(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::AuxScale, memory)
    }

    pub fn d_out_scale(&mut self, memory: &'a impl DeviceVisibleMemory) -> &mut Self {
        self.bind_read(Slot::DOutScale, memory)
    }

    /// Receives the maximum absolute value of D. Always optional.
    pub fn amax_d(&mut self, memory: &'a mut impl DeviceVisibleMemoryMut) -> &mut Self {
        self.bind_write(Slot::AmaxD, memory)
    }

    /// Receives the maximum absolute value of the auxiliary buffer. Always
    /// optional.
    pub fn amax_aux(&mut self, memory: &'a mut impl DeviceVisibleMemoryMut) -> &mut Self {
        self.bind_write(Slot::AmaxAux, memory)
    }
}

impl fmt::Debug for Operands<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let slots = self.slots.borrow();
        let bound: Vec<Slot> = Slot::ALL
            .into_iter()
            .filter(|s| slots[*s as usize].is_some())
            .collect();
        f.debug_struct("Operands")
            .field("alpha", &self.alpha)
            .field("beta", &self.beta)
            .field("bound", &bound)
            .finish()
    }
}

/// The element types a matmul algorithm is selected for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatmulTypes {
    pub compute: crate::enums::ComputeType,
    pub scale: DataType,
    pub a: DataType,
    pub b: DataType,
    pub c: DataType,
    pub d: DataType,
}

impl MatmulTypes {
    pub fn from_desc(desc: &MatmulDesc) -> Self {
        Self {
            compute: desc.compute_type,
            scale: desc.scale_type,
            a: desc.a.data_type,
            b: desc.b.data_type,
            c: desc.c_layout().data_type,
            d: desc.d.data_type,
        }
    }
}

/// One matmul algorithm: an id plus a configuration of tiling and scheduling
/// knobs.
#[derive(Clone, Copy)]
pub struct Algorithm {
    raw: sys::cublasLtMatmulAlgo_t,
    /// Workspace the heuristic reported for this algorithm. Zero for an
    /// algorithm built by [`Algorithm::init`]; use
    /// [`MatmulPlan::check`] to find out.
    pub workspace_size: usize,
    /// Device utilization the heuristic reported, in waves.
    pub waves_count: f32,
}

impl Algorithm {
    /// Builds an algorithm from scratch, for callers running their own
    /// exhaustive search over [`CublasLtHandle::algo_ids`](crate::CublasLtHandle::algo_ids).
    pub fn init(handle: &CublasLtHandle, types: &MatmulTypes, id: i32) -> Result<Self> {
        let mut raw: sys::cublasLtMatmulAlgo_t = unsafe { MaybeUninit::zeroed().assume_init() };
        check(
            unsafe {
                sys::cublasLtMatmulAlgoInit(
                    handle.raw(),
                    types.compute.to_ffi(),
                    types.scale.to_ffi(),
                    types.a.to_ffi(),
                    types.b.to_ffi(),
                    types.c.to_ffi(),
                    types.d.to_ffi(),
                    id,
                    &mut raw,
                )
            },
            "cublasLtMatmulAlgoInit",
        )?;
        Ok(Self {
            raw,
            workspace_size: 0,
            waves_count: 0.0,
        })
    }

    pub fn raw(&self) -> &sys::cublasLtMatmulAlgo_t {
        &self.raw
    }

    /// Reads back every configuration knob.
    pub fn config(&self) -> Result<AlgorithmConfig> {
        use sys::cublasLtMatmulAlgoConfigAttributes_t::*;
        let reduction_raw: u32 = self.config_attr(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)?;
        Ok(AlgorithmConfig {
            id: self.config_attr(CUBLASLT_ALGO_CONFIG_ID)?,
            tile: TileId(self.config_attr(CUBLASLT_ALGO_CONFIG_TILE_ID)?),
            splitk: self.config_attr(CUBLASLT_ALGO_CONFIG_SPLITK_NUM)?,
            reduction_scheme: ReductionScheme::from_raw(reduction_raw).ok_or_else(|| {
                Error::with_message(
                    Status::InvalidValue,
                    "cublasLtMatmulAlgoConfigGetAttribute",
                    format!("unknown reduction scheme {reduction_raw}"),
                )
            })?,
            cta_swizzling: self.config_attr(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)?,
            custom_option: self.config_attr(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)?,
            stages: StagesId(self.config_attr(CUBLASLT_ALGO_CONFIG_STAGES_ID)?),
            inner_shape: InnerShapeId(self.config_attr(CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID)?),
            cluster_shape: ClusterShapeId(self.config_attr(CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID)?),
        })
    }

    /// Reads back everything cuBLASLt will say about what this algorithm
    /// supports.
    pub fn caps(&self) -> Result<AlgorithmCaps> {
        use sys::cublasLtMatmulAlgoCapAttributes_t::*;
        Ok(AlgorithmCaps {
            splitk: self.cap_attr::<i32>(CUBLASLT_ALGO_CAP_SPLITK_SUPPORT)? != 0,
            reduction_scheme_mask: ReductionSchemeMask(
                self.cap_attr(CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK)?,
            ),
            cta_swizzling: self.cap_attr::<u32>(CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT)? != 0,
            strided_batch: self.cap_attr::<i32>(CUBLASLT_ALGO_CAP_STRIDED_BATCH_SUPPORT)? != 0,
            out_of_place: self.cap_attr::<i32>(CUBLASLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT)? != 0,
            tiles: self
                .cap_array(CUBLASLT_ALGO_CAP_TILE_IDS)?
                .into_iter()
                .map(TileId)
                .collect(),
            stages: self
                .cap_array(CUBLASLT_ALGO_CAP_STAGES_IDS)?
                .into_iter()
                .map(StagesId)
                .collect(),
            custom_option_max: self.cap_attr(CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX)?,
            pointer_mode_mask: self.cap_attr(CUBLASLT_ALGO_CAP_POINTER_MODE_MASK)?,
            epilogue_mask: self.cap_attr(CUBLASLT_ALGO_CAP_EPILOGUE_MASK)?,
            numerical_impl: NumericalImplFlags(
                self.cap_attr(CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS)?,
            ),
            min_alignment_bytes: [
                self.cap_attr(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES)?,
                self.cap_attr(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES)?,
                self.cap_attr(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES)?,
                self.cap_attr(CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES)?,
            ],
            ld_negative: self.cap_attr::<i32>(CUBLASLT_ALGO_CAP_LD_NEGATIVE)? != 0,
        })
    }

    pub fn set_tile(&mut self, tile: TileId) -> Result<()> {
        self.set_config_attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_TILE_ID,
            &tile.0,
        )
    }

    pub fn set_splitk(&mut self, splitk: i32) -> Result<()> {
        self.set_config_attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            &splitk,
        )
    }

    pub fn set_reduction_scheme(&mut self, scheme: ReductionScheme) -> Result<()> {
        self.set_config_attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
            &scheme.raw(),
        )
    }

    pub fn set_cta_swizzling(&mut self, swizzling: u32) -> Result<()> {
        self.set_config_attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
            &swizzling,
        )
    }

    pub fn set_custom_option(&mut self, option: u32) -> Result<()> {
        self.set_config_attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
            &option,
        )
    }

    pub fn set_stages(&mut self, stages: StagesId) -> Result<()> {
        self.set_config_attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_STAGES_ID,
            &stages.0,
        )
    }

    pub fn set_inner_shape(&mut self, shape: InnerShapeId) -> Result<()> {
        self.set_config_attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
            &shape.0,
        )
    }

    pub fn set_cluster_shape(&mut self, shape: ClusterShapeId) -> Result<()> {
        self.set_config_attr(
            sys::cublasLtMatmulAlgoConfigAttributes_t::CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            &shape.0,
        )
    }

    fn config_attr<T: Pod>(&self, attr: sys::cublasLtMatmulAlgoConfigAttributes_t) -> Result<T> {
        let mut value = T::zeroed();
        let mut written = 0usize;
        check(
            unsafe {
                sys::cublasLtMatmulAlgoConfigGetAttribute(
                    &self.raw,
                    attr,
                    (&raw mut value).cast(),
                    size_of::<T>(),
                    &mut written,
                )
            },
            "cublasLtMatmulAlgoConfigGetAttribute",
        )?;
        Ok(value)
    }

    fn set_config_attr<T>(
        &mut self,
        attr: sys::cublasLtMatmulAlgoConfigAttributes_t,
        value: &T,
    ) -> Result<()> {
        check(
            unsafe {
                sys::cublasLtMatmulAlgoConfigSetAttribute(
                    &mut self.raw,
                    attr,
                    (value as *const T).cast(),
                    size_of::<T>(),
                )
            },
            "cublasLtMatmulAlgoConfigSetAttribute",
        )
    }

    fn cap_attr<T: Pod>(&self, attr: sys::cublasLtMatmulAlgoCapAttributes_t) -> Result<T> {
        let mut value = T::zeroed();
        let mut written = 0usize;
        check(
            unsafe {
                sys::cublasLtMatmulAlgoCapGetAttribute(
                    &self.raw,
                    attr,
                    (&raw mut value).cast(),
                    size_of::<T>(),
                    &mut written,
                )
            },
            "cublasLtMatmulAlgoCapGetAttribute",
        )?;
        Ok(value)
    }

    /// The two-call protocol for the array-valued caps: size first, then data.
    fn cap_array(&self, attr: sys::cublasLtMatmulAlgoCapAttributes_t) -> Result<Vec<u32>> {
        let mut written = 0usize;
        check(
            unsafe {
                sys::cublasLtMatmulAlgoCapGetAttribute(
                    &self.raw,
                    attr,
                    ptr::null_mut(),
                    0,
                    &mut written,
                )
            },
            "cublasLtMatmulAlgoCapGetAttribute",
        )?;
        let mut values = vec![0u32; written / size_of::<u32>()];
        if values.is_empty() {
            return Ok(values);
        }
        check(
            unsafe {
                sys::cublasLtMatmulAlgoCapGetAttribute(
                    &self.raw,
                    attr,
                    values.as_mut_ptr().cast(),
                    written,
                    &mut written,
                )
            },
            "cublasLtMatmulAlgoCapGetAttribute",
        )?;
        Ok(values)
    }
}

impl fmt::Debug for Algorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Algorithm")
            .field("config", &self.config().ok())
            .field("workspace_size", &self.workspace_size)
            .field("waves_count", &self.waves_count)
            .finish()
    }
}

/// The knob settings of an [`Algorithm`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgorithmConfig {
    pub id: i32,
    pub tile: TileId,
    pub splitk: i32,
    pub reduction_scheme: ReductionScheme,
    pub cta_swizzling: u32,
    pub custom_option: u32,
    pub stages: StagesId,
    pub inner_shape: InnerShapeId,
    pub cluster_shape: ClusterShapeId,
}

/// What an [`Algorithm`] supports, independent of any particular problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlgorithmCaps {
    pub splitk: bool,
    pub reduction_scheme_mask: ReductionSchemeMask,
    pub cta_swizzling: bool,
    pub strided_batch: bool,
    /// Whether `D` may differ from `C`.
    pub out_of_place: bool,
    pub tiles: Vec<TileId>,
    pub stages: Vec<StagesId>,
    pub custom_option_max: i32,
    /// Bitmask over `cublasLtPointerModeMask_t`.
    pub pointer_mode_mask: u32,
    /// Bitmask over [`Epilogue`](crate::Epilogue) values.
    pub epilogue_mask: u32,
    pub numerical_impl: NumericalImplFlags,
    /// Minimum alignment for A, B, C and D, in bytes.
    pub min_alignment_bytes: [u32; 4],
    pub ld_negative: bool,
}

/// What [`MatmulPlan::check`] reports about an algorithm.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlgorithmCheck {
    pub workspace_size: usize,
    pub waves_count: f32,
}

// SAFETY: the plan's interior mutability is all behind a mutex or a `&mut`, and
// cuBLASLt handles are documented as thread-safe.
unsafe impl Send for MatmulPlan {}
unsafe impl Sync for MatmulPlan {}
unsafe impl Send for Algorithm {}
unsafe impl Sync for Algorithm {}
