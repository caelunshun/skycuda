// --- LLM-generated --- //

//! End-to-end tests against a real GPU. Tests whose hardware requirement is
//! unmet return early rather than fail.

use half::{bf16, f16};
use skycuda::{
    device::Device,
    device_memory_pool::DeviceMemoryPool,
    host_pinned_memory::HostPinnedMemory,
    host_pinned_memory_pool::{HostPinnedMemoryPool, HostPinnedMemoryPoolOptions},
    stream::Stream,
};
use skycuda_cublaslt::*;

/// The host-pinned pool guarantees 128-byte alignment, so the plans in these
/// tests promise exactly that rather than cuBLASLt's 256-byte default.
const ALIGN: u32 = 128;

struct Fixture {
    device: Device,
    stream: Stream,
    handle: CublasLtHandle,
    pool: HostPinnedMemoryPool,
}

impl Fixture {
    fn new() -> Self {
        let device = Device::from_index(0).unwrap();
        let stream = device.create_stream().unwrap();
        let handle = CublasLtHandle::new(&device).unwrap();
        let pool = device
            .create_host_pinned_memory_pool(HostPinnedMemoryPoolOptions {
                max_reserved_free_bytes: 1024,
            })
            .unwrap();
        Self {
            device,
            stream,
            handle,
            pool,
        }
    }

    fn upload<T: bytemuck::Pod>(&self, data: &[T]) -> HostPinnedMemory {
        let mut memory = self.pool.alloc_zeroed(size_of_val(data)).unwrap();
        memory
            .view_mut()
            .unwrap()
            .copy_from_slice(bytemuck::cast_slice(data));
        memory
    }

    fn zeroed<T>(&self, len: usize) -> HostPinnedMemory {
        self.pool
            .alloc_zeroed((len * size_of::<T>()).max(1))
            .unwrap()
    }

    fn workspace(&self, plan: &MatmulPlan) -> HostPinnedMemory {
        self.pool
            .alloc_zeroed(plan.max_workspace_size().max(ALIGN as usize))
            .unwrap()
    }
}

fn pref() -> MatmulPreference {
    MatmulPreference::default()
        .min_alignment_bytes(ALIGN)
        .aux_alignment_bytes(ALIGN)
}

fn read<T: bytemuck::Pod>(memory: &mut HostPinnedMemory) -> Vec<T> {
    bytemuck::cast_slice(memory.view_mut().unwrap()).to_vec()
}

// ---------------------------------------------------------------- references

/// A dense matrix held row-by-row for the reference implementation, regardless
/// of how it is laid out on the device.
#[derive(Clone)]
struct Mat {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

impl Mat {
    fn generated(rows: usize, cols: usize, seed: usize) -> Self {
        let data = (0..rows * cols)
            .map(|i| (((i * 37 + seed * 11) % 13) as f32 - 6.0) * 0.125)
            .collect();
        Self { rows, cols, data }
    }

    fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }

    /// Column-major storage with the given leading dimension.
    fn to_col_major(&self, ld: usize) -> Vec<f32> {
        let mut out = vec![0.0; ld * self.cols];
        for c in 0..self.cols {
            for r in 0..self.rows {
                out[c * ld + r] = self.at(r, c);
            }
        }
        out
    }

    /// Row-major storage with the given leading dimension.
    fn to_row_major(&self, ld: usize) -> Vec<f32> {
        let mut out = vec![0.0; ld * self.rows];
        for r in 0..self.rows {
            for c in 0..self.cols {
                out[r * ld + c] = self.at(r, c);
            }
        }
        out
    }

    fn transposed(&self) -> Mat {
        let mut out = Mat {
            rows: self.cols,
            cols: self.rows,
            data: vec![0.0; self.data.len()],
        };
        for r in 0..self.rows {
            for c in 0..self.cols {
                out.data[c * out.cols + r] = self.at(r, c);
            }
        }
        out
    }

    fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows);
        let mut out = Mat {
            rows: self.rows,
            cols: other.cols,
            data: vec![0.0; self.rows * other.cols],
        };
        for r in 0..self.rows {
            for c in 0..other.cols {
                let mut acc = 0.0;
                for k in 0..self.cols {
                    acc += self.at(r, k) * other.at(k, c);
                }
                out.data[r * out.cols + c] = acc;
            }
        }
        out
    }

    /// Reads back a column-major device buffer into row-major form.
    fn from_col_major(rows: usize, cols: usize, ld: usize, raw: &[f32]) -> Mat {
        let mut out = Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        };
        for c in 0..cols {
            for r in 0..rows {
                out.data[r * cols + c] = raw[c * ld + r];
            }
        }
        out
    }
}

#[track_caller]
fn assert_close(got: &Mat, want: &Mat, tolerance: f32) {
    assert_eq!((got.rows, got.cols), (want.rows, want.cols));
    let mut max_err = 0f32;
    for (g, w) in got.data.iter().zip(&want.data) {
        max_err = max_err.max((g - w).abs());
    }
    assert!(
        max_err <= tolerance,
        "result diverged from reference: max abs err = {max_err} (tolerance {tolerance})"
    );
}

#[track_caller]
fn assert_slice_close(got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len());
    let mut max_err = 0f32;
    for (g, w) in got.iter().zip(want) {
        max_err = max_err.max((g - w).abs());
    }
    assert!(
        max_err <= tolerance,
        "result diverged from reference: max abs err = {max_err} (tolerance {tolerance})"
    );
}

// ------------------------------------------------------------------- basics

#[test]
fn handle_and_versions() {
    let f = Fixture::new();
    assert_eq!(f.handle.device(), &f.device);
    assert!(
        version() >= 110000,
        "unexpected cuBLASLt version {}",
        version()
    );
    assert!(cudart_version() > 0);
}

/// `D = A * B` in f32, everything column-major and tightly packed.
#[test]
fn gemm_f32() {
    let f = Fixture::new();
    let (m, n, k) = (64usize, 48usize, 32usize);
    let (a, b) = (Mat::generated(m, k, 1), Mat::generated(k, n, 2));

    let a_mem = f.upload(&a.to_col_major(m));
    let b_mem = f.upload(&b.to_col_major(k));
    let mut d_mem = f.zeroed::<f32>(m * n);

    let desc = MatmulDesc::new(
        ComputeType::F32,
        MatrixLayout::new(DataType::F32, m as u64, k as u64, m as i64),
        MatrixLayout::new(DataType::F32, k as u64, n as u64, k as i64),
        MatrixLayout::new(DataType::F32, m as u64, n as u64, m as i64),
    );
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    assert_close(&got, &a.matmul(&b), 1e-3);
}

/// f16 operands accumulated in f32.
#[test]
fn gemm_f16_in_f32_compute() {
    let f = Fixture::new();
    let (m, n, k) = (64usize, 64usize, 64usize);
    let (a, b) = (Mat::generated(m, k, 3), Mat::generated(k, n, 4));

    let a_h: Vec<f16> = a
        .to_col_major(m)
        .iter()
        .map(|&x| f16::from_f32(x))
        .collect();
    let b_h: Vec<f16> = b
        .to_col_major(k)
        .iter()
        .map(|&x| f16::from_f32(x))
        .collect();
    let a_mem = f.upload(&a_h);
    let b_mem = f.upload(&b_h);
    let mut d_mem = f.zeroed::<f16>(m * n);

    let desc = MatmulDesc::new(
        ComputeType::F32,
        MatrixLayout::new(DataType::F16, m as u64, k as u64, m as i64),
        MatrixLayout::new(DataType::F16, k as u64, n as u64, k as i64),
        MatrixLayout::new(DataType::F16, m as u64, n as u64, m as i64),
    );
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let raw: Vec<f32> = read::<f16>(&mut d_mem).iter().map(|x| x.to_f32()).collect();
    assert_close(&Mat::from_col_major(m, n, m, &raw), &a.matmul(&b), 0.05);
}

#[test]
fn gemm_bf16() {
    let f = Fixture::new();
    let (m, n, k) = (32usize, 32usize, 32usize);
    let (a, b) = (Mat::generated(m, k, 5), Mat::generated(k, n, 6));

    let a_h: Vec<bf16> = a
        .to_col_major(m)
        .iter()
        .map(|&x| bf16::from_f32(x))
        .collect();
    let b_h: Vec<bf16> = b
        .to_col_major(k)
        .iter()
        .map(|&x| bf16::from_f32(x))
        .collect();
    let a_mem = f.upload(&a_h);
    let b_mem = f.upload(&b_h);
    let mut d_mem = f.zeroed::<bf16>(m * n);

    let desc = MatmulDesc::new(
        ComputeType::F32,
        MatrixLayout::new(DataType::BF16, m as u64, k as u64, m as i64),
        MatrixLayout::new(DataType::BF16, k as u64, n as u64, k as i64),
        MatrixLayout::new(DataType::BF16, m as u64, n as u64, m as i64),
    );
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let raw: Vec<f32> = read::<bf16>(&mut d_mem)
        .iter()
        .map(|x| x.to_f32())
        .collect();
    assert_close(&Mat::from_col_major(m, n, m, &raw), &a.matmul(&b), 0.3);
}

/// `D = A^T * B^T`, with A and B stored transposed.
#[test]
fn gemm_transposed_operands() {
    let f = Fixture::new();
    let (m, n, k) = (32usize, 48usize, 64usize);
    let (a, b) = (Mat::generated(m, k, 7), Mat::generated(k, n, 8));
    let (at, bt) = (a.transposed(), b.transposed());

    // A is stored as K x M and B as N x K, so both need transposing.
    let a_mem = f.upload(&at.to_col_major(k));
    let b_mem = f.upload(&bt.to_col_major(n));
    let mut d_mem = f.zeroed::<f32>(m * n);

    let desc = MatmulDesc::new(
        ComputeType::F32,
        MatrixLayout::new(DataType::F32, k as u64, m as u64, k as i64),
        MatrixLayout::new(DataType::F32, n as u64, k as u64, n as i64),
        MatrixLayout::new(DataType::F32, m as u64, n as u64, m as i64),
    )
    .trans_a(Operation::T)
    .trans_b(Operation::T);
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    assert_close(&got, &a.matmul(&b), 1e-3);
}

/// Every operand in `Order::Row`.
#[test]
fn gemm_row_major() {
    let f = Fixture::new();
    let (m, n, k) = (32usize, 32usize, 32usize);
    let (a, b) = (Mat::generated(m, k, 9), Mat::generated(k, n, 10));

    let a_mem = f.upload(&a.to_row_major(k));
    let b_mem = f.upload(&b.to_row_major(n));
    let mut d_mem = f.zeroed::<f32>(m * n);

    let row = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64).order(Order::Row)
    };
    let desc = MatmulDesc::new(ComputeType::F32, row(m, k, k), row(k, n, n), row(m, n, n));
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let raw = read::<f32>(&mut d_mem);
    let want = a.matmul(&b);
    assert_slice_close(&raw[..m * n], &want.to_row_major(n), 1e-3);
}

#[test]
fn gemm_strided_batch() {
    let f = Fixture::new();
    let (batch, m, n, k) = (3usize, 32usize, 16usize, 32usize);

    let a_batch: Vec<Mat> = (0..batch).map(|i| Mat::generated(m, k, 11 + i)).collect();
    let b_batch: Vec<Mat> = (0..batch).map(|i| Mat::generated(k, n, 21 + i)).collect();

    let a_raw: Vec<f32> = a_batch.iter().flat_map(|x| x.to_col_major(m)).collect();
    let b_raw: Vec<f32> = b_batch.iter().flat_map(|x| x.to_col_major(k)).collect();
    let a_mem = f.upload(&a_raw);
    let b_mem = f.upload(&b_raw);
    let mut d_mem = f.zeroed::<f32>(batch * m * n);

    let desc = MatmulDesc::new(
        ComputeType::F32,
        MatrixLayout::new(DataType::F32, m as u64, k as u64, m as i64)
            .batched(batch as i32, (m * k) as i64),
        MatrixLayout::new(DataType::F32, k as u64, n as u64, k as i64)
            .batched(batch as i32, (k * n) as i64),
        MatrixLayout::new(DataType::F32, m as u64, n as u64, m as i64)
            .batched(batch as i32, (m * n) as i64),
    );
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let raw = read::<f32>(&mut d_mem);
    for i in 0..batch {
        let got = Mat::from_col_major(m, n, m, &raw[i * m * n..]);
        assert_close(&got, &a_batch[i].matmul(&b_batch[i]), 1e-3);
    }
}

/// In-place accumulation: C aliases D, so `D = A*B + beta*D`.
#[test]
fn gemm_in_place_accumulate() {
    let f = Fixture::new();
    let (m, n, k) = (32usize, 32usize, 32usize);
    let (a, b) = (Mat::generated(m, k, 31), Mat::generated(k, n, 32));
    let d_init = Mat::generated(m, n, 33);

    let a_mem = f.upload(&a.to_col_major(m));
    let b_mem = f.upload(&b.to_col_major(k));
    let mut d_mem = f.upload(&d_init.to_col_major(m));

    let layout = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64)
    };
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout(m, k, m),
        layout(k, n, k),
        layout(m, n, m),
    );
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem).beta(Scalar::F32(2.0));
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let mut want = a.matmul(&b);
    for (w, init) in want.data.iter_mut().zip(&d_init.data) {
        *w += 2.0 * init;
    }
    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    assert_close(&got, &want, 1e-3);
}

/// `D = alpha*A*B + beta*C` with C a separate matrix.
#[test]
fn gemm_separate_c() {
    let f = Fixture::new();
    let (m, n, k) = (32usize, 32usize, 32usize);
    let (a, b, c) = (
        Mat::generated(m, k, 41),
        Mat::generated(k, n, 42),
        Mat::generated(m, n, 43),
    );

    let a_mem = f.upload(&a.to_col_major(m));
    let b_mem = f.upload(&b.to_col_major(k));
    let c_mem = f.upload(&c.to_col_major(m));
    let mut d_mem = f.zeroed::<f32>(m * n);

    let layout = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64)
    };
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout(m, k, m),
        layout(k, n, k),
        layout(m, n, m),
    )
    .c(layout(m, n, m));
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem)
            .b(&b_mem)
            .c(&c_mem)
            .d(&mut d_mem)
            .alpha(Scalar::F32(0.5))
            .beta(Scalar::F32(-1.5));
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let mut want = a.matmul(&b);
    for (w, cv) in want.data.iter_mut().zip(&c.data) {
        *w = 0.5 * *w - 1.5 * cv;
    }
    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    assert_close(&got, &want, 1e-3);
}

/// Operands living in device-local memory, staged through pinned host memory.
#[test]
fn gemm_device_memory() {
    let f = Fixture::new();
    let device_pool: DeviceMemoryPool = f
        .device
        .create_device_memory_pool(Default::default())
        .unwrap();

    let (m, n, k) = (32usize, 32usize, 32usize);
    let (a, b) = (Mat::generated(m, k, 51), Mat::generated(k, n, 52));

    let a_host = f.upload(&a.to_col_major(m));
    let b_host = f.upload(&b.to_col_major(k));
    let bytes = m * n * size_of::<f32>();

    let mut a_dev = device_pool.alloc_zeroed_on(m * k * 4, &f.stream).unwrap();
    let mut b_dev = device_pool.alloc_zeroed_on(k * n * 4, &f.stream).unwrap();
    let mut d_dev = device_pool.alloc_zeroed_on(bytes, &f.stream).unwrap();
    f.stream.copy(&a_host, &mut a_dev).unwrap();
    f.stream.copy(&b_host, &mut b_dev).unwrap();

    let layout = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64)
    };
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout(m, k, m),
        layout(k, n, k),
        layout(m, n, m),
    );
    // Device pool allocations are 256-byte aligned, so the default preference
    // applies here.
    let plan = f
        .handle
        .build_plan(&desc, &MatmulPreference::default())
        .unwrap();
    let mut workspace = device_pool
        .alloc_zeroed_on(plan.max_workspace_size().max(256), &f.stream)
        .unwrap();

    {
        let mut ops = plan.operands();
        ops.a(&a_dev).b(&b_dev).d(&mut d_dev);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let mut d_host = f.zeroed::<f32>(m * n);
    f.stream.copy(&d_dev, &mut d_host).unwrap();
    f.stream.block_on().unwrap();

    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_host));
    assert_close(&got, &a.matmul(&b), 1e-3);
}

// ----------------------------------------------------------------- epilogues

/// Shapes and data shared by the epilogue tests.
struct EpilogueCase {
    m: usize,
    n: usize,
    k: usize,
    a: Mat,
    b: Mat,
    bias: Vec<f32>,
}

impl EpilogueCase {
    fn new() -> Self {
        let (m, n, k) = (64usize, 32usize, 64usize);
        Self {
            m,
            n,
            k,
            a: Mat::generated(m, k, 61),
            b: Mat::generated(k, n, 62),
            bias: (0..m).map(|i| (i % 5) as f32 * 0.25 - 0.5).collect(),
        }
    }

    /// `A*B` with the bias broadcast down each column.
    fn biased(&self) -> Mat {
        let mut out = self.a.matmul(&self.b);
        for r in 0..out.rows {
            for c in 0..out.cols {
                out.data[r * out.cols + c] += self.bias[r];
            }
        }
        out
    }
}

/// `erf` via Abramowitz & Stegun 7.1.26; accurate to ~1.5e-7, far below the
/// tolerances used here.
fn erf(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let y = 1.0
        - ((((1.061_405_4 * t - 1.453_152_) * t + 1.421_413_7) * t - 0.284_496_74) * t
            + 0.254_829_6)
            * t
            * (-x * x).exp();
    sign * y
}

fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

#[test]
fn epilogue_bias() {
    let f = Fixture::new();
    let case = EpilogueCase::new();
    let (m, n, k) = (case.m, case.n, case.k);

    let a_mem = f.upload(&case.a.to_col_major(m));
    let b_mem = f.upload(&case.b.to_col_major(k));
    let bias_mem = f.upload(&case.bias);
    let mut d_mem = f.zeroed::<f32>(m * n);

    let layout = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64)
    };
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout(m, k, m),
        layout(k, n, k),
        layout(m, n, m),
    )
    .epilogue(Epilogue::Bias);
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).bias(&bias_mem).d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    assert_close(&got, &case.biased(), 1e-3);
}

/// `D = gelu(A*B + bias)`, with the pre-activation values written to the
/// auxiliary buffer.
#[test]
fn epilogue_gelu_aux_bias() {
    let f = Fixture::new();
    let case = EpilogueCase::new();
    let (m, n, k) = (case.m, case.n, case.k);

    let a_mem = f.upload(&case.a.to_col_major(m));
    let b_mem = f.upload(&case.b.to_col_major(k));
    let bias_mem = f.upload(&case.bias);
    let mut d_mem = f.zeroed::<f32>(m * n);
    let mut aux_mem = f.zeroed::<f32>(m * n);

    let layout = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64)
    };
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout(m, k, m),
        layout(k, n, k),
        layout(m, n, m),
    )
    .epilogue(Epilogue::GeluAuxBias)
    .aux_data_type(DataType::F32)
    .aux_ld(m as i64);
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem)
            .b(&b_mem)
            .bias(&bias_mem)
            .aux_out(&mut aux_mem)
            .d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let pre = case.biased();
    let got_aux = Mat::from_col_major(m, n, m, &read::<f32>(&mut aux_mem));
    assert_close(&got_aux, &pre, 1e-3);

    let mut want = pre.clone();
    for v in &mut want.data {
        *v = gelu(*v);
    }
    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    // Loose enough to accept either the exact or the tanh-approximated GELU.
    assert_close(&got, &want, 0.02);
}

/// The backward GELU epilogue, checking that the bias gradient is the row-wise
/// sum of the output it produced.
#[test]
fn epilogue_dgelu_bgrad() {
    let f = Fixture::new();
    let case = EpilogueCase::new();
    let (m, n, k) = (case.m, case.n, case.k);

    let a_mem = f.upload(&case.a.to_col_major(m));
    let b_mem = f.upload(&case.b.to_col_major(k));
    // The forward pass's GELU input.
    let pre = Mat::generated(m, n, 63);
    let aux_mem = f.upload(&pre.to_col_major(m));
    let mut d_mem = f.zeroed::<f32>(m * n);
    let mut bgrad_mem = f.zeroed::<f32>(m);

    let layout = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64)
    };
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout(m, k, m),
        layout(k, n, k),
        layout(m, n, m),
    )
    .epilogue(Epilogue::DGeluBGrad)
    .aux_data_type(DataType::F32)
    .aux_ld(m as i64);
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem)
            .b(&b_mem)
            .aux_in(&aux_mem)
            .bias_grad(&mut bgrad_mem)
            .d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    let got_bgrad = read::<f32>(&mut bgrad_mem);
    let want_bgrad: Vec<f32> = (0..m)
        .map(|r| (0..n).map(|c| got.at(r, c)).sum::<f32>())
        .collect();
    assert_slice_close(&got_bgrad[..m], &want_bgrad, 1e-2);
}

// ----------------------------------------------------------------------- fp8

/// Encodes a value exactly representable in `e4m3`.
fn e4m3(x: f32) -> u8 {
    if x == 0.0 {
        return 0;
    }
    let sign = if x < 0.0 { 0x80u8 } else { 0 };
    let x = x.abs();
    let exp = x.log2().floor() as i32;
    let mantissa = (x / (exp as f32).exp2() - 1.0) * 8.0;
    let mantissa_bits = mantissa.round() as u8;
    assert!(
        (mantissa - mantissa_bits as f32).abs() < 1e-5 && (-6..=8).contains(&exp),
        "{x} is not exactly representable in e4m3"
    );
    sign | (((exp + 7) as u8) << 3) | mantissa_bits
}

/// Values drawn from this set survive the round trip through `e4m3` exactly.
fn e4m3_value(i: usize) -> f32 {
    const VALUES: [f32; 8] = [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0];
    VALUES[i % VALUES.len()]
}

/// FP8 `e4m3` matmul in the required "TN" form, with scalar A/B scales.
#[test]
fn fp8_e4m3_tn_scalar_scales() {
    let f = Fixture::new();
    if (
        f.device.compute_capability_major(),
        f.device.compute_capability_minor(),
    ) < (8, 9)
    {
        eprintln!("skipping: FP8 matmul needs compute capability 8.9 or newer");
        return;
    }

    let (m, n, k) = (64usize, 64usize, 64usize);
    // A is stored K x M (so op(A) = A^T is M x K) and B is stored K x N.
    let a_t = Mat {
        rows: k,
        cols: m,
        data: (0..k * m).map(e4m3_value).collect(),
    };
    let b = Mat {
        rows: k,
        cols: n,
        data: (0..k * n).map(|i| e4m3_value(i * 3 + 1)).collect(),
    };

    let a_bytes: Vec<u8> = a_t.to_col_major(k).iter().map(|&x| e4m3(x)).collect();
    let b_bytes: Vec<u8> = b.to_col_major(k).iter().map(|&x| e4m3(x)).collect();
    let a_mem = f.upload(&a_bytes);
    let b_mem = f.upload(&b_bytes);
    let a_scale_mem = f.upload(&[0.5f32]);
    let b_scale_mem = f.upload(&[2.0f32]);
    let mut d_mem = f.zeroed::<f32>(m * n);

    let desc = MatmulDesc::new(
        ComputeType::F32,
        MatrixLayout::new(DataType::F8E4M3, k as u64, m as u64, k as i64),
        MatrixLayout::new(DataType::F8E4M3, k as u64, n as u64, k as i64),
        MatrixLayout::new(DataType::F32, m as u64, n as u64, m as i64),
    )
    .trans_a(Operation::T)
    .trans_b(Operation::N)
    .fast_accum(false)
    .scales(Scales {
        a: Some(ScaleMode::Scalar32F),
        b: Some(ScaleMode::Scalar32F),
        ..Scales::default()
    });

    let plan = match f.handle.build_plan(&desc, &pref()) {
        Ok(plan) => plan,
        Err(e) if e.status() == Some(Status::NotSupported) => {
            eprintln!("skipping: this device has no FP8 algorithm for the problem ({e})");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem)
            .b(&b_mem)
            .a_scale(&a_scale_mem)
            .b_scale(&b_scale_mem)
            .d(&mut d_mem);
        unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    // The A and B scales multiply into the accumulation: 0.5 * 2.0 == 1.0.
    let want = a_t.transposed().matmul(&b);
    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    assert_close(&got, &want, 1e-3);
}

/// MXFP8 block scaling, which needs Blackwell.
#[test]
fn fp8_mxfp8_vec32_ue8m0() {
    let f = Fixture::new();
    if f.device.compute_capability_major() < 10 {
        eprintln!("skipping: MXFP8 block scaling needs compute capability 10.0 or newer");
        return;
    }

    let (m, n, k) = (128usize, 128usize, 128usize);
    let a_bytes: Vec<u8> = (0..k * m).map(|i| e4m3(e4m3_value(i))).collect();
    let b_bytes: Vec<u8> = (0..k * n).map(|i| e4m3(e4m3_value(i * 3 + 1))).collect();
    let a_mem = f.upload(&a_bytes);
    let b_mem = f.upload(&b_bytes);
    // One UE8M0 exponent per 32 elements along K, with 0x7f meaning 2^0.
    let a_scale_mem = f.upload(&vec![0x7fu8; m * k / 32]);
    let b_scale_mem = f.upload(&vec![0x7fu8; n * k / 32]);
    let mut d_mem = f.zeroed::<f32>(m * n);

    let desc = MatmulDesc::new(
        ComputeType::F32,
        MatrixLayout::new(DataType::F8E4M3, k as u64, m as u64, k as i64),
        MatrixLayout::new(DataType::F8E4M3, k as u64, n as u64, k as i64),
        MatrixLayout::new(DataType::F32, m as u64, n as u64, m as i64),
    )
    .trans_a(Operation::T)
    .trans_b(Operation::N)
    .scales(Scales {
        a: Some(ScaleMode::Vec32UE8M0),
        b: Some(ScaleMode::Vec32UE8M0),
        ..Scales::default()
    });

    let plan = match f.handle.build_plan(&desc, &pref()) {
        Ok(plan) => plan,
        Err(e) if e.status() == Some(Status::NotSupported) => {
            eprintln!("skipping: this device has no MXFP8 algorithm for the problem ({e})");
            return;
        }
        Err(e) => panic!("{e}"),
    };
    let mut workspace = f.workspace(&plan);

    let mut ops = plan.operands();
    ops.a(&a_mem)
        .b(&b_mem)
        .a_scale(&a_scale_mem)
        .b_scale(&b_scale_mem)
        .d(&mut d_mem);
    unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
}

// --------------------------------------------------------------- algorithms

/// Every algorithm the heuristic returned runs and produces the same answer,
/// and a config read off one of them round-trips through `Algorithm::init`
/// plus the setters.
#[test]
fn algorithm_introspection() {
    let f = Fixture::new();
    let (m, n, k) = (64usize, 64usize, 64usize);
    let (a, b) = (Mat::generated(m, k, 71), Mat::generated(k, n, 72));

    let a_mem = f.upload(&a.to_col_major(m));
    let b_mem = f.upload(&b.to_col_major(k));
    let mut d_mem = f.zeroed::<f32>(m * n);

    let layout = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64)
    };
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout(m, k, m),
        layout(k, n, k),
        layout(m, n, m),
    );
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    assert!(!plan.algorithms().is_empty());
    assert!(plan.max_workspace_size() >= plan.workspace_size());

    let want = a.matmul(&b);
    let algorithms: Vec<Algorithm> = plan.algorithms().to_vec();
    let mut workspace = f.workspace(&plan);

    for algo in &algorithms {
        let checked = plan.check(algo).unwrap();
        assert_eq!(checked.workspace_size, algo.workspace_size);

        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe {
            plan.execute_with(algo, &f.stream, &ops, &mut workspace)
                .unwrap()
        };
        drop(ops);

        let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
        assert_close(&got, &want, 1e-3);
    }

    // Caps are readable and internally consistent with the chosen config.
    let config = algorithms[0].config().unwrap();
    let caps = algorithms[0].caps().unwrap();
    assert!(caps.tiles.contains(&config.tile) || config.tile == TileId::UNDEFINED);
    assert!(caps.reduction_scheme_mask.contains(config.reduction_scheme));
    assert!(config.splitk >= 1);

    // Rebuilding the same algorithm by hand yields an algorithm cuBLASLt
    // accepts for the same problem.
    let types = MatmulTypes::from_desc(&desc);
    let ids = f.handle.algo_ids(&types).unwrap();
    assert!(ids.contains(&config.id), "{:?} not in {ids:?}", config.id);

    let mut rebuilt = Algorithm::init(&f.handle, &types, config.id).unwrap();
    rebuilt.set_tile(config.tile).unwrap();
    rebuilt.set_splitk(config.splitk).unwrap();
    rebuilt
        .set_reduction_scheme(config.reduction_scheme)
        .unwrap();
    rebuilt.set_cta_swizzling(config.cta_swizzling).unwrap();
    rebuilt.set_custom_option(config.custom_option).unwrap();
    rebuilt.set_stages(config.stages).unwrap();
    rebuilt.set_inner_shape(config.inner_shape).unwrap();
    rebuilt.set_cluster_shape(config.cluster_shape).unwrap();
    assert_eq!(rebuilt.config().unwrap(), config);
    plan.check(&rebuilt).unwrap();
}

/// A plan built for one shape runs another, reusing its algorithms.
#[test]
fn reshape_to_another_size() {
    let f = Fixture::new();
    let layout = |rows: usize, cols: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, rows as i64)
    };
    let desc = |m: usize, n: usize, k: usize| {
        MatmulDesc::new(ComputeType::F32, layout(m, k), layout(k, n), layout(m, n))
    };

    let plan = f.handle.build_plan(&desc(64, 48, 32), &pref()).unwrap();

    let (m, n, k) = (96usize, 32usize, 64usize);
    let reshaped = plan.reshape(&desc(m, n, k)).unwrap();
    assert_eq!(reshaped.desc().d.rows, m as u64);

    let (a, b) = (Mat::generated(m, k, 3), Mat::generated(k, n, 4));
    let a_mem = f.upload(&a.to_col_major(m));
    let b_mem = f.upload(&b.to_col_major(k));
    let mut d_mem = f.zeroed::<f32>(m * n);
    let mut workspace = f.workspace(&reshaped);

    {
        let mut ops = reshaped.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe { reshaped.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    assert_close(&got, &a.matmul(&b), 1e-3);
}

/// Batching is part of the shape, so it may change too.
#[test]
fn reshape_to_batched() {
    let f = Fixture::new();
    let (m, n, k, batch) = (32usize, 24usize, 16usize, 3usize);
    let layout = |rows: usize, cols: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, rows as i64)
    };
    let batched =
        |rows: usize, cols: usize| layout(rows, cols).batched(batch as i32, (rows * cols) as i64);

    let desc = MatmulDesc::new(ComputeType::F32, layout(m, k), layout(k, n), layout(m, n));
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();

    let reshaped = plan
        .reshape(&MatmulDesc::new(
            ComputeType::F32,
            batched(m, k),
            batched(k, n),
            batched(m, n),
        ))
        .unwrap();

    let mats: Vec<(Mat, Mat)> = (0..batch)
        .map(|i| (Mat::generated(m, k, i + 5), Mat::generated(k, n, i + 9)))
        .collect();
    let mut a_host = Vec::new();
    let mut b_host = Vec::new();
    for (a, b) in &mats {
        a_host.extend(a.to_col_major(m));
        b_host.extend(b.to_col_major(k));
    }
    let a_mem = f.upload(&a_host);
    let b_mem = f.upload(&b_host);
    let mut d_mem = f.zeroed::<f32>(m * n * batch);
    let mut workspace = f.workspace(&reshaped);

    {
        let mut ops = reshaped.operands();
        ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
        unsafe { reshaped.execute(&f.stream, &ops, &mut workspace).unwrap() };
    }

    let d = read::<f32>(&mut d_mem);
    for (i, (a, b)) in mats.iter().enumerate() {
        let got = Mat::from_col_major(m, n, m, &d[i * m * n..(i + 1) * m * n]);
        assert_close(&got, &a.matmul(b), 1e-3);
    }
}

// ------------------------------------------------------------- misuse panics

/// Builds a plain 32x32x32 f32 plan for the misuse tests.
fn simple_plan(f: &Fixture) -> MatmulPlan {
    let layout = MatrixLayout::new(DataType::F32, 32, 32, 32);
    let desc = MatmulDesc::new(ComputeType::F32, layout, layout, layout);
    f.handle.build_plan(&desc, &pref()).unwrap()
}

#[test]
#[should_panic(expected = "needs at least")]
fn rejects_short_a_buffer() {
    let f = Fixture::new();
    let plan = simple_plan(&f);
    let a_mem = f.zeroed::<f32>(32 * 31);
    plan.operands().a(&a_mem);
}

#[test]
#[should_panic(expected = "was not bound")]
fn rejects_missing_d() {
    let f = Fixture::new();
    let plan = simple_plan(&f);
    let a_mem = f.zeroed::<f32>(32 * 32);
    let b_mem = f.zeroed::<f32>(32 * 32);
    let mut workspace = f.workspace(&plan);

    let mut ops = plan.operands();
    ops.a(&a_mem).b(&b_mem);
    unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
}

#[test]
#[should_panic(expected = "cannot change")]
fn rejects_reshape_with_different_type() {
    let f = Fixture::new();
    let plan = simple_plan(&f);
    let layout = MatrixLayout::new(DataType::F32, 64, 64, 64);
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout,
        layout,
        MatrixLayout::new(DataType::F16, 64, 64, 64),
    );
    let _ = plan.reshape(&desc);
}

#[test]
#[should_panic(expected = "does not use the C operand")]
fn rejects_c_on_in_place_plan() {
    let f = Fixture::new();
    let plan = simple_plan(&f);
    let c_mem = f.zeroed::<f32>(32 * 32);
    plan.operands().c(&c_mem);
}

#[test]
#[should_panic(expected = "already bound")]
fn rejects_double_binding() {
    let f = Fixture::new();
    let plan = simple_plan(&f);
    let a_mem = f.zeroed::<f32>(32 * 32);
    plan.operands().a(&a_mem).a(&a_mem);
}

#[test]
#[should_panic(expected = "byte aligned")]
fn rejects_misaligned_pointer() {
    let f = Fixture::new();
    let plan = simple_plan(&f);
    let a_mem = f.zeroed::<f32>(32 * 32 + 8);
    let b_mem = f.zeroed::<f32>(32 * 32);
    let mut d_mem = f.zeroed::<f32>(32 * 32);
    let mut workspace = f.workspace(&plan);

    // Eight bytes into a 128-byte-aligned allocation.
    let a_slice = a_mem.slice(8..);
    let mut ops = plan.operands();
    ops.a(&a_slice).b(&b_mem).d(&mut d_mem);
    unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
}

#[test]
#[should_panic(expected = "workspace")]
fn rejects_small_workspace() {
    let f = Fixture::new();
    let layout = MatrixLayout::new(DataType::F32, 1024, 1024, 1024);
    let desc = MatmulDesc::new(ComputeType::F32, layout, layout, layout);
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();

    // Find an algorithm that actually wants a workspace; otherwise there is
    // nothing to be too small for.
    let Some(algo) = plan.algorithms().iter().find(|a| a.workspace_size > 0) else {
        panic!("no algorithm needs scratch space; nothing to test");
    };

    let a_mem = f.zeroed::<f32>(1024 * 1024);
    let b_mem = f.zeroed::<f32>(1024 * 1024);
    let mut d_mem = f.zeroed::<f32>(1024 * 1024);
    let mut workspace = f.zeroed::<u8>(algo.workspace_size - 1);

    let mut ops = plan.operands();
    ops.a(&a_mem).b(&b_mem).d(&mut d_mem);
    unsafe {
        plan.execute_with(algo, &f.stream, &ops, &mut workspace)
            .unwrap()
    };
}

#[test]
#[should_panic(expected = "scale type")]
fn rejects_wrong_scalar_type() {
    let f = Fixture::new();
    let plan = simple_plan(&f);
    plan.operands().alpha(Scalar::F64(1.0));
}

/// The same `Operands` drives several calls; the descriptor's pointer
/// attributes are written once and then left alone.
#[test]
fn reuses_operands_across_executions() {
    let f = Fixture::new();
    let case = EpilogueCase::new();
    let (m, n, k) = (case.m, case.n, case.k);

    let a_mem = f.upload(&case.a.to_col_major(m));
    let b_mem = f.upload(&case.b.to_col_major(k));
    let bias_mem = f.upload(&case.bias);
    let mut d_mem = f.zeroed::<f32>(m * n);

    let layout = |rows: usize, cols: usize, ld: usize| {
        MatrixLayout::new(DataType::F32, rows as u64, cols as u64, ld as i64)
    };
    let desc = MatmulDesc::new(
        ComputeType::F32,
        layout(m, k, m),
        layout(k, n, k),
        layout(m, n, m),
    )
    .epilogue(Epilogue::Bias);
    let plan = f.handle.build_plan(&desc, &pref()).unwrap();
    let mut workspace = f.workspace(&plan);

    {
        let mut ops = plan.operands();
        ops.a(&a_mem).b(&b_mem).bias(&bias_mem).d(&mut d_mem);
        for _ in 0..3 {
            unsafe { plan.execute(&f.stream, &ops, &mut workspace).unwrap() };
        }
    }

    // beta defaults to 0, so every call overwrites D with the same result.
    let got = Mat::from_col_major(m, n, m, &read::<f32>(&mut d_mem));
    assert_close(&got, &case.biased(), 1e-3);
}
