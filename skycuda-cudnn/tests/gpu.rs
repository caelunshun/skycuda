use half::f16;
use skycuda::{
    device::Device,
    host_pinned_memory_pool::{HostPinnedMemoryPool, HostPinnedMemoryPoolOptions},
};
use skycuda_cudnn::{
    BuiltGraph, CudnnHandle, DataType, GraphBuilder, HeurMode, MatmulAttributes, TensorAttributes,
    TensorId,
};

#[test]
fn matmul_end_to_end() {
    let device = Device::from_index(0).unwrap();
    let stream = device.create_stream().unwrap();
    let handle = CudnnHandle::new(&stream).unwrap();
    let pool = device
        .create_host_pinned_memory_pool(HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1024,
        })
        .unwrap();

    // Shapes: A[B,M,K] @ B[B,K,N] = C[B,M,N]
    let (b, m, k, n) = (2usize, 32usize, 64usize, 48usize);

    // Host inputs (row-major).
    let a_host: Vec<f16> = (0..(b * m * k))
        .map(|i| f16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
        .collect();
    let b_host: Vec<f16> = (0..(b * k * n))
        .map(|i| f16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
        .collect();

    let mut a_memory = pool.alloc_zeroed(a_host.len() * size_of::<f16>()).unwrap();
    a_memory
        .view_mut()
        .unwrap()
        .copy_from_slice(bytemuck::cast_slice(&a_host));

    let mut b_memory = pool.alloc_zeroed(b_host.len() * size_of::<f16>()).unwrap();
    b_memory
        .view_mut()
        .unwrap()
        .copy_from_slice(bytemuck::cast_slice(&b_host));

    let mut c_memory = pool.alloc_zeroed(b * m * n * size_of::<f16>()).unwrap();

    // Build the graph.
    let graph = GraphBuilder::new(&handle).unwrap();
    graph.set_io_data_type(DataType::Half);
    graph.set_intermediate_data_type(DataType::Float);
    graph.set_compute_data_type(DataType::Float);

    let a = graph
        .tensor(
            &TensorAttributes::new(&[b, m, k], &[m * k, k, 1])
                .data_type(DataType::Half)
                .uid(TensorId(1))
                .name("A"),
        )
        .unwrap();
    let bt = graph
        .tensor(
            &TensorAttributes::new(&[b, k, n], &[k * n, n, 1])
                .data_type(DataType::Half)
                .uid(TensorId(2))
                .name("B"),
        )
        .unwrap();
    let c = graph
        .matmul(
            a,
            bt,
            &MatmulAttributes::new().compute_data_type(DataType::Float),
        )
        .unwrap();
    c.set_output(true);
    c.set_uid(TensorId(3));
    c.set_data_type(DataType::Half);

    graph.validate().unwrap();
    let graph = graph.build(&[HeurMode::A, HeurMode::B]).unwrap();

    let ws_size = graph.workspace_size();
    let mut workspace = pool.alloc_zeroed(ws_size).unwrap();

    {
        let mut variant_pack = graph.new_variant_pack(&handle);
        variant_pack
            .add(TensorId(1), &a_memory)
            .unwrap()
            .add(TensorId(2), &b_memory)
            .unwrap()
            .add_mut(TensorId(3), &mut c_memory)
            .unwrap();

        unsafe {
            graph.execute(&variant_pack, &mut workspace).unwrap();
        }
    }

    let c_host: &[f16] = bytemuck::cast_slice(c_memory.view_mut().unwrap());

    // CPU reference.
    let mut reference = vec![0f32; b * m * n];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    let av = a_host[bi * m * k + mi * k + ki].to_f32();
                    let bv = b_host[bi * k * n + ki * n + ni].to_f32();
                    acc += av * bv;
                }
                reference[bi * m * n + mi * n + ni] = acc;
            }
        }
    }

    let mut max_abs_err = 0f32;
    for (got, want) in c_host.iter().zip(reference.iter()) {
        max_abs_err = max_abs_err.max((got.to_f32() - want).abs());
    }
    assert!(max_abs_err.is_finite());
    assert!(
        max_abs_err < 0.5,
        "matmul result diverged from reference: max abs err = {max_abs_err}"
    );
}

/// Builds `C[2,32,48] = A[2,32,64] @ B[2,64,48]` in half precision with uids 1, 2, 3.
fn build_matmul_graph(handle: &CudnnHandle) -> BuiltGraph {
    let (b, m, k, n) = (2usize, 32usize, 64usize, 48usize);
    let graph = GraphBuilder::new(handle).unwrap();
    graph.set_io_data_type(DataType::Half);
    graph.set_intermediate_data_type(DataType::Float);
    graph.set_compute_data_type(DataType::Float);

    let a = graph
        .tensor(&TensorAttributes::new(&[b, m, k], &[m * k, k, 1]).uid(TensorId(1)))
        .unwrap();
    let bt = graph
        .tensor(&TensorAttributes::new(&[b, k, n], &[k * n, n, 1]).uid(TensorId(2)))
        .unwrap();
    let c = graph.matmul(a, bt, &MatmulAttributes::new()).unwrap();
    c.set_output(true);
    c.set_uid(TensorId(3));

    graph.validate().unwrap();
    graph.build(&[HeurMode::A, HeurMode::B]).unwrap()
}

fn setup() -> (CudnnHandle, HostPinnedMemoryPool) {
    let device = Device::from_index(0).unwrap();
    let stream = device.create_stream().unwrap();
    let handle = CudnnHandle::new(&stream).unwrap();
    let pool = device
        .create_host_pinned_memory_pool(HostPinnedMemoryPoolOptions {
            max_reserved_free_bytes: 1024,
        })
        .unwrap();
    (handle, pool)
}

#[test]
fn tensor_requirements_after_build() {
    let (handle, _pool) = setup();
    let graph = build_matmul_graph(&handle);

    let a = graph.tensor_requirements(TensorId(1)).unwrap();
    assert!(!a.is_output());
    assert_eq!(a.data_type(), DataType::Half);
    assert_eq!(a.min_len(), Some(2 * 32 * 64 * 2));

    let c = graph.tensor_requirements(TensorId(3)).unwrap();
    assert!(c.is_output());
    assert_eq!(c.dim(), &[2, 32, 48]);
    assert_eq!(c.min_len(), Some(2 * 32 * 48 * 2));

    assert!(graph.tensor_requirements(TensorId(4)).is_none());
    assert_eq!(graph.all_tensor_requirements().count(), 3);
}

#[test]
#[should_panic(expected = "needs at least")]
fn variant_pack_rejects_short_allocation() {
    let (handle, pool) = setup();
    let graph = build_matmul_graph(&handle);
    let a_memory = pool.alloc_zeroed(2 * 32 * 64 * 2 - 1).unwrap();
    graph
        .new_variant_pack(&handle)
        .add(TensorId(1), &a_memory)
        .unwrap();
}

#[test]
#[should_panic(expected = "use `add_mut`")]
fn variant_pack_rejects_immutable_output() {
    let (handle, pool) = setup();
    let graph = build_matmul_graph(&handle);
    let c_memory = pool.alloc_zeroed(2 * 32 * 48 * 2).unwrap();
    graph
        .new_variant_pack(&handle)
        .add(TensorId(3), &c_memory)
        .unwrap();
}

#[test]
#[should_panic(expected = "not a non-virtual tensor")]
fn variant_pack_rejects_unknown_uid() {
    let (handle, pool) = setup();
    let graph = build_matmul_graph(&handle);
    let memory = pool.alloc_zeroed(16).unwrap();
    graph
        .new_variant_pack(&handle)
        .add(TensorId(7), &memory)
        .unwrap();
}

#[test]
#[should_panic(expected = "already added")]
fn variant_pack_rejects_duplicate_uid() {
    let (handle, pool) = setup();
    let graph = build_matmul_graph(&handle);
    let a_memory = pool.alloc_zeroed(2 * 32 * 64 * 2).unwrap();
    let mut pack = graph.new_variant_pack(&handle);
    pack.add(TensorId(1), &a_memory)
        .unwrap()
        .add(TensorId(1), &a_memory)
        .unwrap();
}
