use cudnn_frontend::{
    CudnnHandle, DataType, GraphBuilder, HeurMode, MatmulAttributes, TensorAttributes,
};
use half::f16;
use skycuda::{device::Device, host_pinned_memory_pool::HostPinnedMemoryPoolOptions};

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
    let (b, m, k, n) = (2i64, 32i64, 64i64, 48i64);

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

    let mut c_memory = pool.alloc_zeroed((b * m * n) as usize).unwrap();

    // Build the graph.
    let graph = GraphBuilder::new(&handle).unwrap();
    graph.set_io_data_type(DataType::Half).unwrap();
    graph.set_intermediate_data_type(DataType::Float).unwrap();
    graph.set_compute_data_type(DataType::Float).unwrap();

    let a = graph
        .tensor(
            &TensorAttributes::new(&[b, m, k], &[m * k, k, 1])
                .data_type(DataType::Half)
                .uid(1)
                .name("A"),
        )
        .unwrap();
    let bt = graph
        .tensor(
            &TensorAttributes::new(&[b, k, n], &[k * n, n, 1])
                .data_type(DataType::Half)
                .uid(2)
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
    c.set_output(true).unwrap();
    c.set_uid(3).unwrap();
    c.set_data_type(DataType::Half).unwrap();

    graph.validate().unwrap();
    let graph = graph.build(&[HeurMode::A, HeurMode::B]).unwrap();

    let ws_size = graph.workspace_size().unwrap();
    let mut workspace = pool.alloc_zeroed(ws_size as usize).unwrap();

    {
        let mut variant_pack = graph.new_variant_pack(&handle);
        variant_pack
            .add(1, &a_memory)
            .unwrap()
            .add(2, &b_memory)
            .unwrap()
            .add_mut(3, &mut c_memory)
            .unwrap();

        unsafe {
            graph.execute(&variant_pack, &mut workspace).unwrap();
        }
    }

    let c_host: &[f16] = bytemuck::cast_slice(c_memory.view_mut().unwrap());

    // CPU reference.
    let mut reference = vec![0f32; (b * m * n) as usize];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    let av = a_host[(bi * m * k + mi * k + ki) as usize].to_f32();
                    let bv = b_host[(bi * k * n + ki * n + ni) as usize].to_f32();
                    acc += av * bv;
                }
                reference[(bi * m * n + mi * n + ni) as usize] = acc;
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
