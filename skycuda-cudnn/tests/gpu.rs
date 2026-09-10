use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut};
use half::f16;

use cudnn_frontend::{
    CudnnHandle, DataType, GraphBuilder, HeurMode, MatmulAttributes, SdpaAttributes,
    TensorAttributes,
};

#[test]
fn matmul_end_to_end() {
    let ctx = CudaContext::new(0).expect("cuda context");
    let stream = ctx.default_stream();
    let handle = CudnnHandle::new(&stream).unwrap();

    // Shapes: A[B,M,K] @ B[B,K,N] = C[B,M,N]
    let (b, m, k, n) = (2i64, 32i64, 64i64, 48i64);

    // Host inputs (row-major).
    let a_host: Vec<f16> = (0..(b * m * k))
        .map(|i| f16::from_f32(((i % 7) as f32 - 3.0) * 0.1))
        .collect();
    let b_host: Vec<f16> = (0..(b * k * n))
        .map(|i| f16::from_f32(((i % 5) as f32 - 2.0) * 0.1))
        .collect();

    let a_dev = stream.clone_htod(&a_host).unwrap();
    let b_dev = stream.clone_htod(&b_host).unwrap();
    let mut c_dev = stream.alloc_zeros::<f16>((b * m * n) as usize).unwrap();

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
    let workspace = stream.alloc_zeros::<u8>(ws_size.max(1) as usize).unwrap();

    {
        let (a_ptr, _a_g) = a_dev.device_ptr(&stream);
        let (b_ptr, _b_g) = b_dev.device_ptr(&stream);
        let (c_ptr, _c_g) = c_dev.device_ptr_mut(&stream);
        let (ws_ptr, _ws_g) = workspace.device_ptr(&stream);

        unsafe {
            graph
                .execute(&handle, &[(1, a_ptr), (2, b_ptr), (3, c_ptr)], ws_ptr)
                .unwrap();
        }

        stream.synchronize().unwrap();
    }

    let c_host = stream.clone_dtoh(&c_dev).unwrap();

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
    assert!(
        max_abs_err < 0.5,
        "matmul result diverged from reference: max abs err = {max_abs_err}"
    );
}

#[test]
fn sdpa_forward_end_to_end() {
    let (b, h, s, d) = (1i64, 2i64, 64i64, 64i64);
    let attn_scale = 1.0f32 / (d as f32).sqrt();

    let ctx = CudaContext::new(0).expect("cuda context");
    let stream = ctx.default_stream();

    let handle = CudnnHandle::new(&stream).unwrap();

    let elems = (b * h * s * d) as usize;
    // Small values keep softmax well-conditioned.
    let mk = |seed: i64| -> Vec<f16> {
        (0..elems)
            .map(|i| f16::from_f32((((i as i64 + seed) % 11) as f32 - 5.0) * 0.02))
            .collect()
    };
    let q_host = mk(0);
    let k_host = mk(3);
    let v_host = mk(7);

    let q_dev = stream.clone_htod(&q_host).unwrap();
    let k_dev = stream.clone_htod(&k_host).unwrap();
    let v_dev = stream.clone_htod(&v_host).unwrap();
    let mut o_dev = stream.alloc_zeros::<f16>(elems).unwrap();

    let graph = GraphBuilder::new(&handle).unwrap();
    graph.set_io_data_type(DataType::Half).unwrap();
    graph.set_intermediate_data_type(DataType::Float).unwrap();
    graph.set_compute_data_type(DataType::Float).unwrap();

    let stride = [h * s * d, s * d, d, 1];
    let q = graph
        .tensor(
            &TensorAttributes::new(&[b, h, s, d], &stride)
                .data_type(DataType::Half)
                .uid(1)
                .name("Q"),
        )
        .unwrap();
    let k = graph
        .tensor(
            &TensorAttributes::new(&[b, h, s, d], &stride)
                .data_type(DataType::Half)
                .uid(2)
                .name("K"),
        )
        .unwrap();
    let v = graph
        .tensor(
            &TensorAttributes::new(&[b, h, s, d], &stride)
                .data_type(DataType::Half)
                .uid(3)
                .name("V"),
        )
        .unwrap();

    let out = graph
        .sdpa(
            q,
            k,
            v,
            &SdpaAttributes::new()
                .compute_data_type(DataType::Float)
                .attn_scale(attn_scale)
                .generate_stats(false)
                .name("sdpa"),
        )
        .unwrap();
    let o = out[0].expect("sdpa produced O");
    o.set_output(true).unwrap();
    o.set_dim(&[b, h, s, d]).unwrap();
    o.set_stride(&stride).unwrap();
    o.set_uid(4).unwrap();
    o.set_data_type(DataType::Half).unwrap();

    let graph = graph.build(&[HeurMode::A]).unwrap();

    let ws = graph.workspace_size().unwrap();
    let workspace = stream.alloc_zeros::<u8>(ws.max(1) as usize).unwrap();

    {
        let (qp, _qg) = q_dev.device_ptr(&stream);
        let (kp, _kg) = k_dev.device_ptr(&stream);
        let (vp, _vg) = v_dev.device_ptr(&stream);
        let (op, _og) = o_dev.device_ptr_mut(&stream);
        let (wp, _wg) = workspace.device_ptr(&stream);
        unsafe {
            graph
                .execute(&handle, &[(1, qp), (2, kp), (3, vp), (4, op)], wp)
                .unwrap();
        }
        stream.synchronize().unwrap();
    }

    let o_host = stream.clone_dtoh(&o_dev).unwrap();

    let v_max = v_host.iter().map(|x| x.to_f32().abs()).fold(0.0, f32::max);
    let mut all_zero = true;
    for x in &o_host {
        let f = x.to_f32();
        assert!(f.is_finite(), "non-finite SDPA output: {f}");
        assert!(
            f.abs() <= v_max + 1e-2,
            "SDPA output {f} exceeds V range {v_max}"
        );
        if f != 0.0 {
            all_zero = false;
        }
    }
    assert!(!all_zero, "SDPA output was entirely zero");
}
