use half::f16;
use skycuda::{
    device::Device,
    host_pinned_memory_pool::{HostPinnedMemoryPool, HostPinnedMemoryPoolOptions},
};
use skycuda_cudnn::{
    BuildPlanPolicy, BuiltGraph, CudnnHandle, DataType, GraphBuilder, HeurMode, MatmulAttributes,
    TensorAttributes, TensorId, backend_version,
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

    let reference = matmul_reference(&a_host, &b_host, b, m, k, n);
    assert_close(c_host, &reference);
}

/// Row-major batched matmul `C[b,m,n] = A[b,m,k] @ B[b,k,n]` in f32.
fn matmul_reference(a: &[f16], bt: &[f16], b: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut reference = vec![0f32; b * m * n];
    for bi in 0..b {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    acc += a[bi * m * k + mi * k + ki].to_f32()
                        * bt[bi * k * n + ki * n + ni].to_f32();
                }
                reference[bi * m * n + mi * n + ni] = acc;
            }
        }
    }
    reference
}

fn assert_close(got: &[f16], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let mut max_abs_err = 0f32;
    for (got, want) in got.iter().zip(want) {
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
    build_matmul_graph_with_policy(handle, BuildPlanPolicy::HeuristicsChoice)
}

fn build_matmul_graph_with_policy(handle: &CudnnHandle, policy: BuildPlanPolicy) -> BuiltGraph {
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
    graph
        .build_with_policy(&[HeurMode::A, HeurMode::B], policy)
        .unwrap()
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

fn build_override_matmul_graph(handle: &CudnnHandle) -> BuiltGraph {
    let (b, m, k, n) = (1usize, 32usize, 64usize, 32usize);
    let graph = GraphBuilder::new(handle).unwrap();
    graph.set_io_data_type(DataType::Half);
    graph.set_intermediate_data_type(DataType::Float);
    graph.set_compute_data_type(DataType::Float);
    graph.set_override_shape_enabled(true);

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
    graph.build(&[HeurMode::A, HeurMode::Fallback]).unwrap()
}

#[test]
fn matmul_shape_override_end_to_end() {
    if backend_version() < 92100 {
        return;
    }
    let (handle, pool) = setup();
    let graph = build_override_matmul_graph(&handle);
    assert!(graph.is_override_shape_enabled());

    for (b, m, k, n) in [(2usize, 48usize, 96usize, 40usize), (1, 128, 128, 128)] {
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

        let mut variant_pack = graph.new_variant_pack(&handle);
        variant_pack
            .add_with_shape(TensorId(1), &a_memory, &[b, m, k], &[m * k, k, 1])
            .unwrap()
            .add_with_shape(TensorId(2), &b_memory, &[b, k, n], &[k * n, n, 1])
            .unwrap()
            .add_mut_with_shape(TensorId(3), &mut c_memory, &[b, m, n], &[m * n, n, 1])
            .unwrap();
        assert!(variant_pack.has_shape_overrides());

        let ws_size = graph.workspace_size_for(&variant_pack).unwrap();
        let mut workspace = pool.alloc_zeroed(ws_size).unwrap();
        unsafe {
            graph.execute(&variant_pack, &mut workspace).unwrap();
        }
        drop(variant_pack);

        let reference = matmul_reference(&a_host, &b_host, b, m, k, n);
        assert_close(
            bytemuck::cast_slice(c_memory.view_mut().unwrap()),
            &reference,
        );
    }
}

#[test]
fn workspace_size_for_without_overrides_matches_static() {
    if backend_version() < 92100 {
        return;
    }
    let (handle, pool) = setup();
    let graph = build_override_matmul_graph(&handle);

    let a_memory = pool.alloc_zeroed(32 * 64 * 2).unwrap();
    let mut variant_pack = graph.new_variant_pack(&handle);
    variant_pack.add(TensorId(1), &a_memory).unwrap();
    assert!(!variant_pack.has_shape_overrides());
    assert_eq!(
        graph.workspace_size_for(&variant_pack).unwrap(),
        graph.workspace_size()
    );
}

#[test]
#[should_panic(expected = "set_override_shape_enabled")]
fn variant_pack_rejects_override_without_flag() {
    let (handle, pool) = setup();
    let graph = build_matmul_graph(&handle);
    let a_memory = pool.alloc_zeroed(4 * 32 * 64 * 2).unwrap();
    graph
        .new_variant_pack(&handle)
        .add_with_shape(TensorId(1), &a_memory, &[4, 32, 64], &[32 * 64, 64, 1])
        .unwrap();
}

#[test]
#[should_panic(expected = "needs at least")]
fn variant_pack_rejects_short_allocation_for_override_shape() {
    if backend_version() < 92100 {
        panic!("skipped: needs at least cuDNN 9.21.0");
    }
    let (handle, pool) = setup();
    let graph = build_override_matmul_graph(&handle);
    // Sized for the cache shape (1,32,64), but the override shape is larger.
    let a_memory = pool.alloc_zeroed(32 * 64 * 2).unwrap();
    graph
        .new_variant_pack(&handle)
        .add_with_shape(TensorId(1), &a_memory, &[2, 48, 96], &[48 * 96, 96, 1])
        .unwrap();
}

#[test]
#[should_panic(expected = "rank")]
fn variant_pack_rejects_override_rank_mismatch() {
    if backend_version() < 92100 {
        panic!("skipped: needs at least cuDNN 9.21.0");
    }
    let (handle, pool) = setup();
    let graph = build_override_matmul_graph(&handle);
    let a_memory = pool.alloc_zeroed(32 * 64 * 2).unwrap();
    graph
        .new_variant_pack(&handle)
        .add_with_shape(TensorId(1), &a_memory, &[32, 64], &[64, 1])
        .unwrap();
}

#[test]
fn heuristics_choice_builds_only_candidate() {
    let (handle, _pool) = setup();
    let graph = build_matmul_graph(&handle);

    let candidate = graph.candidate_plan();
    assert!(candidate.is_built());
    assert!(candidate.is_candidate());
    assert_eq!(candidate.workspace_size().unwrap(), graph.workspace_size());
    assert_eq!(graph.max_workspace_size(), graph.workspace_size());

    // Every plan is describable even when it was not built.
    for plan in graph.plans() {
        assert!(!plan.name().is_empty());
        let config = plan.engine_config().unwrap();
        assert!(config.engine_id < graph.engine_count());
        let _ = plan.behavior_notes();
        let _ = plan.numerical_notes();
        if plan.index() != candidate.index() {
            assert!(!plan.is_built());
            assert!(plan.workspace_size().is_err());
        }
    }
}

#[test]
fn build_all_plans_and_execute_each() {
    let (handle, pool) = setup();
    let graph = build_matmul_graph_with_policy(&handle, BuildPlanPolicy::All);

    let plans: Vec<_> = graph.plans().filter(|p| p.is_built()).collect();
    assert!(!plans.is_empty());
    assert!(plans.iter().any(|p| p.is_candidate()));
    assert_eq!(graph.plans().len(), graph.execution_plan_count());
    println!(
        "{} of {} plans built",
        plans.len(),
        graph.execution_plan_count()
    );

    for plan in &plans {
        println!("{plan:?} {:?}", plan.engine_config().unwrap());
        assert!(!plan.name().is_empty());
        assert!(plan.workspace_size().unwrap() <= graph.max_workspace_size());
        for choice in plan.engine_config().unwrap().knobs {
            assert_ne!(choice.knob_type, skycuda_cudnn::KnobType::NotSet);
        }
        for note in plan.numerical_notes() {
            assert_ne!(note, skycuda_cudnn::NumericalNote::NotSet);
        }
    }
    let knobs = graph
        .knobs_for_engine(plans[0].engine_config().unwrap().engine_id)
        .unwrap();
    for knob in knobs {
        assert!(knob.min_value <= knob.max_value);
    }

    let (b, m, k, n) = (2usize, 32usize, 64usize, 48usize);
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
    let mut workspace = pool
        .alloc_zeroed(graph.max_workspace_size().max(1))
        .unwrap();
    let reference = matmul_reference(&a_host, &b_host, b, m, k, n);

    for plan in &plans {
        c_memory.view_mut().unwrap().fill(0);
        {
            let mut variant_pack = graph.new_variant_pack(&handle);
            variant_pack
                .add(TensorId(1), &a_memory)
                .unwrap()
                .add(TensorId(2), &b_memory)
                .unwrap()
                .add_mut(TensorId(3), &mut c_memory)
                .unwrap();
            assert_eq!(
                plan.workspace_size_for(&variant_pack).unwrap(),
                plan.workspace_size().unwrap()
            );
            unsafe { plan.execute(&variant_pack, &mut workspace).unwrap() };
        }
        let c_host: &[f16] = bytemuck::cast_slice(c_memory.view_mut().unwrap());
        assert_close(c_host, &reference);
    }
}
