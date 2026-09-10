use std::{
    env,
    path::{Path, PathBuf},
};

fn first_existing(candidates: &[PathBuf], needle: &str) -> Option<PathBuf> {
    candidates.iter().find(|d| d.join(needle).exists()).cloned()
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let csrc = manifest_dir.join("csrc");
    let shim_cpp = csrc.join("cudnn_fe_shim.cpp");
    let shim_h = csrc.join("cudnn_fe_shim.h");

    // Upstream header-only cudnn-frontend tree lives at repo_root/include.
    // From rust/cudnn-frontend-sys that is ../../include.
    let fe_include = manifest_dir.join("cudnn-frontend/include");
    assert!(
        fe_include.join("cudnn_frontend.h").exists(),
        "cudnn_frontend.h not found in {}. Set CUDNN_FRONTEND_INCLUDE_DIR.",
        fe_include.display()
    );

    let cuda_root = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/usr/local/cuda"));
    let cuda_include = {
        let candidates = [
            cuda_root.join("include"),
            PathBuf::from("/usr/local/cuda/include"),
        ];
        first_existing(&candidates, "cuda_runtime.h").unwrap_or_else(|| {
            panic!(
                "cuda_runtime.h not found (looked under {}). Set CUDA_PATH.",
                cuda_root.display()
            )
        })
    };

    // cuDNN header (cudnn.h). Search env, then common locations.
    let cudnn_include = {
        let mut candidates = Vec::new();
        if let Ok(d) = env::var("CUDNN_INCLUDE_DIR") {
            candidates.push(PathBuf::from(d));
        }
        candidates.push(cuda_include.clone());
        candidates.push(PathBuf::from("/usr/include/x86_64-linux-gnu"));
        candidates.push(PathBuf::from("/usr/include"));
        first_existing(&candidates, "cudnn.h").unwrap_or_else(|| {
            panic!("cudnn.h not found. Set CUDNN_INCLUDE_DIR to the directory containing it.")
        })
    };

    // --- Compile the C++ shim ---
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(&shim_cpp)
        .include(&csrc)
        .include(&fe_include)
        .include(&cuda_include)
        .include(&cudnn_include)
        .define("NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING", None)
        // The frontend headers are third-party; silence their warnings.
        .flag_if_supported("-Wno-deprecated-declarations")
        .flag_if_supported("-Wno-unused-parameter");
    build.compile("cudnn_fe_shim");

    // Dynamic loading: cuDNN + CUDA symbols resolve at runtime via dlopen.
    // We only need libdl at link time (no -lcudnn, no -lnvrtc).
    println!("cargo:rustc-link-lib=dylib=dl");

    // --- Generate FFI bindings from the pure-C shim header ---
    let bindings = bindgen::Builder::default()
        .header(shim_h.to_string_lossy())
        .clang_arg(format!("-I{}", csrc.display()))
        .allowlist_function("cudnn_fe_.*")
        .allowlist_type("cudnnFe.*")
        .default_enum_style(bindgen::EnumVariation::Consts)
        .generate_comments(true)
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed to generate FFI bindings for cudnn_fe_shim.h");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    // --- Rerun triggers ---
    for p in [&shim_cpp, &shim_h] {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    for v in [
        "CUDNN_FRONTEND_INCLUDE_DIR",
        "CUDA_PATH",
        "CUDA_HOME",
        "CUDNN_INCLUDE_DIR",
    ] {
        println!("cargo:rerun-if-env-changed={}", v);
    }
    let _ = Path::new(&fe_include); // silence unused in some cfgs
}
