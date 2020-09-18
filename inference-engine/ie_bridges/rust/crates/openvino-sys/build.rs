use std::path::{Path, PathBuf};

use bindgen;
use cmake;

fn main() {
    // Trigger rebuild on changes to build.rs and Cargo.toml and every source file.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    let cb = |p: PathBuf| println!("cargo:rerun-if-changed={}", p.display());
    visit_dirs(Path::new("src"), &cb).expect("to visit source files");

    // Generate bindings from C header.
    let openvino_c_api_header =
        file("upstream/inference-engine/ie_bridges/c/include/c_api/ie_c_api.h");
    let bindings = bindgen::Builder::default()
        .header(openvino_c_api_header.to_string_lossy())
        // While understanding the warnings in https://docs.rs/bindgen/0.36.0/bindgen/struct.Builder.html#method.rustified_enum
        // that these enums could result in unspecified behavior if constructed from an invalid
        // value, the expectation here is that OpenVINO only returns valid layout and precision
        // values. This assumption is reasonable because otherwise OpenVINO itself would be broken.
        .rustified_enum("layout_e")
        .rustified_enum("precision_e")
        .rustified_enum("resize_alg_e")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("generate C API bindings");
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    // Find OpenVINO libraries, either pre-installed or by building from source.
    let library_paths = if let Some(path) = std::env::var_os("OPENVINO_INSTALL_DIR") {
        // Given a path to an OpenVINO installation (e.g. https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html),
        // find the paths to the necessary libraries and add them to the library search path.
        // It would be preferable to use pkg-config here to retrieve the libraries when they are
        // installed system-wide but there are issues:
        //  - OpenVINO does not install itself a system library, e.g., through ldconfig.
        //  - OpenVINO relies on a `plugins.xml` file for finding target-specific libraries
        //    and it is unclear how we would discover this in a system-install scenario.
        let deployment_tools = dir(path).join("deployment_tools");
        let openvino_libraries = deployment_tools.join("inference_engine/lib/intel64");
        let tbb_libraries = deployment_tools.join("inference_engine/external/tbb/lib");
        let ngraph_libraries = deployment_tools.join("ngraph/lib");
        vec![openvino_libraries, tbb_libraries, ngraph_libraries]
    } else {
        // Build OpenVINO with CMake.
        fn cmake(out_dir: &str) -> cmake::Config {
            let mut config = cmake::Config::new("upstream");
            config
                .very_verbose(true)
                .define("NGRAPH_ONNX_IMPORT_ENABLE", "ON")
                .define("ENABLE_OPENCV", "OFF")
                .define("ENABLE_CPPLINT", "OFF")
                // Because OpenVINO by default wants to build its binaries in its own tree, we must specify
                // that we actually want them in Cargo's output directory.
                .define("OUTPUT_ROOT", out_dir);
            config
        }

        // Specifying the build targets reduces the build time somewhat; this one will trigger
        // builds for other necessary shared libraries (e.g. inference_engine).
        let build_path = cmake(out.to_str().unwrap())
            .build_target("inference_engine_c_api")
            .build();

        // Unfortunately, `inference_engine_c_api` will not build the OpenVINO plugins used for
        // the actual computation. Here we re-run CMake for each plugin the user specifies using
        // Cargo features (see `Cargo.toml`).
        for plugin in get_plugin_target_from_features() {
            cmake(out.to_str().unwrap()).build_target(plugin).build();
        }

        // Collect the locations of the libraries. Note that ngraph should also be found with the
        // built OpenVINO libraries.
        let openvino_libraries =
            find_and_append_cmake_build_type(build_path.join("bin/intel64")).join("lib");

        // Copy the TBB libraries into the OpenVINO library directory. Since ngraph already exists
        // here and because the TBB directory is weirdly downloaded in-tree rather than under target
        // (meaning that the TBB path would be stripped from LD_LIBRARY_PATH, see
        // https://doc.rust-lang.org/cargo/reference/environment-variables.html#dynamic-library-paths),
        // copying the files over makes some sense. Also, I have noticed compatibility issues with
        // pre-installed libtbb (on some systems, the nodes_count symbol is not present in the
        // system-provided libtbb) so it may be important to include OpenVINO's version of libtbb
        // here.
        let tbb_libraries = dir("upstream/inference-engine/temp/tbb/lib");
        visit_dirs(&tbb_libraries, &|from: PathBuf| {
            let to = openvino_libraries.join(from.file_name().unwrap());
            println!("Copying {} to {}", from.display(), to.display());
            std::fs::copy(from, to).expect("failed copying TBB libraries");
        })
        .expect("failed visiting TBB directory");

        vec![openvino_libraries]
    };

    // Check that the plugins.xml file exists and create an environment variable recording this
    // location.
    let openvino_plugins = library_paths[0].join("plugins.xml");
    assert!(
        openvino_plugins.is_file(),
        "Unable to find OpenVINO plugins.xml file at: {}",
        openvino_plugins.display()
    );
    println!(
        "cargo:rustc-env=OPENVINO_LIB_DIR={}",
        library_paths[0].display()
    );

    // Configure the build-time library search paths.
    for library in &library_paths {
        add_buildtime_library_path(library);
    }

    // Dynamically link the necessary OpenVINO libraries.
    let libraries = vec![
        "inference_engine",
        "inference_engine_legacy",
        "inference_engine_transformations",
        "inference_engine_c_api",
        "ngraph",
        "tbb",
    ];
    for library in &libraries {
        println!("cargo:rustc-link-lib=dylib={}", library);
    }
}

/// Ensure a path is valid and add it to the build-time library search path.
fn add_buildtime_library_path<P: AsRef<Path>>(path: P) {
    let path = path.as_ref();
    assert!(
        path.is_dir(),
        "Invalid library search path: {}",
        path.display()
    );
    println!("cargo:rustc-link-search=native={}", path.display());
}

/// Add a path (or concatenated paths with ':') to rustc's runtime library search path. Note that
/// this will *NOT* set up the library paths for executables (e.g. cargo test, cargo run).
#[allow(dead_code)]
fn add_runtime_library_paths(paths: &[PathBuf]) {
    let var_name = if cfg!(target_os = "linux") {
        "LD_LIBRARY_PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_FALLBACK_LIBRARY_PATH"
    } else if cfg!(target_os = "windows") {
        "PATH"
    } else {
        panic!("No way to know how to add to the runtime library search path")
    };

    let concatenated_library_paths = paths
        .iter()
        .map(|l| l.to_str().unwrap())
        .collect::<Vec<_>>()
        .join(":");

    println!(
        "cargo:rustc-env={}={}",
        var_name, concatenated_library_paths
    );
}

/// Canonicalize a path as well as verify that it exists.
fn file<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    if !path.exists() || !path.is_file() {
        panic!("Unable to find file: {}", path.display())
    }
    path.canonicalize()
        .expect("to be able to canonicalize the path")
}

/// Canonicalize a path as well as verify that it exists.
fn dir<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    if !path.exists() || !path.is_dir() {
        panic!("Unable to find directory: {}", path.display())
    }
    path.canonicalize()
        .expect("to be able to canonicalize the path")
}

/// Helper for recursively visiting the files in this directory; see https://doc.rust-lang.org/std/fs/fn.read_dir.html.
fn visit_dirs(dir: &Path, cb: &dyn Fn(PathBuf)) -> std::io::Result<()> {
    if dir.is_dir() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit_dirs(&path, cb)?;
            } else {
                cb(path);
            }
        }
    }
    Ok(())
}

/// Determine CMake targets for the various OpenVINO plugins. The plugin mapping is available in
/// OpenVINO's `plugins.xml` file and, usign that, this function wires up the exposed Cargo
/// features of openvino-sys to the correct CMake target.
fn get_plugin_target_from_features() -> Vec<&'static str> {
    let mut plugins = vec![];
    if cfg!(feature = "all") {
        plugins.push("ie_plugins")
    } else {
        if cfg!(feature = "cpu") {
            plugins.push("MKLDNNPlugin")
        }
        if cfg!(feature = "gpu") {
            plugins.push("clDNNPlugin")
        }
        if cfg!(feature = "gna") {
            plugins.push("GNAPlugin")
        }
        if cfg!(feature = "hetero") {
            plugins.push("HeteroPlugin")
        }
        if cfg!(feature = "multi") {
            plugins.push("MultiDevicePlugin")
        }
        if cfg!(feature = "myriad") {
            plugins.push("myriadPlugin")
        }
    }
    assert!(!plugins.is_empty());
    plugins
}

/// According to https://docs.rs/cmake/0.1.44/cmake/struct.Config.html#method.profile, the cmake
/// crate will tries to infer the appropriate CMAKE_BUILD_TYPE from a combination of Rust opt-level
/// and debug. To avoid duplicating https://docs.rs/cmake/0.1.44/src/cmake/lib.rs.html#553-559, this
/// helper searches for build type directories and appends it to the path if a result is found; this
/// will panic otherwise.
fn find_and_append_cmake_build_type(build_path: PathBuf) -> PathBuf {
    let types = ["Debug", "Release", "RelWithDebInfo", "MinSizeRel"];
    let found: Vec<_> = types
        .iter()
        .filter(|&&t| build_path.join(t).is_dir())
        .collect();
    match found.len() {
        0 => panic!(
            "No CMake build directory found in {}; expected one of {:?}",
            build_path.display(),
            types
        ),
        1 => build_path.join(found[0]),
        _ => panic!(
            "Too many CMake build directories found in {}",
            build_path.display()
        ),
    }
}
