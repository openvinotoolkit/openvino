# openvino-rs

[![Build Status](https://github.com/abrown/openvino-rs/workflows/CI/badge.svg)][ci]
[![Documentation Status](https://docs.rs/openvino-rs/badge.svg)][docs]

This repository contains the [openvino-sys] crate (low-level, unsafe bindings) and the [openvino] crate (high-level, 
ergonomic bindings) for accessing OpenVINO™ functionality in Rust.

[openvino-sys]: crates/openvino-sys
[openvino]: crates/openvino
[upstream]: crates/openvino-sys/upstream
[docs]: https://docs.rs/openvino
[ci]: https://github.com/abrown/openvino-rs/actions?query=workflow%3ACI


### Prerequisites

The [openvino-sys] crate creates bindings to the OpenVINO™ C API using `bindgen`; this requires a local installation of
`libclang`. Also, be sure to retrieve all Git submodules.



### Build from an OpenVINO™ installation

```shell script
git submodule update --init --recursive
OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo build -v
source /opt/intel/openvino/bin/setupvars.sh && \
  OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo test -v
```

The quickest method to build [openvino] and [openvino-sys] is with a local installation of OpenVINO™ (see, e.g., 
[installing from an apt repository][install-apt]). Provide the `OPENVINO_INSTALL_DIR` environment variable to any 
`cargo` commands and ensure that the environment is configured (i.e. library search paths) using OpenVINO™'s 
`setupvars.sh` before running any executables that use these libraries. This also applies to any crates using these 
libraries as a dependency.

[install-apt]: https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_apt.html



### Build from OpenVINO™ sources

```shell script
git submodule update --init --recursive
cargo build -vv
cargo test
```

[openvino] and [openvino-sys] can also be built directly from OpenVINO™'s source code using CMake. This build process
can be quite slow and there are quite a few dependencies. Some notes:
 - first, install the necessary packages to build OpenVINO™; steps are included in the [CI workflow](.github/workflows)
   but reference the [OpenVINO™ build documentation](https://github.com/openvinotoolkit/openvino/blob/master/build-instruction.md)
   for the full documentation
 - OpenVINO™ has a plugin system for device-specific libraries (e.g. GPU); building all of these libraries along with the
   core inference libraries can take >20 minutes. To avoid over-long build times, [openvino-sys] exposes several
   Cargo features. By default, [openvino-sys] will only build the CPU plugin; to build all plugins, use 
   `--features all` (see [Cargo.toml](crates/openvino-sys/Cargo.toml)).
 - OpenVINO™ includes other libraries (e.g. ngraph, tbb); see the [build.rs](crates/openvino-sys/build.rs) file for how
   these are linked to these libraries.



### Use

After building:
  - peruse the documentation for the [openvino crate][docs]; this is the library you likely want to interact with from
  Rust.
  - follow along with the [classification example](crates/openvino/tests/classify.rs); this example classifies an image 
  using a [pre-built model](crates/openvino/tests/fixture). The examples (and all tests) are runnable using `cargo test`
  (or `OPENVINO_INSTALL_DIR=/opt/intel/openvino cargo test` when building from an installation).



### License

`openvino-rs` is released under the same license as OpenVINO™: the [Apache License Version 2.0][license]. By 
contributing to the project, you agree to the license and copyright terms therein and release your contribution under
these terms.

[license]: LICENSE
