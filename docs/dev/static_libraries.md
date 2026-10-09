# Building OpenVINO static libraries

## Contents

- [Introduction](#introduction)
- [System requirements](#system-requirements)
- [Configure OpenVINO runtime in CMake stage](#configure-openvino-runtime-in-the-cmake-stage)
- [Build static OpenVINO libraries](#build-static-openvino-libraries)
- [Link static OpenVINO runtime](#link-static-openvino-runtime)
- [Static OpenVINO libraries + Conditional compilation for particular models](#static-openvino-libraries--conditional-compilation-for-particular-models)
- [Building with static MSVC Runtime](#building-with-static-msvc-runtime)
- [Building static OpenVINO for a specific device](#building-static-openvino-for-a-specific-device)
- [Validating a static build](#validating-a-static-build)
- [Limitations](#limitations)
- [See also](#see-also)

## Introduction

Building static OpenVINO Runtime libraries allows to additionally reduce the size of a binary when it is used together with conditional compilation.
It is possible because not all interface symbols of OpenVINO Runtime libraries are exported to end users during a static build and can be removed by linker. See [Static OpenVINO libraries + Conditional compilation for particular models](#static-openvino-libraries--conditional-compilation-for-particular-models)

## System requirements

* CMake version 3.26 or higher must be used to build static OpenVINO libraries.
* Supported OSes:
    * Windows x64
    * Linux x64
    * All other OSes may work, but have not been explicitly tested

## Configure OpenVINO Runtime in the CMake stage

The default architecture of OpenVINO Runtime assumes that the following components are subject to dynamic loading during execution:
* (Device) Inference backends (CPU, GPU, NPU, MULTI, HETERO, etc.)
* (Model) Frontends (IR, ONNX, PDPD, TF, JAX, etc.)

With the static OpenVINO Runtime, all these modules should be linked into a final user application and **the list of modules/configuration must be known for the CMake configuration stage**. To minimize the total binary size, you can explicitly turn `OFF` unnecessary components. Use [CMake Options for Custom Compilation](cmake_options_for_custom_compilation.md) as a reference for OpenVINO CMake configuration.

For example, to enable only IR v11 reading and CPU inference capabilities, use:
```sh
cmake -DENABLE_INTEL_GPU=OFF \
      -DENABLE_INTEL_NPU=OFF \
      -DENABLE_TEMPLATE=OFF \
      -DENABLE_HETERO=OFF \
      -DENABLE_MULTI=OFF \
      -DENABLE_AUTO=OFF \
      -DENABLE_AUTO_BATCH=OFF \
      -DENABLE_OV_ONNX_FRONTEND=OFF \
      -DENABLE_OV_PADDLE_FRONTEND=OFF \
      -DENABLE_OV_TF_FRONTEND=OFF \
      -DENABLE_OV_TF_LITE_FRONTEND=OFF \
      -DENABLE_OV_JAX_FRONTEND=OFF \
      -DENABLE_OV_PYTORCH_FRONTEND=OFF \
      -DENABLE_OV_JAX_FRONTEND=OFF \
      -DENABLE_INTEL_CPU=ON \
      -DENABLE_OV_IR_FRONTEND=ON
```

> **NOTE**: Inference backends located in external repositories can also be used in a static build. Use `-DOPENVINO_EXTRA_MODULES=<path to external plugin root>` to enable them. `OpenVINODeveloperPackage.cmake` must not be used to build external plugins, only `OPENVINO_EXTRA_MODULES` is a working solution.

> **NOTE**: The `ENABLE_LTO` CMake option can also be passed to enable link time optimizations to reduce the binary size. But such property should also be enabled on the target which links with static OpenVINO libraries via `set_target_properties(<target_name> PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)`

## Build static OpenVINO libraries

To build OpenVINO Runtime in a static mode, you need to specify the additional CMake option:

```sh
cmake -DBUILD_SHARED_LIBS=OFF <all other CMake options> <openvino_sources root>
```

Then, use the usual CMake 'build' command:

```sh
cmake --build . --config Release -j12
```

> **NOTE**: Do not restrict the build to `--target openvino` — the install step below expects other components too (such as `openvino_c`), and fails if they were not built:
> ```
> file INSTALL cannot find ".../bin/intel64/Release/openvino_c.lib": No error.
> ```
> To build only a subset of targets, disable the components you don't need at configure time (e.g. `-DENABLE_JS=OFF`), or use a component-scoped install: `cmake --install . --component core`.

Then, the installation step:

```sh
cmake -DCMAKE_INSTALL_PREFIX=<install_root> -P cmake_install.cmake
```

The OpenVINO runtime is located in `<install_root>/runtime/lib`

> **NOTE**: Build artifacts default to the *source* tree, not the build directory. If you build more than one static configuration from the same checkout (e.g. per-device, or `/MD` vs `/MT`), pass a separate `-DOUTPUT_ROOT=<build_dir>` per configuration, or they will clobber each other's generated files.

## Link static OpenVINO Runtime

Once you build static OpenVINO Runtime libraries and install them, you can use one of the two ways to add them to your project:

### CMake interface

Just use CMake's `find_package` as usual and link `openvino::runtime`:

```cmake
find_package(OpenVINO REQUIRED)
target_link_libraries(<application> PRIVATE openvino::runtime)
```

`openvino::runtime` transitively adds all other static OpenVINO libraries to a linker command. 

### Pass libraries to linker directly

If you want to configure your project directly, you need to pass all libraries from `<install_root>/runtime/lib` to linker command.

> **NOTE**: Since the proper order of static libraries must be used (dependent library should come **before** dependency in a linker command), consider using the following compiler specific flags to link static OpenVINO libraries:

Microsoft Visual Studio compiler:
```sh
/WHOLEARCHIVE:<ov_library 0> /WHOLEARCHIVE:<ov_library 1> ...
```

GCC like compiler:
```sh
gcc main.cpp -Wl,--whole-archive <all libraries from <root>/runtime/lib> > -Wl,--no-whole-archive -o a.out
```

## Static OpenVINO libraries + Conditional compilation for particular models

OpenVINO Runtime can be compiled for particular models, as shown in the [Conditional compilation for particular models](conditional_compilation.md) guide.
The conditional compilation feature can be paired with static OpenVINO libraries to build even smaller end-user applications in terms of binary size. The following procedure can be used, (based on the detailed [Conditional compilation for particular models](conditional_compilation.md) guide):

* Build OpenVINO Runtime as usual with the CMake option of `-DSELECTIVE_BUILD=COLLECT`.
* Run target applications on target models and target platforms to collect traces.
* Build the final OpenVINO static Runtime with `-DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv -DBUILD_SHARED_LIBS=OFF`

## Building with static MSVC Runtime

In order to build with static MSVC runtime, use the special [OpenVINO toolchain](https://github.com/openvinotoolkit/openvino/blob/master/cmake/toolchains/mt.runtime.win32.toolchain.cmake) file:

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<openvino source dir>/cmake/toolchains/mt.runtime.win32.toolchain.cmake <other options>
```

> **NOTE**: all other dependent application and libraries must be built with the same `mt.runtime.win32.toolchain.cmake ` toolchain to have conformed values of the `MSVC_RUNTIME_LIBRARY` target property.

## Building static OpenVINO for a specific device

The general instructions above apply to every device. The notes below cover additional, device-specific behavior of `-DBUILD_SHARED_LIBS=OFF` that is not obvious from CMake errors alone.

### CPU

No additional options are required; CPU builds statically the same way as the common runtime described above.

### GPU

GPU builds statically with the default `-DGPU_RT_TYPE=OCL` runtime, no extra flags needed.

> **NOTE**: `OV_GPU_WITH_SYCL` (SYCL-enabled GPU runtime) is not compatible with `-DENABLE_LTO=ON` and fails CMake configuration with `FATAL_ERROR: Intel GPU plugin with SYCL is not supported with ENABLE_LTO=ON`. If SYCL is enabled for GPU, do not enable `ENABLE_LTO`.

### NPU

No additional options are required. `ENABLE_INTEL_NPU_INTERNAL`, which controls internal NPU tooling such as `compile_tool`, follows `BUILD_SHARED_LIBS` and is disabled by default in a static build, so the install manifest does not expect those tools ([cmake/features.cmake](../../cmake/features.cmake)). If you explicitly re-enable it with `-DENABLE_INTEL_NPU_INTERNAL=ON`, build the corresponding targets as well (or build the default target as described above), otherwise the install step will fail looking for binaries that were never built.

The NPU offline compiler is a prebuilt shared library and is **not available in static builds**: `ENABLE_INTEL_NPU_COMPILER` defaults to `${BUILD_SHARED_LIBS}` and cannot be forced `ON` for a static build. When it's unavailable, `CompilerAdapterFactory` falls back to the driver compiler (`NPU_COMPILER_TYPE=DRIVER`) instead of the plugin compiler (`NPU_COMPILER_TYPE=PLUGIN`) used by default in shared builds — see [compiler_adapter_factory.cpp](../../src/plugins/intel_npu/src/compiler_adapter/src/compiler_adapter_factory.cpp).

### Building multiple device configurations from one checkout

This only applies if you deliberately build *separate* static packages per device (for example, to validate each device independently, or to ship a smaller single-device binary) — not to a single build with several devices enabled together. Give each configuration its own `OUTPUT_ROOT` and install prefix (see the note above):

```sh
cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DOUTPUT_ROOT=$PWD/build_cpu
cmake --build build_cpu --config Release -j12
cmake --install build_cpu --config Release --prefix install_cpu

cmake -S . -B build_npu -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DOUTPUT_ROOT=$PWD/build_npu
cmake --build build_npu --config Release -j12
cmake --install build_npu --config Release --prefix install_npu
```

## Validating a static build

The existing OpenVINO samples can be used to validate a static build:

* `hello_query_device` reports the devices and plugins compiled into the binary; use it to confirm that the plugins you enabled (for example CPU, GPU, NPU) are present and loadable.
* `hello_classification` (or `benchmark_app`) loads a model and runs inference on a selected device, verifying that the statically-linked plugin and frontend also work correctly at run time, not just that they link successfully.

```sh
cmake -S samples/cpp -B samples_build -DOpenVINO_DIR=<install_root>/runtime/cmake
cmake --build samples_build --config Release
samples_build/intel64/Release/hello_query_device
samples_build/intel64/Release/hello_classification <model.xml> <image> <DEVICE>
```

These samples link against the static package the same way any consuming application would, following the steps described in [Link static OpenVINO Runtime](#link-static-openvino-runtime).

## Limitations

* Device support in a static build:

  | Device | Builds statically | Notes |
  |---|---|---|
  | CPU | Yes | |
  | GPU | Yes | See [GPU](#gpu) notes above |
  | NPU | Yes | See [NPU](#npu) notes above |
  | MULTI, HETERO, AUTO, BATCH | Yes | |
  | IR, ONNX, PDPD, TF, TF Lite frontends | Yes | |

* Static build support means building static libraries only for OpenVINO Runtime libraries. Third-party prebuilt dependencies keep their original format — in particular `TBB` stays a shared library: it is not rebuilt as part of a static OpenVINO build, and the prebuilt package is copied to `<install_root>/runtime/3rdparty/tbb/bin`, so it must be available to your application at run time (on `PATH` on Windows, or `LD_LIBRARY_PATH` on Linux).

  > **NOTE**: The prebuilt TBB package can also be downloaded directly:
  > * Linux x64: `https://storage.openvinotoolkit.org/dependencies/thirdparty/linux/oneapi-tbb-2021.13.1-lin-release.tgz`
  > * Windows x64: `https://storage.openvinotoolkit.org/dependencies/thirdparty/windows/oneapi-tbb-2021.13.3-vs2022-win.zip`
  >
  > The TBB version differs per OS and changes between OpenVINO releases; check [cmake/dependencies.cmake](../../cmake/dependencies.cmake) for the version matching your checkout before downloading.

### Using a custom TBB build

To build against your own TBB instead of the prebuilt package, follow the [oneTBB installation instructions](https://github.com/uxlfoundation/oneTBB/blob/master/INSTALL.md) and set `TBBROOT` to the installation directory before configuring OpenVINO.

If you build that oneTBB as a static library, read [Static Linking of oneTBB](https://uxlfoundation.github.io/oneTBB/main/intro/static_linking.html) first. It is not a recommended configuration: it is safe only if exactly one copy of oneTBB ends up in the process, and it disables features that need run-time dynamic loading (`tbbbind` topology constraints, `tbbmalloc_proxy`, TCM).

> **NOTE**: You may notice the `ENABLE_TBBBIND_2_5` option is `OFF` on Windows x64 in a static build. This is harmless — the option only matters for old TBB packages that lack a dynamic `tbbbind_2_5` library. The prebuilt oneTBB ships one, so NUMA and hybrid-core detection work regardless.

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [How to Build OpenVINO](build.md)
