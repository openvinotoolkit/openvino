# Building OpenVINO static libraries

## Contents

- [Introduction](#introduction)
- [System requirements](#system-requirements)
- [Configure OpenVINO runtime in CMake stage](#configure-openvino-runtime-in-cmake-stage)
- [Build static OpenVINO libraries](#build-static-openvino-libraries)
- [Link static OpenVINO runtime](#link-static-openvino-runtime)
- [Static OpenVINO libraries + Conditional compilation for particular models](#static-openvino-libraries--conditional-compilation-for-particular-models)
- [Building with static MSVC Runtime](#building-with-static-msvc-runtime)
- [Limitations](#limitations)
- [See also](#see-also)

## Introduction

Building static OpenVINO Runtime libraries allows to additionally reduce the size of a binary when it is used together with conditional compilation.
It is possible because not all interface symbols of OpenVINO Runtime libraries are exported to end users during a static build and can be removed by linker. See [Static OpenVINO libraries + Conditional compilation for particular models](#static-openvino-libraries--conditional-compilation-for-particular-models)

## System requirements

* CMake version 3.18 or higher must be used to build static OpenVINO libraries.
* Supported OSes:
    * Windows x64
    * Linux x64
    * All other OSes may work, but have not been explicitly tested

## Configure OpenVINO Runtime in the CMake stage

The default architecture of OpenVINO Runtime assumes that the following components are subject to dynamic loading during execution:
* (Device) Inference backends (CPU, GPU, NPU, MULTI, HETERO, etc.)
* (Model) Frontends (IR, ONNX, PDPD, TF, JAX, etc.)

With the static OpenVINO Runtime, all these modules should be linked into a final user application and **the list of modules/configuration must be known for the CMake configuration stage**. To minimize the total binary size, you can explicitly turn `OFF` unnecessary components. Use [[CMake Options for Custom Compilation|CMakeOptionsForCustomCompilation ]] as a reference for OpenVINO CMake configuration.

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
-
## Build static OpenVINO libraries

To build OpenVINO Runtime in a static mode, you need to specify the additional CMake option:

```sh
cmake -DBUILD_SHARED_LIBS=OFF <all other CMake options> <openvino_sources root>
```

Then, use the usual CMake 'build' command:

```sh
cmake --build . --target openvino --config Release -j12
```

Then, the installation step:

```sh
cmake -DCMAKE_INSTALL_PREFIX=<install_root> -P cmake_install.cmake
```

The OpenVINO runtime is located in `<install_root>/runtime/lib`

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

OpenVINO Runtime can be compiled for particular models, as shown in the [[Conditional compilation for particular models|ConditionalCompilation]] guide.
The conditional compilation feature can be paired with static OpenVINO libraries to build even smaller end-user applications in terms of binary size. The following procedure can be used, (based on the detailed [[Conditional compilation for particular models|ConditionalCompilation]] guide):

* Build OpenVINO Runtime as usual with the CMake option of `-DSELECTIVE_BUILD=COLLECT`.
* Run target applications on target models and target platforms to collect traces.
* Build the final OpenVINO static Runtime with `-DSELECTIVE_BUILD=ON -DSELECTIVE_BUILD_STAT=/path/*.csv -DBUILD_SHARED_LIBS=OFF`

## Building with static MSVC Runtime

In order to build with static MSVC runtime, use the special [OpenVINO toolchain](https://github.com/openvinotoolkit/openvino/blob/master/cmake/toolchains/mt.runtime.win32.toolchain.cmake) file:

```sh
cmake -DCMAKE_TOOLCHAIN_FILE=<openvino source dir>/cmake/toolchains/mt.runtime.win32.toolchain.cmake <other options>
```

> **NOTE**: all other dependent application and libraries must be built with the same `mt.runtime.win32.toolchain.cmake ` toolchain to have conformed values of the `MSVC_RUNTIME_LIBRARY` target property.

## Limitations

* The enabled and tested capabilities of OpenVINO Runtime in a static build:
    * OpenVINO common runtime - work with `ov::Model`, perform model loading on particular device
    * MULTI, HETERO, AUTO, and BATCH inference modes
    * IR, ONNX, PDPD, TF and TF Lite frontends to read `ov::Model`
* Static build support for building static libraries only for OpenVINO Runtime libraries. All other third-party prebuilt dependencies remain in the same format:
    * `TBB` is a shared library; to provide your own TBB build from [[oneTBB source code|https://github.com/oneapi-src/oneTBB]] use `export TBBROOT=<tbb_root>` before OpenVINO CMake scripts are run.

    > **NOTE**: The TBB team does not recommend using oneTBB as a static library, see [[Why onetbb does not like a static library?|https://github.com/oneapi-src/oneTBB/issues/646]]

* `TBBBind_2_5` is not available on Windows x64 during a static OpenVINO build (see description for `ENABLE_TBBBIND_2_5` CMake option [[here|CMakeOptionsForCustomCompilation]] to understand what this library is responsible for). So, capabilities enabled by `TBBBind_2_5` are not available. To enable them, build [[oneTBB from source code|https://github.com/oneapi-src/oneTBB]] and provide the path to built oneTBB artifacts via `TBBROOT` environment variable before OpenVINO CMake scripts are run.

## See also

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [How to Build OpenVINO](build.md)
