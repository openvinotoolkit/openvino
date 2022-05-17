oneAPI Deep Neural Network Library (oneDNN)
===========================================

> This software was previously known as
> **Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)**
> and **Deep Neural Network Library (DNNL)**.

<img align="left" src="https://spec.oneapi.io/oneapi-logo-white-scaled.jpg" alt="oneAPI logo">

oneAPI Deep Neural Network Library (oneDNN) is an open-source cross-platform
performance library of basic building blocks for deep learning applications.
oneDNN is part of [oneAPI](https://oneapi.io).
The library is optimized for Intel(R) Architecture Processors, Intel Processor
Graphics and Xe Architecture graphics. oneDNN has experimental support
for the following architectures: Arm\* 64-bit Architecture (AArch64),
NVIDIA\* GPU, OpenPOWER\* Power ISA (PPC64), IBMz\* (s390x), and RISC-V.

oneDNN is intended for deep learning applications and framework
developers interested in improving application performance
on Intel CPUs and GPUs. Deep learning practitioners should use one of the
[applications enabled with oneDNN](#applications-enabled-with-onednn).

# Table of Contents

- [Documentation](#documentation)
- [Installation](#installation)
- [System Requirements](#system-requirements)
- [Applications Enabled with oneDNN](#applications-enabled-with-onednn)
- [Support](#support)
- [Contributing](#contributing)
- [License](#license)
- [Security](#security)
- [Trademark Information](#trademark-information)

# Documentation

* [Developer guide](https://oneapi-src.github.io/oneDNN) explains programming
  model, supported functionality, and implementation details, and
  includes annotated examples.
* [API reference](https://oneapi-src.github.io/oneDNN/group_dnnl_api.html) provides
  a comprehensive reference of the library API.

# Installation

Binary distribution of this software is available in:
* [Anaconda](https://anaconda.org/conda-forge/onednn)
* [Intel oneAPI](https://software.intel.com/en-us/oneapi/onednn)

The packages do not include library dependencies and these need to be resolved
in the application at build time. See the
[System Requirements](#system-requirements) section below and the
[Build Options](https://oneapi-src.github.io/oneDNN/dev_guide_build_options.html)
section in the [developer guide](https://oneapi-src.github.io/oneDNN) for more
details on CPU and GPU runtimes.

If the configuration you need is not available, you can
[build the library from source](https://oneapi-src.github.io/oneDNN/dev_guide_build.html).

# System Requirements

oneDNN supports platforms based on the following architectures:
- [Intel 64 or AMD64](https://en.wikipedia.org/wiki/X86-64),
- [Arm 64-bit Architecture (AArch64)](https://developer.arm.com/architectures/cpu-architecture/a-profile).
- [OpenPOWER](https://openpowerfoundation.org/) / [IBM Power ISA](https://en.wikipedia.org/wiki/Power_ISA).
- [IBMz z/Architecture (s390x)](https://en.wikipedia.org/wiki/Z/Architecture).

> **WARNING**
>
> Arm 64-bit Architecture (AArch64), Power ISA (PPC64) and IBMz (s390x) support
> is **experimental** with limited testing validation.

The library is optimized for the following CPUs:
* Intel Atom(R) processors (at least Intel SSE4.1 support is required)
* Intel Core(TM) processors (at least Intel SSE4.1 support is required)
* Intel Xeon(R) processor E3, E5, and E7 family (formerly Sandy Bridge,
  Ivy Bridge, Haswell, and Broadwell)
* Intel Xeon Phi(TM) processor (formerly Knights Landing and Knights Mill)
* Intel Xeon Scalable processor (formerly Skylake, Cascade Lake, and Cooper
  Lake)
* future Intel Xeon Scalable processor (code name Sapphire Rapids)

On a CPU based on Intel 64 or on AMD64 architecture, oneDNN detects
the instruction set architecture (ISA) at runtime and uses just-in-time (JIT)
code generation to deploy the code optimized for the latest supported ISA.
Future ISAs may have initial support in the library disabled by default and
require the use of run-time controls to enable them. See
[CPU dispatcher control](https://oneapi-src.github.io/oneDNN/dev_guide_cpu_dispatcher_control.html)
for more details.

On a CPU based on Arm AArch64 architecture, oneDNN can be built with Arm Compute Library
integration. Compute Library is an open-source library for machine learning applications
and provides AArch64 optimized implementations of core functions. This functionality currently
requires that Compute Library is downloaded and built separately, see
[Build from Source](https://oneapi-src.github.io/oneDNN/dev_guide_build.html). oneDNN is only
compatible with Compute Library versions 21.08 or later.

> **WARNING**
>
> On macOS, applications that use oneDNN may need to request special
> entitlements if they use the hardened runtime. See the
> [linking guide](https://oneapi-src.github.io/oneDNN/dev_guide_link.html)
> for more details.

The library is optimized for the following GPUs:
* Intel Processor Graphics based on Gen9, Gen9.5 and Gen11, and Gen12 architectures
* Intel Iris(R) Xe graphics (formerly DG1)
* future Intel Arc(TM) graphics (code name Alchemist and DG2)

## Requirements for Building from Source

oneDNN supports systems meeting the following requirements:
* Operating system with Intel 64 / Arm 64 / Power / IBMz architecture support
* C++ compiler with C++11 standard support
* [CMake](https://cmake.org/download/) 2.8.12 or later
* [Arm Compute Library](https://github.com/arm-software/ComputeLibrary)
  for builds using Compute Library on AArch64.

The following tools are required to build oneDNN documentation:
* [Doxygen](http://www.doxygen.nl/download.html#srcbin) 1.8.5 or later
* [Doxyrest](https://github.com/vovkos/doxyrest) 2.1.2 or later
* [Sphinx](https://www.sphinx-doc.org/en/master/usage/installation.html) 4.0.2 or later
* [sphinx-book-theme](https://sphinx-book-theme.readthedocs.io/en/latest) 0.0.41 or later

Configurations of CPU and GPU engines may introduce additional build time
dependencies.

### CPU Engine

oneDNN CPU engine is used to execute primitives on Intel Architecture
Processors, 64-bit Arm Architecture (AArch64) processors,
64-bit Power ISA (PPC64) processors, IBMz (s390x), and compatible devices.

The CPU engine is built by default but can be disabled at build time by setting
`DNNL_CPU_RUNTIME` to `NONE`. In this case, GPU engine must be enabled.
The CPU engine can be configured to use the OpenMP, TBB or DPCPP runtime.
The following additional requirements apply:
* OpenMP runtime requires C++ compiler with OpenMP 2.0 or later
  standard support
* TBB runtime requires
[Threading Building Blocks (TBB)](https://www.threadingbuildingblocks.org/)
2017 or later.
* DPCPP runtime requires
  * [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler)
  * [Threading Building Blocks (TBB)](https://www.threadingbuildingblocks.org/)

Some implementations rely on OpenMP 4.0 SIMD extensions. For the best
performance results on Intel Architecture Processors we recommend using the
Intel C++ Compiler.

### GPU Engine

Intel Processor Graphics and Xe Architecture graphics are supported by
the oneDNN GPU engine. The GPU engine is disabled in the default build
configuration. The following additional requirements apply when GPU engine
is enabled:
* OpenCL runtime requires
    * OpenCL\* runtime library (OpenCL version 1.2 or later)
    * OpenCL driver (with kernel language support for OpenCL C 2.0 or later)
      with Intel subgroups and USM extensions support
* DPCPP runtime requires
    * [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler)
    * OpenCL runtime library (OpenCL version 1.2 or later)
    * [oneAPI Level Zero](https://github.com/oneapi-src/level-zero)
* DPCPP runtime with NVIDIA GPU support requires
    * [oneAPI DPC++ Compiler](https://github.com/intel/llvm)
    * OpenCL runtime library (OpenCL version 1.2 or later)
    * NVIDIA CUDA\* driver
    * cuBLAS 10.1 or later
    * cuDNN 7.6 or later

> **WARNING**
>
> NVIDIA GPU support is experimental. General information, build instructions
> and implementation limitations is available in
> [NVIDIA backend readme](https://github.com/oneapi-src/oneDNN/blob/master/src/gpu/nvidia/README.md).

### Runtime Dependencies

When oneDNN is built from source, the library runtime dependencies
and specific versions are defined by the build environment.

#### Linux

Common dependencies:
* GNU C Library (`libc.so`)
* GNU Standard C++ Library v3 (`libstdc++.so`)
* Dynamic Linking Library (`libdl.so`)
* C Math Library (`libm.so`)
* POSIX Threads Library (`libpthread.so`)

Runtime-specific dependencies:

| Runtime configuration    | Compiler                      | Dependency
| :----------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP`   | GCC                           | GNU OpenMP runtime (`libgomp.so`)
| `DNNL_CPU_RUNTIME=OMP`   | Intel C/C++ Compiler          | Intel OpenMP runtime (`libiomp5.so`)
| `DNNL_CPU_RUNTIME=OMP`   | Clang                         | Intel OpenMP runtime (`libiomp5.so`)
| `DNNL_CPU_RUNTIME=TBB`   | any                           | TBB (`libtbb.so`)
| `DNNL_CPU_RUNTIME=DPCPP` | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`libsycl.so`), TBB (`libtbb.so`), OpenCL loader (`libOpenCL.so`)
| `DNNL_GPU_RUNTIME=OCL`   | any                           | OpenCL loader (`libOpenCL.so`)
| `DNNL_GPU_RUNTIME=DPCPP` | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`libsycl.so`), OpenCL loader (`libOpenCL.so`), oneAPI Level Zero loader (`libze_loader.so`)

#### Windows

Common dependencies:
* Microsoft Visual C++ Redistributable (`msvcrt.dll`)

Runtime-specific dependencies:

| Runtime configuration    | Compiler                      | Dependency
| :----------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP`   | Microsoft Visual C++ Compiler | No additional requirements
| `DNNL_CPU_RUNTIME=OMP`   | Intel C/C++ Compiler          | Intel OpenMP runtime (`iomp5.dll`)
| `DNNL_CPU_RUNTIME=TBB`   | any                           | TBB (`tbb.dll`)
| `DNNL_CPU_RUNTIME=DPCPP` | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`sycl.dll`), TBB (`tbb.dll`), OpenCL loader (`OpenCL.dll`)
| `DNNL_GPU_RUNTIME=OCL`   | any                           | OpenCL loader (`OpenCL.dll`)
| `DNNL_GPU_RUNTIME=DPCPP` | Intel oneAPI DPC++ Compiler   | Intel oneAPI DPC++ Compiler runtime (`sycl.dll`), OpenCL loader (`OpenCL.dll`), oneAPI Level Zero loader (`ze_loader.dll`)

#### macOS

Common dependencies:
* System C/C++ runtime (`libc++.dylib`, `libSystem.dylib`)

Runtime-specific dependencies:

| Runtime configuration  | Compiler                      | Dependency
| :--------------------- | :---------------------------- | :---------
| `DNNL_CPU_RUNTIME=OMP` | Intel C/C++ Compiler          | Intel OpenMP runtime (`libiomp5.dylib`)
| `DNNL_CPU_RUNTIME=TBB` | any                           | TBB (`libtbb.dylib`)

### Validated Configurations

CPU engine was validated on RedHat\* Enterprise Linux 7 with
* GNU Compiler Collection 4.8, 5.4, 6.1, 7.2, 8.1, and 9.1
* Clang\* 3.8.1, 7.1, 8.0, and 9.0
* [Intel C/C++ Compiler](https://software.intel.com/content/www/us/en/develop/tools/parallel-studio-xe.html)
  19.1
* [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) 2021.1

on Windows Server\* 2016 with
* Microsoft Visual Studio 2015, 2017, and 2019
* [Intel C/C++ Compiler](https://software.intel.com/content/www/us/en/develop/tools/parallel-studio-xe.html)
  19.1
* [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) 2021.1

on macOS 10.13 (High Sierra) with
* Apple LLVM version 9.1
* [Intel C/C++ Compiler](https://software.intel.com/content/www/us/en/develop/tools/parallel-studio-xe.html)
  19.1

GPU engine was validated on Ubuntu\* 20.04 with
* GNU Compiler Collection 7.2, 8.1, and 9.1
* Clang 3.8.1, 7.1, 8.0, and 9.0
* [Intel C/C++ Compiler](https://software.intel.com/content/www/us/en/develop/tools/parallel-studio-xe.html)
  19.1
* [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) 2021.1
* [Intel Software for General Purpose GPU capabilities](https://dgpu-docs.intel.com/index.html)
latest stable version available at the time of release

on Windows Server 2019 with
* Microsoft Visual Studio 2015, 2017, and 2019
* [Intel C/C++ Compiler](https://software.intel.com/content/www/us/en/develop/tools/parallel-studio-xe.html)
  19.1
* [Intel oneAPI DPC++ Compiler](https://software.intel.com/en-us/oneapi/dpc-compiler) 2021.1
* [Intel Graphics - Windows 10 DCH Drivers](https://downloadcenter.intel.com/download/29808/Intel-Graphics-Windows-10-DCH-Drivers)
latest stable version available at the time of release

## Requirements for Pre-built Binaries

See the README included in the corresponding binary package.

# Applications Enabled with oneDNN

* [Apache\* MXNet](https://mxnet.apache.org)
* [Apache\* SINGA](https://singa.apache.org)
* [DeepLearning4J\*](https://deeplearning4j.org)
* [Flashlight\*](https://github.com/facebookresearch/flashlight)
* [Korali](https://github.com/cselab/korali)
* [MATLAB\* Deep Learning Toolbox](https://www.mathworks.com/help/deeplearning/)
* [ONNX Runtime](https://github.com/microsoft/onnxruntime)
* [OpenVINO(TM) toolkit](https://01.org/openvinotoolkit)
* [PaddlePaddle\*](http://www.paddlepaddle.org)
* [PyTorch\*](https://pytorch.org/)
* [Tensorflow\*](https://www.tensorflow.org)

# Support

Please submit your questions, feature requests, and bug reports on the
[GitHub issues](https://github.com/oneapi-src/oneDNN/issues) page.

You may reach out to project maintainers privately
at dnnl.maintainers@intel.com.

> **WARNING**
>
> This is pre-production software and functionality may change without prior
> notice.

# Contributing

We welcome community contributions to oneDNN. If you have an idea on how
to improve the library:

* For changes impacting the public API or library overall, such as adding new
  primitives or changes to the architecture, submit an
  [RFC pull request](https://github.com/oneapi-src/oneDNN/tree/rfcs).
* Ensure that the changes are consistent with the
  [code contribution guidelines](CONTRIBUTING.md#code-contribution-guidelines)
  and [coding standards](CONTRIBUTING.md#coding-standards).
* Ensure that you can build the product and run all the examples with your
  patch.
* Submit a [pull request](https://github.com/oneapi-src/oneDNN/pulls).

For additional details, see [contribution guidelines](CONTRIBUTING.md).

This project is intended to be a safe, welcoming space for collaboration, and
contributors are expected to adhere to the
[Contributor Covenant](CODE_OF_CONDUCT.md) code of conduct.

# License

oneDNN is licensed under [Apache License Version 2.0](LICENSE). Refer to the
"[LICENSE](LICENSE)" file for the full license text and copyright notice.

This distribution includes third party software governed by separate license
terms.

3-clause BSD license:
* [Xbyak](https://github.com/herumi/xbyak)
* [gtest](https://github.com/google/googletest)
* [Instrumentation and Tracing Technology API (ITT API)](https://github.com/intel/IntelSEAPI/tree/master/ittnotify)
* [CMake](https://github.com/Kitware/CMake)

2-clause BSD license:
* [Sphinx](https://www.sphinx-doc.org/)

Apache License Version 2.0:
* [Xbyak_aarch64](https://github.com/fujitsu/xbyak_aarch64)

Boost Software License, Version 1.0:
* [Boost C++ Libraries](https://www.boost.org/)

MIT License:
* [Intel Graphics Compute Runtime for oneAPI Level Zero and OpenCL Driver](https://github.com/intel/compute-runtime)
* [Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler)
* [Doxyrest](https://github.com/vovkos/doxyrest)

This third party software, even if included with the distribution of
the Intel software, may be governed by separate license terms, including
without limitation, third party license terms, other Intel software license
terms, and open source software license terms. These separate license terms
govern your use of the third party programs as set forth in the
"[THIRD-PARTY-PROGRAMS](THIRD-PARTY-PROGRAMS)" file.

# Security

See Intel's [Security Center](https://www.intel.com/content/www/us/en/security-center/default.html)
for information on how to report a potential security issue or vulnerability.

See also: [Security Policy](SECURITY.md)

# Trademark Information

Intel, the Intel logo, Arc, Intel Atom, Intel Core, Intel Xeon Phi, Iris,
OpenVINO, the OpenVINO logo, Pentium, VTune, and Xeon are trademarks
of Intel Corporation or its subsidiaries.

\* Other names and brands may be claimed as the property of others.

Microsoft, Windows, and the Windows logo are trademarks, or registered
trademarks of Microsoft Corporation in the United States and/or other
countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission
by Khronos.

(C) Intel Corporation

