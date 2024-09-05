# CMake Options for Custom Compilation

This document provides description and default values for CMake options that can be used to build a custom OpenVINO runtime using the open source version. To understand all the dependencies when creating a custom runtime from the open source repository, refer to the [OpenVINO Runtime Introduction].

## Table of contents:

* [Disable / enable plugins build and other components](#disable--enable-plugins-build-and-other-components)
* [Options affecting binary size](#options-affecting-binary-size)
* [Building with custom OpenCV](#building-with-custom-opencv)
* [Building with custom TBB](#building-with-custom-tbb)
* [Test capabilities](#test-capabilities)
* [Other options](#other-options)
- [Additional Resources](#additional-resources)

## Disable / enable plugins build and other components

* Inference plugins:
    * `ENABLE_INTEL_CPU` enables CPU plugin compilation:
        * `ON` is default for x86 platforms; `OFF`, otherwise.
    * `ENABLE_INTEL_GPU` enables Intel GPU plugin compilation:
        * `ON` is default for x86 platforms; not available, otherwise.
    * `ENABLE_INTEL_NPU` enables Intel NPU plugin compilation:
        * `ON` is default for Windows and Linux x86 platforms; not available, otherwise.
    * `ENABLE_HETERO` enables HETERO plugin build:
        * `ON` is default.
    * `ENABLE_MULTI` enables MULTI plugin build:
        * `ON` is default.
    * `ENABLE_AUTO` enables AUTO plugin build:
        * `ON` is default.
    * `ENABLE_TEMPLATE` enables TEMPLATE plugin build:
        * `ON` is default.
    * `ENABLE_AUTO_BATCH` enables Auto Batch plugin build:
        * `ON` is default.
    * `ENABLE_PROXY` enables Proxy plugin compilation:
        * `ON` is default.
* Frontends to work with models from frameworks:
    * `ENABLE_OV_ONNX_FRONTEND` enables [ONNX] frontend plugin for OpenVINO Runtime:
        * `ON` is default.
    * `ENABLE_OV_PADDLE_FRONTEND` enables [PDPD] frontend plugin for OpenVINO Runtime:
        * `ON` is default.
    * `ENABLE_OV_TF_FRONTEND` enables [TensorFlow] frontend plugin for OpenVINO Runtime:
        * `ON` is default.
    * `ENABLE_OV_TF_LITE_FRONTEND` enables [TensorFlow Lite] frontend plugin for OpenVINO Runtime:
        * `ON` is default.
    * `ENABLE_OV_PYTORCH_FRONTEND` enables [PyTorch] frontend plugin for OpenVINO Runtime:
        * `ON` is default.
    * `ENABLE_OV_JAX_FRONTEND` enables [JAX] frontend plugin for OpenVINO Runtime:
        * `ON` is default.
    * `ENABLE_OV_IR_FRONTEND` enables OpenVINO Intermediate Representation frontend plugin for OpenVINO Runtime:
        * `ON` is default.
* `OPENVINO_EXTRA_MODULES` specifies path to add extra OpenVINO modules to the build.
    * See [OpenVINO Contrib] to add extra modules from.
* `ENABLE_SAMPLES` enables OpenVINO Runtime samples build:
    * `ON` is default.
* `ENABLE_PYTHON` enables [Python] API build:
    * `ON` if python requirements are satisfied (auto-discovered by CMake).
* `ENABLE_WHEEL` enables [Python] OpenVINO Runtime and Development wheels build:
    * `ON` if requirements are satisfied (auto-discovered by CMake).
* `ENABLE_TESTS` enables tests compilation:
    * `OFF` is default.
* `ENABLE_DOCS` enables building the OpenVINO documentation:
    * `OFF` is on Debian (Ubuntu) OSes
    * `OFF` is in other cases.
* `ENABLE_SYSTEM_PUGIXML` builds with system version of [pugixml] if it is available on the system.
    * `ON` is default.
    * [OpenVINO thirdparty pugixml] is used by default.
* `ENABLE_SYSTEM_PROTOBUF` use [protobuf] installed on the system (used by ONNX, PaddlePaddle and TensorFlow frontends):
    * `OFF` is default.
* `ENABLE_SYSTEM_FLATBUFFERS` use [FlatBuffers] installed on the system (used by Tensorflow Lite frontend):
    * `ON` is default.
* `ENABLE_SYSTEM_TBB` use TBB installed on the system:
    * `ON` is on Debian (Ubuntu) OSes.
    * `OFF` is in other cases.
* `ENABLE_SYSTEM_OPENCL` use OpenCL installed on the system:
    * `ON` is default.

## Options affecting binary size

* `ENABLE_LTO` boolean option to enable [Link Time Optimizations]:
    * `OFF` is default, because it takes longer time to link binaries.
    * `ON` is enabled for OpenVINO release binaries.
    * Available on Unix* compilers only like GCC or CLANG.
    * Gives 30% decrease in binary size together with other optimization options used to build OpenVINO.
* `THREADING` points to the OpenVINO threading interface:
    * `TBB` is the default option, which enables build with [Intel TBB] and `tbb::static_partitioner`.
    * `TBB_AUTO` enables building with [Intel TBB].
    * `OMP` enables building with Intel OpenMP.
    * `SEQ` disables threading optimizations. Can be used in cases when TBB binaries are absent.
    * **Note:** because TBB is a template library, it increases binary size because of multiple instantiations of `tbb::parallel_for`
* `ENABLE_TBBBIND_2_5` enables prebuilt static TBBBind 2.5 usage:
    * `ON` is default, because OpenVINO Runtime should be generic out of box.

> **Note:** TBBBind 2.5 is needed when OpenVINO **inference** targets CPUs with:
> * NUMA support (Non-Unified Memory Architecture), e.g. to detect a number of NUMA nodes
> * Hybrid architecture to separate Performance / Efficiency cores and schedule tasks in the optimal way.

> **Note:** if you build OpenVINO runtime with [oneTBB] support where TBBBind 2.5 is automatically loaded by TBB in runtime, then set `ENABLE_TBBBIND_2_5` to `OFF`
> * make sure libtbbbind.so is in the same folder as libtbb.so. For example, oneTBB package on Ubuntu 22.04 has libtbbbind missed. https://bugs.launchpad.net/ubuntu/+source/onetbb/+bug/2006898
> * oneTBB relies on higher version hwloc to recognize hybrid CPU core types correctly, on some machines, they require higher hwloc version to work correctly. Check if hwloc-info --version returns hwloc version >= 2.7.0, Ubuntu 20.04 with hwloc 2.1.0

* `ENABLE_SSE42` enables SSE4.2 optimizations:
    * `ON` is default for x86 platforms; not available for other platforms.
    * Affects only OpenVINO Runtime common part and preprocessing plugin, **does not affect the oneDNN library**
* `ENABLE_AVX2` enables AVX2 optimizations:
    * `ON` is default for x86 platforms, not available for other platforms.
    * Affects only OpenVINO Runtime common part and preprocessing plugin, **does not affect the oneDNN library**
* `ENABLE_AVX512F` enables AVX512 optimizations:
    * `ON` is default for x86 platforms, not available for other platforms.
    * Affects only OpenVINO Runtime common part and preprocessing plugin, **does not affect the oneDNN library**
* `ENABLE_PROFILING_ITT` enables profiling with [Intel ITT and VTune].
    * `OFF` is default, because it increases binary size.
* `SELECTIVE_BUILD` enables [[Conditional compilation|ConditionalCompilation]] feature.
    * `OFF` is default.
* `ENABLE_MLAS_FOR_CPU` enables MLAS library for CPU plugin
    * `ON` is default for x86_64 and AARCH64 platforms
    * Affects only OpenVINO CPU plugin

## Building with OpenCV

Some OpenVINO samples can benefit from OpenCV usage, e.g. can read more image formats as inputs. If you have OpenCV on your machine, you can pass it via CMake option:

```sh
cmake -DOpenCV_DIR=<path to OpenCVConfig.cmake> ...
```

## Building with custom TBB

When OpenVINO CMake scripts are run with TBB enabled (`-DTHREADING=TBB` which is default, or `-DTHREADING=TBB_AUTO`), CMake OpenVINO scripts automatically download prebuilt version of TBB which is ABI-compatible with the default compiler of your system. If you have a non-default compiler or want to use custom TBB, you can use:

```sh
export TBBROOT=<path to TBB install root>
cmake ...
```
In this case OpenVINO CMake scripts take `TBBROOT` environment variable into account and provided TBB will be used.

**Note:** if you are building TBB from source files, please install TBB after and use `TBBROOT` to point to installation root.
**Note:** reference to oneTBB Note in [Options affecting binary size](#options-affecting-binary-size)

## Test capabilities

* `ENABLE_SANITIZER` builds with clang [address sanitizer] support:
    * `OFF` is default.
* `ENABLE_THREAD_SANITIZER` builds with clang [thread-sanitizer] support:
    * `OFF` is default.
* `ENABLE_COVERAGE` adds option to enable coverage. See dedicated guide [[how to measure test coverage|InferenceEngineTestsCoverage]]:
    * `OFF` is default.
* `ENABLE_FUZZING` enables instrumentation of code for fuzzing:
    * `OFF` is default.

## Other options

* `ENABLE_CPPLINT` enables code style check using [cpplint] static code checker:
    * `ON` is default.
* `ENABLE_CLANG_FORMAT` enables [Clang format] code style check:
    * `ON` is default.
* `ENABLE_FASTER_BUILD` enables [precompiled headers] and [unity build] using CMake:
    * `OFF` is default.
* `ENABLE_INTEGRITYCHECK` builds DLLs with [/INTEGRITYCHECK] flag:
    * `OFF` is default.
    * Available on MSVC compiler only.
* `ENABLE_QSPECTRE` builds with [/Qspectre] flag:
    * `OFF` is default.
    * Available on MSVC compiler only.

## Additional Resources

 * [OpenVINO README](../../README.md)
 * [OpenVINO Developer Documentation](index.md)
 * [How to build OpenVINO](build.md)

[Link Time Optimizations]:https://llvm.org/docs/LinkTimeOptimization.html
[thread-sanitizer]:https://clang.llvm.org/docs/ThreadSanitizer.html
[address sanitizer]:https://clang.llvm.org/docs/AddressSanitizer.html
[Intel ITT and VTune]:https://software.intel.com/content/www/us/en/develop/documentation/vtune-help/top/api-support/instrumentation-and-tracing-technology-apis.html
[precompiled headers]:https://cmake.org/cmake/help/git-stage/command/target_precompile_headers.html
[unity build]:https://cmake.org/cmake/help/latest/prop_tgt/UNITY_BUILD.html
[/INTEGRITYCHECK]:https://docs.microsoft.com/en-us/cpp/build/reference/integritycheck-require-signature-check?view=msvc-160
[/Qspectre]:https://learn.microsoft.com/en-us/cpp/build/reference/qspectre?view=msvc-170
[Intel TBB]:https://software.intel.com/content/www/us/en/develop/tools/threading-building-blocks.html
[Python]:https://www.python.org/
[Java]:https://www.java.com/ru/
[cpplint]:https://github.com/cpplint/cpplint
[Clang format]:http://clang.llvm.org/docs/ClangFormat.html
[OpenVINO Contrib]:https://github.com/openvinotoolkit/openvino_contrib
[OpenVINO thirdparty pugixml]:https://github.com/openvinotoolkit/openvino/tree/master/inference-engine/thirdparty/pugixml
[pugixml]:https://pugixml.org/
[ONNX]:https://onnx.ai/
[protobuf]:https://github.com/protocolbuffers/protobuf
[OpenVINO Runtime Introduction]:https://docs.openvino.ai/2024/openvino-workflow/running-inference/integrate-openvino-with-your-application.html
[PDPD]:https://github.com/PaddlePaddle/Paddle
[TensorFlow]:https://www.tensorflow.org/
[TensorFlow Lite]:https://www.tensorflow.org/lite
[PyTorch]:https://pytorch.org/
[FlatBuffers]:https://google.github.io/flatbuffers/
[oneTBB]:https://github.com/oneapi-src/oneTBB
[JAX]:https://github.com/google/jax
