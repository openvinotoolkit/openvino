# OpenVINO Intel GPU Plugin

GPU plugin in [OpenVINO toolkit](https://github.com/openvinotoolkit/openvino) supports inference on Intel® GPUs starting from Gen8 architecture.

## Key Contacts

For assistance regarding GPU, contact a member of [openvino-ie-gpu-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-gpu-maintainers) group.

## Components

GPU Plugin contains the following components:

* [docs](./docs/) - developer documentation pages for the component.
* [include](./include/) - public API.
* [src](./src/) - sources of the component.
* [tests](./tests/) - tests for OpenVINO Plugin component.
* [thirdparty](./thirdparty/) - third-party modules.

## Tutorials

* [Simplified workflow](./docs/simplified_workflow.md)
  * [Graph Optimization Passes](./docs/graph_optimization_passes.md)
  * [Execution of Inference](./docs/execution_of_inference.md)
* [Source code structure](./docs/source_code_structure.md)
  * [Basic data structures of gpu graph and overall flow](./docs/basic_data_structures.md)
  * [Memory allocation in GPU plugin](./docs/memory_allocation_gpu_plugin.md)
* [Memory formats](./docs/gpu_memory_formats.md)
* [Kernels and kernel selectors](./docs/gpu_kernels.md)
* [GPU plugin operations enabling flow](./docs/gpu_plugin_ops_enabling.md)
* [Debug utils](./docs/gpu_debug_utils.md)
* [OpenCL Runtime issues troubleshooting](./docs/gpu_plugin_driver_troubleshooting.md)
* [GPU plugin unit test](./docs/gpu_plugin_unit_test.md)
* [Run benchmark from device_mem](./docs/use_device_mem.md)

## Documentation on dynamic-shape
This contents explain the internal implementation of dynamic shape support in the GPU Plugin. For general usage of dynamic shape and limitations of the GPU plugin, please refer to this link: [GPU Device — OpenVINO™ documentation - Version(2023.1)](https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_supported_plugins_GPU.html#dynamic-shapes).

* [Overall flow for dynamic shape execution](./docs/dynamic_shape/overall_flow.md)
* Implementation details
  * [Preprocessing: Shape inference / update weight / realloc memory](./docs/dynamic_shape/preprocessing.md)
  * [dynamic impl of kernels](./docs/dynamic_shape/dynamic_impl.md)
  * [in-memory kernel cache](./docs/dynamic_shape/in_memory_cache.md)
  * [async kernel compilation](./docs/dynamic_shape/async_compilation.md)
  <!-- * weight compression (TBD)) -->
* Optimization features
  * [Memory preallocation](./docs/dynamic_shape/memory_preallocation.md)
<!--  * Fake alignment of shape (TBD)
  * Shape-of subgraph on CPU (TBD)
  * Runtime buffer fusing (TBD)
  * Runtime reorder skip (TBD)
  * KV cache (TBD)
* Performance analysis and debugging features (TBD)
* Model caching for dynamic shape (TBD)
-->

## Attached licenses

GPU plugin uses 3<sup>rd</sup>-party components licensed under following licenses:
- *googletest* under [Google License](https://github.com/google/googletest/blob/master/googletest/LICENSE)
- *OpenCL™ ICD and C++ Wrapper under [Khronos™ License](https://github.com/KhronosGroup/OpenCL-CLHPP/blob/master/LICENSE.txt)
- *RapidJSON* under [Tencent License](https://github.com/Tencent/rapidjson/blob/master/license.txt)

## Support

To report issues and make suggestions, see [GitHub issues](https://github.com/openvinotoolkit/openvino/issues).

## How to Contribute

Community contributions to GPU plugin are highly welcome. If you have a suggestion on how to improve the library:

- Share your proposal via
 [GitHub issues](https://github.com/openvinotoolkit/openvino/issues)
- Ensure you can build the product and run all the tests with your patch
- In case of a larger feature, create a test
- Submit a [pull request](https://github.com/openvinotoolkit/openvino/pulls)

We will review your contribution and, if any additional fixes or modifications
are necessary, we may provide feedback to guide you. Once your pull request 
has been approved, it will be merged into our GitHub repository.

## System Requirements

GPU plugin supports Intel® HD Graphics, Intel® Iris® Graphics and Intel® Arc™ Graphics and is optimized for Gen9-Gen12LP, Gen12HP architectures

GPU plugin currently uses OpenCL™ with multiple Intel OpenCL™ extensions and requires Intel® Graphics Driver to run.

GPU plugin requires CPU with Intel® SSE/Intel® AVX support.

---

The software dependencies are:
- [CMake](https://cmake.org/download/) 3.5 or later
- C++ compiler with C++11 standard support compatible with:
    * GNU Compiler Collection 4.8 or later
    * clang 3.5 or later
    * [Intel® C++ Compiler](https://software.intel.com/en-us/intel-parallel-studio-xe) 17.0 or later
    * Visual C++ 2015 (MSVC++ 19.0) or later
- [python™](https://www.python.org/downloads/) 3.8 or later.

## Trademark Information

Intel, the Intel logo, Intel Atom, Intel Core, Intel Xeon Phi, Iris, OpenVINO,
the OpenVINO logo, Pentium, VTune, and Xeon are trademarks
of Intel Corporation or its subsidiaries.

\* Other names and brands may be claimed as the property of others.

Microsoft, Windows, and the Windows logo are trademarks, or registered
trademarks of Microsoft Corporation in the United States and/or other
countries.

OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission
by Khronos.

Copyright © 2023, Intel Corporation

## See also

 * [OpenVINO™ README](../../../README.md)
 * [OpenVINO Core Components](../../README.md)
 * [OpenVINO Plugins](../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
