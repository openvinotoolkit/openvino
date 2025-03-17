# GPU Plugin Structure

Historically, GPU plugin was built on top of standalone [clDNN library](https://github.com/intel/clDNN) for DNNs inference on Intel® GPUs,
but at some point clDNN became a part of OpenVINO, so now it's a part of overall GPU plugin code. Intel® Arc™ Graphics Xe-HPG is supported
via embedding of [oneDNN library](https://github.com/oneapi-src/oneDNN)

OpenVINO GPU plugin is responsible for:
 1. [IE Plugin API](https://docs.openvino.ai/2025/documentation/openvino-extensibility/openvino-plugin-library.html) implementation.
 2. Translation of a model from common IE semantic (`ov::Function`) into plugin-specific one (`cldnn::topology`), which is then compiled into
 GPU graph representation (`cldnn::network`).
 3. Implementation of OpenVINO operation set for Intel® GPU.
 4. Device-specific graph transformations.
 5. Memory allocation and management logic.
 6. Processing of incoming InferRequests, using clDNN objects.
 7. Actual execution on GPU device.

Intel GPU Plugin source code structure is shown below:
<pre>
src/plugins/intel_gpu                  - root GPU plugin folder
             ├── include
             │   ├── intel_gpu         - library internal headers
             │   │   ├── graph         - headers for internal graph representations
             │   │   ├── plugin        - definition of classes required for OpenVINO plugin API implementation
             │   │   ├── primitives    - primitive definitions for all supported operations
             │   │   └── runtime       - abstraction for execution runtime entities (memory, device, engine, etc)
             │   └── va
             ├── src
             │   ├── graph - all sources related to internal graph representation
             │   │    ├── graph_optimizer - passes for graph transformations
             │   │    ├── impls - definition of primitive implementations
             │   │    └── include - headers with graph nodes
             │   │
             │   ├── kernel_selector - OpenCL™ kernels (host+device parts) + utils for optimal kernels selection
             │   │   ├── common      - definition of some generic classes/structures used in kernel_selector
             │   │   └── core        - kernels, kernel selectors, and kernel parameters definitions
             │   │       ├── actual_kernels  - host side part of OpenCL™ kernels including applicability checks, performance heuristics and Local/Global work-groups description
             │   │       ├── cache  - cache.json - tuning cache of the kernels which is redistributed with the plugin to improve kernels and kernel parameters selection for better performance
             │   │       ├── cl_kernels - templates of GPU kernels (device part) written on OpenCL™
             │   │       └── common - utils for code generation and kernels selection
             │   ├── plugin - implementation of OpenVINO plugin API
             │   │    └── ops - factories for conversion of OpenVINO operations to internal primitives
             │   └── runtime
             │        └── ocl/ - implementation for OpenCL™ based runtime
             ├── tests
             │   ├── test_cases
             │   └── test_utils
             └── thirdparty
                 ├── onednn_gpu - <a href="https://github.com/oneapi-src/oneDNN">oneDNN</a> submodule which may be used to accelerate some primitives
                 └── rapidjson  - thirdparty <a href="https://github.com/Tencent/rapidjson">RapidJSON</a> lib for reading json files (cache.json)
</pre>

It is worth it to mention the functional tests, which are located in:
```
src/tests/functional/plugin/gpu
```
Most of the tests are reused across plugins, and each plugin only needs to add the test instances with some specific parameters.

Shared tests are located in:
```
src/tests/functional/plugin/shared                        <--- test definitions
src/tests/functional/plugin/gpu/shared_tests_instances    <--- instances for GPU plugin
```

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)