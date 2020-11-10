# GPU plugin (CLDNN based) structure {#openvino_docs_gpu_plugin_structure}

Historically GPU plugin was built on top of standalone [clDNN library](https://github.com/intel/clDNN) for DNNs inference on Intel® GPUs,
but at some point clDNN became a part of OpenVINO,
and the clDNN source code is a part of [openvino repository](https://github.com/openvinotoolkit/openvino/tree/master/inference-engine/thirdparty/clDNN).
Due to this reason source code of GPU plugin is located in 2 places:
 1. inference-engine/src/cldnn_engine - this part is usually called "clDNN Plugin"
 2. inference-engine/thirdparty/clDNN - this part is referred as "clDNN" or "cldnn"

clDNN Plugin is a relatively small module that is responsible for:
 1. [IE Plugin API](../Intro.md) implementation.
 2. Translation of model from common IE semantic (CNNNetwork) into clDNN specific one (cldnn::topology) which is then compiled into
 gpu graph representation (cldnn::network).
 3. Processing of incoming InferRequests using clDNN objects.

clDNN library itself is responsible for:
 1. Implementation of OpenVINO operation set for Intel® GPU.
 2. Device specific graph transformations.
 3. Memory allocation and management logic.
 4. Actual execution on GPU device.

As clDNN Plugin source code structure is relatively simple, let's more focus on the structure of clDNN:
```
- inferene-engine/thirdparty/clDNN - root clDNN folder
 - api/ - clDNN library API. Contains API headers for all supported primitives, clDNN graph representation, GPU context API and so on
 - api_extension/ - some internal primitives that are not supposed to be used by clDNN users (clDNN Plugin)
 - common/ - [Google Tests framework](https://github.com/google/googletest) and OpenCL™ utils (ICD and [Khronos OpenCL™ API Headers](https://github.com/KhronosGroup/OpenCL-Headers.git))
 - kernel_selector/ - OpenCL™ kernels (host+device parts) + utils for optimal kernels selection
   - common/ - definition of some generic classes/structures used in kernel_selector
   - core/ - kernels, kernel selectors, and kernel parameters definitions
     - actual_kernels/ - host side part of OpenCL™ kernels including applicability checks, performance heuristics and Local/Global work-groups description
     - cache/
       - cache.json - tuning cache of the kernels which is redistributed with the plugin to improve kernels and kernel parameters selection for better performance
     - cl_kernels/ - templates of GPU kernels (device part) written on OpenCL™
     - common/ - utils for code generation and kernels selection
 - src/
   - gpu/ - definition of nodes and other gpu specific structures
   - graph_optimizer/ - passes for graph transformations
   - include/ - headers with graph nodes and runtime
 - tests/ - unit tests
 - tutorial/ - examples how to work with clDNN api
 - utils/
   - build/ - cmake scripts for building clDNN
   - rapidjson/ - thirdparty [RapidJSON](https://github.com/Tencent/rapidjson) lib for reading json files (cache.json)
```
