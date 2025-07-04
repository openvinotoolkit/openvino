# Memory Allocation in GPU Plugin

## Allocation types

GPU plugin supports four types of memory allocation as below. Note that the prefix `usm_` indicates the allocation type using Intel Unified Shared Memory (USM) extension for OpenCL. For more detailed information about the USM extension, refer to [this](https://www.khronos.org/registry/OpenCL/extensions/intel/cl_intel_unified_shared_memory.html) page.
* `cl_mem` : Standard OpenCL cl_mem allocation.
* `usm_host` : Allocated in host memory and accessible by both of host and device. Non-migratable.
* `usm_shared` : Allocated in host and devices and accessible by both host and device. The memories are automatically migrated on demand.
* `usm_device` : Allocated in device memory and accessible only by the device which owns the memory. Non-migratable.

Note that there are a few restrictions on a memory allocation:

* Allocation of a single memory object should not exceed the available device memory size, that is, the value obtained by `CL_DEVICE_GLOBAL_MEM_SIZE`.
* The sum of all memory objects required to execute a kernel (that is, the sum of inputs and outputs of a kernel) should not exceed the target available memory. For example, if you want to allocate a memory object to the device memory, the above restrictions should be satisfied against the device memory. Otherwise, the memory object should be allocated on the host memory.

## Memory allocation API

In GPU plugin, the allocation for each allocation type can be done with [engine::allocate_memory](https://github.com/openvinotoolkit/openvino/blob/de47a3b4a4ba1f8464b85a665c4d58403e0d16b8/src/plugins/intel_gpu/include/intel_gpu/runtime/engine.hpp#L51), which
calls the corresponding memory object wrapper for each allocation type: [gpu_buffer](https://github.com/openvinotoolkit/openvino/blob/de47a3b4a4ba1f8464b85a665c4d58403e0d16b8/src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp#L35), [gpu_usm](https://github.com/openvinotoolkit/openvino/blob/de47a3b4a4ba1f8464b85a665c4d58403e0d16b8/src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp#L291).

## Dump memory allocation history

The memory allocation history is being managed by the `engine`, which can be dumped by setting the environment variable `OV_VERBOSE=2` if OpenVINO is built with the cmake configuration `ENABLE_DEBUG_CAPS=ON`.
```cpp
...
GPU_Debug: Allocate 58982400 bytes of usm_host allocation type (current=117969612; max=117969612)
GPU_Debug: Allocate 44621568 bytes of usm_device allocation type (current=44626380; max=44626380)
GPU_Debug: Allocate 44236800 bytes of usm_host allocation type (current=162206412; max=162206412)
GPU_Debug: Allocate 14873856 bytes of usm_device allocation type (current=59500236; max=59500236)
...
```
Here, `current` denotes the amount of total allocated memory at that moment, while `max` denotes the peak record of the total memory allocation until that moment.

## Allocated memory objects

The typical memory allocation performed in the GPU plugin can be categorized as follows:
* `Constant memory allocation`: In GPU plugin, constant data are held by the `data` primitives and the required memory objects are [allocated](https://github.com/openvinotoolkit/openvino/blob/de47a3b4a4ba1f8464b85a665c4d58403e0d16b8/src/plugins/intel_gpu/src/plugin/ops/constant.cpp#L181) and assigned at the creation of the data primitive. First, it is allocated on the host memory and the constant data are copied from the corresponding blob in ngraph. Once all the transformation and optimization processes in `cldnn::program` are finished and the user nodes of the data are known as the GPU operations using the device memory, then the memory is reallocated on the device memory and the constant data is copied to there (that is, [transferred](https://github.com/openvinotoolkit/openvino/blob/de47a3b4a4ba1f8464b85a665c4d58403e0d16b8/src/plugins/intel_gpu/src/graph/program.cpp#L457)). Note that constant data is shared within batches and streams.
* `Output memory allocation`: A memory object to store the output result of each primitive is created at the creation of each `primitive_inst` ([link](https://github.com/openvinotoolkit/openvino/blob/de47a3b4a4ba1f8464b85a665c4d58403e0d16b8/src/plugins/intel_gpu/src/graph/primitive_inst.cpp#L263)), except when the output is reusing the input memory. Note that the creation of a `primitive_inst` is done in descending order of the output memory size for achieving better memory reusing efficiency.

* `Intermediate memory allocation`: Some primitives such as _detection_output_ and _non_max_suppression_ consisting of multiple kernels require intermediate memories to exchange data b/w those kernels. The allocation of such intermediate memories happens after all allocation for `primitive_insts` is finished ([link](https://github.com/openvinotoolkit/openvino/blob/4c01d6c50c6d314373dffd2a8ddbc294011b2508/src/plugins/intel_gpu/src/graph/network.cpp#L592)). After all, it needs to be processed in a processing order to use the predecessors' allocation information to decide whether to allocate it on device memory or not by checking the memory allocation restriction described above.

## Memory dependency and memory pool

In GPU plugin, multiple memory objects can be allocated at the same address, when there is no dependency between their users. For example, a memory region of a `program_node` _A_'s output memory can be allocated for another `program_node` _B_'s output, if the output of _A_ is no longer used by any other `program_node`, when the result of the _B_ is to be stored. This mechanism is realized by the following two parts;
1. `Memory dependency` : memory_dependencies of a `program_node` is set by the memory dependency passes. There are two kinds of memory dependency passes:
    * `basic_memory_dependencies` : Assuming an in-order-queue execution, this pass adds dependencies to a `program_node`, which are deduced by checking its direct input and output nodes only.
    * `oooq_memory_dependencies` : Assuming an out-of-order-queue execution, this pass adds dependencies to all pairs of `program_nodes` that can potentially be executed at the same time.
2. `Memory pool` : The GPU plugin has a [memory pool](https://github.com/openvinotoolkit/openvino/blob/de47a3b4a4ba1f8464b85a665c4d58403e0d16b8/src/plugins/intel_gpu/include/intel_gpu/runtime/memory_pool.hpp), which is responsible for the decision of allocation or reuse for an allocation request. This `memory_pool` utilizes the memory dependencies set by the above two passes in the decision of reuse of not. Note that each `cldnn::network` has its own `memory_pool`.

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
