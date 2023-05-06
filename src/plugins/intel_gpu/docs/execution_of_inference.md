# Execution of Inference

Network execution is triggered when the  `inferRequest->infer()` or `inferRequest->start_async()` methods are called. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/samples/cpp/benchmark_app/main.cpp#L929)

At high level, all that is required to do is enqueuing OCL kernels with buffers. For that purpose, you need to find the `cldnn::network` instance, as it contains the required buffers for execution. [(link)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/basic_data_structures.md#network-impl) `CPUStreamExecutor` is holding streams, and the stream corresponds to the `cldnn::network` structure. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/inference/src/threading/ie_cpu_streams_executor.cpp#L263)

The main body of network execution is `cldnn::network::execute_impl`. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/graph/network.cpp#L663) In this function, `set_arguments()` is called to set OpenCL arguments and `execute_primitive` is called to enqueue kernels to OCL queue.
In case of a synchronous API call (that is, `inferRequest->infer()`), waiting for the completion of kernels is also required. It is called from the `cldnn::network_output::get_memory()` function. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/include/intel_gpu/graph/network.hpp#L31)

## Optimized-out node

During graph compilation [(link)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/graph_optimization_passes.md), some nodes may be optimized out.

For example, concat operation may be executed _implicitly_, or in other words, concat may be _optimized out_. Implicit concat is possible when the input of concat can put the output tensor directly into the resulting tensor of concat.

In such case, you do not remove the node in the graph for the integrity of the node connection. Concat layer is just marked as **optimized-out** and not executed during runtime. [(src)](https://github.com/openvinotoolkit/openvino/blob/dc6e5c51ee4bfb8a26a02ebd7a899aa6a8eeb239/src/plugins/intel_gpu/src/graph/impls/ocl/primitive_base.hpp#L155)

## Dumping layer in/out buffer during execution
The `cldnn::network::execute_impl` function also contains some logic to dump layer in/out buffers for debugging purposes. As it is related to memory usage, it deserves some description, too.

To dump buffers, you need to wait for the moment that the kernel is about to be called (for source buffer) or just called (for destination buffer). In other moments, you do not have the layer's buffer as the buffers are reused from the memory pool. [(link)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/memory_allocation_gpu_plugin.md#memory-dependency-and-memory-pool)

The `get_stream().finish()` function is called first as you need to be synchronous with kernel execution. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/graph/network.cpp#L712). Then, you can access the buffer. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/graph/network.cpp#L114). This access varies depending on the kind of the buffer. If it is `usm_host` or `usm_shared`, it is just accessed directly. If it is `usm_device`, it is accessed after copying the data into host memory because the host cannot access `usm_device` directly. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp#L312) If it is OCL memory, you map this into host memory. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp#L46)

Typical network execution happens with `usm_host` for network input and output and `usm_device` for the buffers inside the network.

For usage of this dumping feature, see this [link](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/gpu_debug_utils.md#layer-inout-buffer-dumps).

## See also

 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)