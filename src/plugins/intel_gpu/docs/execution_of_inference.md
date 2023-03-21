# Execution of Inference

Network execution happens when user calls `inferRequest->infer()` or `inferRequest->start_async()`. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/samples/cpp/benchmark_app/main.cpp#L929)

In high level, all we need to do is enqueuing OCL kernels with buffers. For that purpose, we need to find the `cldnn::network` instance as it contains the required buffers for execution. [(link)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/basic_data_structures.md#network-impl) `CPUStreamExecutor` is holding streams and the stream corresponds to the `cldnn::network` structure. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/inference/src/threading/ie_cpu_streams_executor.cpp#L263)

The main body of network execution is `cldnn::network::execute_impl`. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/graph/network.cpp#L663) In this function, `set_arguments()` is called to set OpenCL arguments and `execute_primitive` is called to enqueue kernels to OCL queue.
In case of synchronous API call(i.e. `inferRequest->infer()`), waiting for completion of kernels is also required. It is called from `cldnn::network_output::get_memory()` function. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/include/intel_gpu/graph/network.hpp#L31)

## Optimized-out node
During graph compilation [(link)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/graph_optimization_passes.md), some nodes may be optimized out.

For example, concat operation may be executed _implicitly_, or in other words, concat may be _optimized out_. Implicit concat is possible when the input of concat can put the output tensor directly into the result tensor of concat.

In such case, we don't remove the node in the graph for integrity of node connection. Concat layer is just marked as **optimized-out** and not executed during runtime. [(src)](https://github.com/openvinotoolkit/openvino/blob/dc6e5c51ee4bfb8a26a02ebd7a899aa6a8eeb239/src/plugins/intel_gpu/src/graph/impls/ocl/primitive_base.hpp#L155)

## Dumping layer in/out buffer during execution
`cldnn::network::execute_impl` also contains some logic to dump layer in/out buffers for debugging purpose. As it is related to memory usage, it deserves some description, too.

In order to dump buffers, we need to wait for the moment that the kernel is about to be called(for source buffer) or just called(for destination buffer). In other moments, we don't have the layer's buffer as the buffers are reused from memory pool. [(link)](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/memory_allocation_gpu_plugin.md#memory-dependency-and-memory-pool)

`get_stream().finish()` is called firstly as we need to be synchronous with kernel execution. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/graph/network.cpp#L712) Then we can access the buffer. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/graph/network.cpp#L114) This access varies depending on the kind of buffer. If it is `usm_host` or `usm_shared`, it is just accessed directly. If it is `usm_device`, it is accessed after copying the data into host memory because host cannot access `usm_device` directly. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp#L312) If it is ocl memory, we map this into host memory. [(src)](https://github.com/openvinotoolkit/openvino/blob/f48b23362965fba7e86b0077319ea0d7193ec429/src/plugins/intel_gpu/src/runtime/ocl/ocl_memory.cpp#L46) 

Typical network execution happens with `usm_host` for network input and output and `usm_device` for the buffers inside the network.

For usage of this dumping feature, please see [link](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/docs/gpu_debug_utils.md#layer-inout-buffer-dumps).

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO Core Components](../../../README.md)
 * [OpenVINO Plugins](../../README.md)
 * [OpenVINO GPU Plugin](../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)