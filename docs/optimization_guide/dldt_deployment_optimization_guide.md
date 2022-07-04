# Runtime Inference Optimizations {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_deployment_optimization_guide_common
   openvino_docs_deployment_optimization_guide_latency
   openvino_docs_deployment_optimization_guide_tput
   openvino_docs_deployment_optimization_guide_tput_advanced
   openvino_docs_deployment_optimization_guide_internals

@endsphinxdirective

Runtime optimizations, or deployment optimizations, focus on tuning inference parameters and execution means (e.g., the optimum number of requests executed simultaneously). Unlike model-level optimizations, they are highly specific to the hardware and case they are used for, and often come at a cost.
`ov::hint::inference_precision` is a "typical runtime configuration" which trades accuracy for performance, allowing `fp16/bf16` execution for the layers that remain in `fp32` after quantization of the original `fp32` model. 

Therefore, optimization should start with defining the use case. For example, if it is about processing millions of samples by overnight jobs in data centers, throughput could be prioritized over latency. On the other hand, real-time usages would likely trade off throughput to deliver the results at minimal latency. A combined scenario is also possible, targeting the highest possible throughput, while maintaining a specific latency threshold.

It is also important to understand how the full-stack application would use the inference component "end-to-end." For example, to know what stages need to be orchestrated to save workload devoted to fetching and preparing input data. 

For more information on this topic, see the following articles:
* [feature support by device](@ref features_support_matrix),
 
* [Inputs Pre-processing with the OpenVINO](@ref inputs_pre_processing).
* [Async API](@ref async_api).
* [The 'get_tensor' Idiom](@ref tensor_idiom).
* For variably-sized inputs, consider [dynamic shapes](../OV_Runtime_UG/ov_dynamic_shapes.md).

See the [latency](./dldt_deployment_optimization_latency.md) and [throughput](./dldt_deployment_optimization_tput.md) optimization guides, for **use-case-specific optimizations** 

## Writing Performance-Portable Inference Applications
Although inference performed in OpenVINO Runtime can be configured with a multitude of low-level performance settings, it is not recommended in most cases. Firstly, achieving the best performance with such adjustments requires deep understanding of device architecture and the inference engine.


Secondly, such optimization may not translate well to other device-model combinations. In other words, one set of execution parameters is likely to result in different performance when used under different conditions. For example:
   * both the CPU and GPU support the notion of [streams](./dldt_deployment_optimization_tput_advanced.md), yet they deduce their optimal number very differently. 
   * Even among devices of the same type, different execution configurations can be considered optimal, as in the case of instruction sets or the number of cores for the CPU and the batch size for the GPU. 
   * Different models have different optimal parameter configurations, considering factors such as compute vs memory-bandwidth, inference precision, and possible model quantization. 
   * Execution "scheduling" impacts performance strongly and is highly device-specific, for example, GPU-oriented optimizations like batching, combining multiple inputs to achieve the optimal throughput, [do not always map well to the CPU](dldt_deployment_optimization_internals.md). 
 
 
To make the configuration process much easier and its performance optimization more portable, the option of [Performance Hints](../OV_Runtime_UG/performance_hints.md) has been introduced. It comprises two high-level "presets" focused on either **latency** or **throughput** and, essentially, makes execution specifics irrelevant.

The Performance Hints functionality makes configuration transparent to the application, for example, anticipates the need for explicit (application-side) batching or streams, and facilitates parallel processing of separate infer requests for different input sources 


Additional materials:
* [Using Async API and running multiple inference requests in parallel to leverage throughput](@ref throughput_app_design).
* [The throughput approach implementation details for specific devices](dldt_deployment_optimization_internals.md) 
* [Details on throughput](dldt_deployment_optimization_tput.md)
* [Details on latency](dldt_deployment_optimization_latency.md)
* [API examples and details](../OV_Runtime_UG/performance_hints.md).
