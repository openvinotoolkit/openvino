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

Runtime (or deployment) optimization is focused on tuning of the inference "parameters" (i.e., optimal number of the requests executed simultaneously) and other means of how a model is "executed". The relevant "runtime" configuration is the `ov::hint::inference_precision` which trades the accuracy for the performance (i.e., by allowing the `fp16/bf16` execution for the layers that remain in `fp32` after quantization of the original `fp32` model). 

After that, a possible optimization should start by defining the use case. For example, whether the target scenario emphasizes throughput over latency, i.e., processing millions of samples by overnight jobs in the data centers.
In contrast, real-time usages would likely trade off the throughput to deliver the results at minimal latency. Often this is a combined scenario that targets highest possible throughput, while maintaining a specific latency threshold.  

Also it is important to understand how the full-stack application uses the inference "end-to-end" component. For example, what are the stages that need to be orchestrated? In some cases, a significant part of the workload time is spent on bringing and preparing the input data. Further in the documentation, there are the tips on efficient way of connecting the data input pipeline and the model inference.
These are also a common performance solutions that help with both latency and throughput scenarios. 

The articles below cover the associated "runtime" performance optimizations subjects. For more information on this topic, see the [matrix support of the features by the individual devices](@ref features_support_matrix).
 
* [Inputs Pre-processing with the OpenVINO](@ref inputs_pre_processing).
* [Async API](@ref async_api).
* [The 'get_tensor' Idiom](@ref tensor_idiom).
* For variably-sized inputs, consider [dynamic shapes](../OV_Runtime_UG/ov_dynamic_shapes.md).

**For use case specific optimizations** see the guides, depending on whether you want to optimize for [latency](./dldt_deployment_optimization_latency.md) or [throughput](./dldt_deployment_optimization_tput.md).

## Writing Performance Portable Inference Application
Each of the [supported devices](../OV_Runtime_UG/supported_plugins/Supported_Devices.md) in OpenVINO offers a bunch of low-level performance settings. 

> **NOTE**: Alerting this detailed configuration requires deep architecture understanding.

While the resulting performance may be optimal for the specific combination of the device and model that is inferred, it is actually neither device/model nor future-proof. Even within a family of the devices (like various CPUs), different instruction set or number of CPU cores would eventually result in different execution configuration to be optimal. Likewise, the optimal batch size is highly specific to the particular instance of the GPU. Compute vs memory-bandwidth requirements for the model being inferenced, as well as inference precision, possible model quantization also contribute to the optimal parameters selection. Finally, the optimal execution parameters of one device do not map transparently to another device type. For example, both the CPU and GPU devices support the notion of the [streams](./dldt_deployment_optimization_tput_advanced.md), yet the optimal number of the streams is deduced very differently.
 
Therefore, to mitigate the performance configuration complexity, the **performance hints** offer the high-level "presets" for the **latency** and **throughput**. For more details, see the [Performance Hints usage](../OV_Runtime_UG/performance_hints.md).

Beyond "parameters" execution, there is a device-specific "scheduling" that strongly influences the performance. 
Specifically, GPU-oriented optimizations like batching, which combines many of inputs to achieve optimal throughput, do not always map well to the CPU. For detailed examples, see the [further internals](dldt_deployment_optimization_internals.md) sections. In the same sections, there are details of the implementation (particularly how OpenVINO implements the *throughput* approach) for the specific devices. Keep in mind that the hints make this transparent to the application. For example, the hints anticipate the need for explicit (application-side) batching or streams.

The hints are sufficient to keep separate infer requests per camera or per another source of input and process the requests in parallel, using Async API as explained in the [application design considerations section](@ref throughput_app_design). The main requirement for the application to leverage the throughput is **running multiple inference requests in parallel**.

In summary, when the performance "portability" is of concern, consider the Performance Hints as a solution. For the API examples and details, see the [High-level Performance Hints](../OV_Runtime_UG/performance_hints.md).
