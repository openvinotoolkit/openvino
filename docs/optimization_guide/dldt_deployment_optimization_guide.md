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

Runtime or deployment optimizations are focused on tuning of the inference _parameters_ (e.g. optimal number of the requests executed simultaneously) and other means of how a model is _executed_. 

As referenced in the parent [performance introduction topic](./dldt_optimization_guide.md), the [dedicated document](./model_optimization_guide.md) covers the  **model-level optimizations** like quantization that unlocks the 8-bit inference. Model-optimizations are most general and help any scenario and any device (that e.g. accelerates the quantized models). The relevant _runtime_ configuration is `ov::hint::inference_precision` allowing the devices to trade the accuracy for the performance (e.g. by allowing the fp16/bf16 execution for the layers that remain in fp32 after quantization of the original fp32 model).

Then, possible optimization should start with defining the use-case. For example, whether the target scenario emphasizes throughput over latency like processing millions of samples by overnight jobs in the data centers.
In contrast, real-time usages would likely trade off the throughput to deliver the results at minimal latency. Often this is a combined scenario that targets highest possible throughput while maintaining a specific latency threshold.
Below you can find summary on the associated tips.  

How the full-stack application uses the inference component _end-to-end_ is also important.  For example, what are the stages that needs to be orchestrated? In some cases a significant part of the workload time is spent on bringing and preparing the input data. Below you can find multiple tips on connecting the data input pipeline and the model inference efficiently.
These are also common performance tricks that help both latency and throughput scenarios.

Further documents cover the associated  _runtime_ performance optimizations topics. Please also consider [matrix support of the features by the individual devices](@ref  features_support_matrix).

[General, application-level optimizations](dldt_deployment_optimization_common.md), and specifically:
 
* [Inputs Pre-processing with the OpenVINO](../OV_Runtime_UG/preprocessing_overview.md)

* [Async API and 'get_tensor' Idiom](dldt_deployment_optimization_common.md)

* For variably-sized inputs, consider [dynamic shapes](../OV_Runtime_UG/ov_dynamic_shapes.md)

**Use-case specific optimizations** such as optimizing for [latency](./dldt_deployment_optimization_latency.md) or [throughput](./dldt_deployment_optimization_tput.md) 

## Writing Performance Portable Inference Application
Each of the OpenVINO's [supported devices](../OV_Runtime_UG/supported_plugins/Supported_Devices.md) offers a bunch of low-level performance settings. 
Tweaking this detailed configuration requires deep architecture understanding.

Also, while the resulting performance may be optimal for the specific combination of the device and the model that is inferred, it is actually neither device/model nor future-proof:
- Even within a family of the devices (like various CPUs), different instruction set, or number of CPU cores would eventually result in different execution configuration to be optimal.
- Similarly the optimal batch size is very much specific to the particular instance of the GPU.
- Compute vs memory-bandwidth requirements for the model being inferenced, as well as inference precision, possible model's quantization also contribute to the optimal parameters selection.
- Finally, the optimal execution parameters of one device do not transparently map to another device type, for example:
    - Both the CPU and GPU devices support the notion of the [streams](./dldt_deployment_optimization_tput_advanced.md), yet the optimal number of the streams is deduced very differently.
 
Here, to mitigate the performance configuration complexity the **Performance Hints** offer the high-level "presets" for the **latency** and **throughput**, as detailed in the [Performance Hints usage document](../OV_Runtime_UG/performance_hints.md).

Beyond execution _parameters_ there is a device-specific _scheduling_ that greatly affects the performance. 
Specifically, GPU-oriented optimizations like batching, which combines many (potentially tens) of inputs to achieve optimal throughput, do not always map well to the CPU, as e.g. detailed in the [further internals](dldt_deployment_optimization_internals.md) sections.

The hints really hide the _execution_ specifics required to saturate the device. In the [internals](dldt_deployment_optimization_internals.md) sections you can find the implementation details (particularly how the OpenVINO implements the 'throughput' approach) for the specific devices. Keep in mind that the hints make this transparent to the application. For example, the hints obviates the need for explicit (application-side) batching or streams.

With the hints, it is enough to keep separate infer requests per camera or another source of input and process the requests in parallel using Async API as explained in the [application design considerations section](@ref throughput_app_design). The main requirement for the application to leverage the throughput is **running multiple inference requests in parallel**.


In summary, when the performance _portability_ is of concern, consider the Performance Hints as a solution. You may find further details and API examples [here](../OV_Runtime_UG/performance_hints.md).
