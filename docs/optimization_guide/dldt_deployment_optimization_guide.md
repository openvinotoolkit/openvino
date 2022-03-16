# Runtime Inference Optimizations {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
    
   openvino_docs_deployment_optimization_guide_common
   openvino_docs_deployment_optimization_guide_latency
   openvino_docs_deployment_optimization_guide_tput
   openvino_docs_deployment_optimization_guide_hints

@endsphinxdirective

## Deployment Optimizations Overview {#openvino_docs_deployment_optimization_guide_overview}
Runtime or deployment optimizations focus is tuning of the inference parameters (e.g. optimal number of the requests executed simultaneously) and other means of how a model is _executed_. 

Here, possible optimization should start with defining the use-case. For example, whether the target scenario emphasizes throughput over latency like processing millions of samples by overnight jobs in the data centers.
In contrast, real-time usages would likely trade off the throughput to deliver the results at minimal latency. 
Often this is a combined scenario that targets highest possible throughput while maintaining a specific latency threshold.

Each of the [OpenVINO supported devices](../OV_Runtime_UG/supported_plugins/Device_Plugins.md) offers low-level performance configuration. This allows to leverage the optimal model performance on the _specific_ device, but may require careful re-tuning when the model or device has changed.
**If the performance portability is of concern, consider using the [OpenVINO High-Level Performance Hints](../OV_Runtime_UG/performance_hints.md) first.**  

Finally, how the full-stack application uses the inference component _end-to-end_ is important.  
For example, what are the stages that needs to be orchestrated? In some cases a significant part of the workload time is spent on bringing and preparing the input data. As detailed in the section on the [general optimizations](./dldt_deployment_optimization_common.md), the inputs population can be performed asynchronously to the inference. Also, in many cases the (image) [pre-processing can be offloaded to the OpenVINO](../OV_Runtime_UG/preprocessing_overview.md). For variably-sized inputs, consider [dynamic shapes](../OV_Runtime_UG/ov_dynamic_shapes.md) to efficiently connect the data input pipeline and the model inference.
These are common performance tricks that help both latency and throughput scenarios. 

 Similarly, the _model-level_ optimizations like [quantization that unlocks the int8 inference](../OV_Runtime_UG/Int8Inference.md) are general and help any scenario. As referenced in the [performance introduction topic](./dldt_optimization_guide.md), these are covered in the [dedicated document](./model_optimization_guide.md). Additionally, the  `ov::hint::inference_precision` allows the devices to trade the accuracy for the performance at the _runtime_ (e.g. by allowing the fp16/bf16 execution for the layers that remain in fp32 after quantization of the original fp32 model). 
 
Further documents cover the  _runtime_ performance optimizations topics. Please also consider [matrix support of the features by the individual devices](../OV_Runtime_UG/supported_plugins/Device_Plugins.md).

[General, application-level optimizations](./dldt_deployment_optimization_common.md):
 
* Inputs Pre-processing with the OpenVINO

* Async API and 'get_tensor' Idiom

Use-case specific optimizations along with some implementation details:
 
* Optimizing for [throughput](./dldt_deployment_optimization_tput.md) and [latency](./dldt_deployment_optimization_latency.md)
 
* [OpenVINO's high-level performance hints](./dldt_deployment_optimization_hints.md) as the portable, future-proof approach for performance configuration
