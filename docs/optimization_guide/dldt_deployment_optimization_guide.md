# Runtime Inference Optimizations {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_deployment_optimization_guide_common
   openvino_docs_deployment_optimization_guide_latency
   openvino_docs_deployment_optimization_guide_tput
   openvino_docs_deployment_optimization_guide_hints
   openvino_docs_deployment_optimization_guide_internals

@endsphinxdirective

## Deployment Optimizations Overview {#openvino_docs_deployment_optimization_guide_overview}
Runtime or deployment optimizations are focused on tuning of the inference _parameters_ (e.g. optimal number of the requests executed simultaneously) and other means of how a model is _executed_.

As referenced in the parent [performance introduction topic](./dldt_optimization_guide.md), the [dedicated document](./model_optimization_guide.md) covers the  **model-level optimizations** like quantization that unlocks the 8-bit inference. Model-optimizations are most general and help any scenario and any device (that e.g. accelerates the quantized models). The relevant _runtime_ configuration is `ov::hint::inference_precision` allowing the devices to trade the accuracy for the performance (e.g. by allowing the fp16/bf16 execution for the layers that remain in fp32 after quantization of the original fp32 model).

Then, possible optimization should start with defining the use-case. For example, whether the target scenario emphasizes throughput over latency like processing millions of samples by overnight jobs in the data centers.
In contrast, real-time usages would likely trade off the throughput to deliver the results at minimal latency. Often this is a combined scenario that targets highest possible throughput while maintaining a specific latency threshold.
Below you can find summary on the associated tips.

How the full-stack application uses the inference component _end-to-end_ is also important.  For example, what are the stages that needs to be orchestrated? In some cases a significant part of the workload time is spent on bringing and preparing the input data. Below you can find multiple tips on connecting the data input pipeline and the model inference efficiently.
These are also common performance tricks that help both latency and throughput scenarios.

Further documents cover the associated  _runtime_ performance optimizations topics. Please also consider [matrix support of the features by the individual devices](@ref features_support_matrix).

**General, application-level optimizations**, and specifically:

* [Inputs Pre-processing with the OpenVINO](../OV_Runtime_UG/preprocessing_overview.md)

* [Async API and 'get_tensor' Idiom](./dldt_deployment_optimization_common.md)

* For variably-sized inputs, consider [dynamic shapes](../OV_Runtime_UG/ov_dynamic_shapes.md)

**Use-case specific optimizations** along with some implementation details:

* Optimizing for [throughput](./dldt_deployment_optimization_tput.md) and [latency](./dldt_deployment_optimization_latency.md)

* [OpenVINO's high-level performance hints](./dldt_deployment_optimization_hints.md) as the portable, future-proof approach for performance configuration, thar does not requires re-tuning when the model or device has changed.
    * **If the performance portability is of concern, consider using the [hints](../OV_Runtime_UG/performance_hints.md) first.**