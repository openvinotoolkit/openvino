.. {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide}

Optimize Inference
==================


.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_deployment_optimization_guide_common
   openvino_docs_OV_UG_Performance_Hints
   openvino_docs_OV_UG_Precision_Control
   openvino_docs_deployment_optimization_guide_latency
   openvino_docs_deployment_optimization_guide_tput
   openvino_docs_deployment_optimization_guide_tput_advanced
   openvino_docs_OV_UG_Preprocessing_Overview
   openvino_docs_deployment_optimization_guide_internals
   openvino_docs_memory_optimization_guide

.. meta::
   :description: Improving inference performance involves model and runtime 
                 optimizations that can be done independently. Inference 
                 speed depends on latency and throughput.


Runtime optimization, or deployment optimization, focuses on tuning inference parameters and execution means (e.g., the optimum number of requests executed simultaneously). Unlike model-level optimizations, they are highly specific to the hardware and case they are used for, and often come at a cost.
`ov::hint::inference_precision <groupov_runtime_cpp_prop_api.html#doxid-group-ov-runtime-cpp-prop-api-1gad605a888f3c9b7598ab55023fbf44240>`__ is a "typical runtime configuration" which trades accuracy for performance, allowing ``fp16/bf16`` execution for the layers that remain in ``fp32`` after quantization of the original ``fp32`` model.

Therefore, optimization should start with defining the use case. For example, if it is about processing millions of samples by overnight jobs in data centers, throughput could be prioritized over latency. On the other hand, real-time usages would likely trade off throughput to deliver the results at minimal latency. A combined scenario is also possible, targeting the highest possible throughput, while maintaining a specific latency threshold.

It is also important to understand how the full-stack application would use the inference component "end-to-end." For example, to know what stages need to be orchestrated to save workload devoted to fetching and preparing input data.

For more information on this topic, see the following articles:

* :ref:`feature support by device <devicesupport-feature-support-matrix>`
* :ref:`Inputs Pre-processing with the OpenVINO <inputs_pre_processing>`
* :ref:`Async API <async_api>`
* :ref:`The 'get_tensor' Idiom <tensor_idiom>`
* For variably-sized inputs, consider :doc:`dynamic shapes <openvino_docs_OV_UG_DynamicShapes>`


See the :doc:`latency <openvino_docs_deployment_optimization_guide_latency>` and :doc:`throughput <openvino_docs_deployment_optimization_guide_tput>` optimization guides, for **use-case-specific optimizations**

Writing Performance-Portable Inference Applications
###################################################

Although inference performed in OpenVINO Runtime can be configured with a multitude of low-level performance settings, it is not recommended in most cases. Firstly, achieving the best performance with such adjustments requires deep understanding of device architecture and the inference engine.


Secondly, such optimization may not translate well to other device-model combinations. In other words, one set of execution parameters is likely to result in different performance when used under different conditions. For example:

* both the CPU and GPU support the notion of :ref:`streams <openvino_docs_deployment_optimization_guide_tput_advanced>`, yet they deduce their optimal number very differently.
* Even among devices of the same type, different execution configurations can be considered optimal, as in the case of instruction sets or the number of cores for the CPU and the batch size for the GPU.
* Different models have different optimal parameter configurations, considering factors such as compute vs memory-bandwidth, inference precision, and possible model quantization.
* Execution "scheduling" impacts performance strongly and is highly device-specific, for example, GPU-oriented optimizations like batching, combining multiple inputs to achieve the optimal throughput, :doc:`do not always map well to the CPU <openvino_docs_deployment_optimization_guide_internals>`.


To make the configuration process much easier and its performance optimization more portable, the option of :doc:`Performance Hints <openvino_docs_OV_UG_Performance_Hints>` has been introduced. It comprises two high-level "presets" focused on either **latency** or **throughput** and, essentially, makes execution specifics irrelevant.

The Performance Hints functionality makes configuration transparent to the application, for example, anticipates the need for explicit (application-side) batching or streams, and facilitates parallel processing of separate infer requests for different input sources


Additional Resources
####################

* :ref:`Using Async API and running multiple inference requests in parallel to leverage throughput <throughput_app_design>`.
* :doc:`The throughput approach implementation details for specific devices <openvino_docs_deployment_optimization_guide_internals>`
* :doc:`Details on throughput <openvino_docs_deployment_optimization_guide_tput>`
* :doc:`Details on latency <openvino_docs_deployment_optimization_guide_latency>`
* :doc:`API examples and details <openvino_docs_OV_UG_Performance_Hints>`

