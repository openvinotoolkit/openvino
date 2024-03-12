.. {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide}

Optimize Inference
==================


.. toctree::
   :maxdepth: 1
   :hidden:

   optimize-inference/general-optimizations
   optimize-inference/high-level-performance-hints
   optimize-inference/precision-control
   optimize-inference/optimizing-latency
   optimize-inference/optimizing-throughput
   optimize-inference/optimize-preprocessing
   optimize-inference/optimizing-low-level-implementation
   Optimizing memory usage <optimize-inference/optimizing-memory-usage>


.. meta::
   :description: Improving inference performance involves model and runtime
                 optimizations that can be done independently. Inference
                 speed depends on latency and throughput.


Runtime optimization, or deployment optimization, focuses on tuning inference parameters and execution means (e.g., the optimum number of requests executed simultaneously). Unlike model-level optimizations, they are highly specific to the hardware and case they are used for, and often come at a cost.
``ov::hint::inference_precision`` is a "typical runtime configuration" which trades accuracy for performance, allowing ``fp16/bf16`` execution for the layers that remain in ``fp32`` after quantization of the original ``fp32`` model.

Therefore, optimization should start with defining the use case. For example, if it is about processing millions of samples by overnight jobs in data centers, throughput could be prioritized over latency. On the other hand, real-time usages would likely trade off throughput to deliver the results at minimal latency. A combined scenario is also possible, targeting the highest possible throughput, while maintaining a specific latency threshold.

It is also important to understand how the full-stack application would use the inference component "end-to-end." For example, to know what stages need to be orchestrated to save workload devoted to fetching and preparing input data.

For more information on this topic, see the following articles:

* :doc:`Supported Devices <../../about-openvino/compatibility-and-support/supported-devices>`
* :doc:`Inference Devices and Modes <inference-devices-and-modes>`
* :ref:`Inputs Pre-processing with the OpenVINO <inputs_pre_processing>`
* :ref:`Async API <async_api>`
* :ref:`The 'get_tensor' Idiom <tensor_idiom>`
* For variably-sized inputs, consider :doc:`dynamic shapes <dynamic-shapes>`


See the :doc:`latency <optimize-inference/optimizing-latency>` and :doc:`throughput <optimize-inference/optimizing-throughput>` optimization guides, for **use-case-specific optimizations**

Writing Performance-Portable Inference Applications
###################################################

Although inference performed in OpenVINO Runtime can be configured with a multitude of low-level performance settings, it is not recommended in most cases. Firstly, achieving the best performance with such adjustments requires deep understanding of device architecture and the inference engine.


Secondly, such optimization may not translate well to other device-model combinations. In other words, one set of execution parameters is likely to result in different performance when used under different conditions. For example:

* both the CPU and GPU support the notion of :ref:`streams <optimize-inference/optimizing-throughput_advanced>`, yet they deduce their optimal number very differently.
* Even among devices of the same type, different execution configurations can be considered optimal, as in the case of instruction sets or the number of cores for the CPU and the batch size for the GPU.
* Different models have different optimal parameter configurations, considering factors such as compute vs memory-bandwidth, inference precision, and possible model quantization.
* Execution "scheduling" impacts performance strongly and is highly device-specific, for example, GPU-oriented optimizations like batching, combining multiple inputs to achieve the optimal throughput, :doc:`do not always map well to the CPU <optimize-inference/optimizing-low-level-implementation>`.


To make the configuration process much easier and its performance optimization more portable, the option of :doc:`Performance Hints <optimize-inference/high-level-performance-hints>` has been introduced. It comprises two high-level "presets" focused on either **latency** or **throughput** and, essentially, makes execution specifics irrelevant.

The Performance Hints functionality makes configuration transparent to the application, for example, anticipates the need for explicit (application-side) batching or streams, and facilitates parallel processing of separate infer requests for different input sources


Additional Resources
####################

* :ref:`Using Async API and running multiple inference requests in parallel to leverage throughput <throughput_app_design>`.
* :doc:`The throughput approach implementation details for specific devices <optimize-inference/optimizing-low-level-implementation>`
* :doc:`Details on throughput <optimize-inference/optimizing-throughput>`
* :doc:`Details on latency <optimize-inference/optimizing-latency>`
* :doc:`API examples and details <optimize-inference/high-level-performance-hints>`

