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

Runtime, or deployment optimization focuses on tuning inference and execution parameters. Unlike
model-level optimization, it is highly specific to the hardware you use and the goal you want
to achieve. You need to plan whether to prioritize accuracy or performance,
:doc:`throughput <optimize-inference/optimizing-throughput>` or :doc:`latency <optimize-inference/optimizing-latency>`,
or aim at the golden mean. You should also predict how scalable your application needs to be
and how exactly it is going to work with the inference component. This way, you will be able
to achieve the best results for your product.

.. note::

   For more information on this topic, see the following articles:

   * :doc:`Inference Devices and Modes <inference-devices-and-modes>`
   * :ref:`Inputs Pre-processing with the OpenVINO <inputs_pre_processing>`
   * :ref:`Async API <async_api>`
   * :ref:`The 'get_tensor' Idiom <tensor_idiom>`
   * For variably-sized inputs, consider :doc:`dynamic shapes <dynamic-shapes>`

Performance-Portable Inference
################################

To make configuration easier and performance optimization more portable, OpenVINO offers the
:doc:`Performance Hints <optimize-inference/high-level-performance-hints>` feature. It comprises
two high-level “presets” focused on latency **(default)** or throughput.

Although inference with OpenVINO Runtime can be configured with a multitude
of low-level performance settings, it is not recommended, as:

* It requires deep understanding of device architecture and the inference engine.
* It may not translate well to other device-model combinations. For example:

  * CPU and GPU deduce their optimal number of streams differently.
  * Different devices of the same type, favor different execution configurations.
  * Different models favor different parameter configurations (e.g., compute vs memory-bandwidth,
    inference precision, and possible model quantization).
  * Execution “scheduling” impacts performance strongly and is highly device specific. GPU-oriented
    optimizations :doc:`do not always map well to the CPU <optimize-inference/optimizing-low-level-implementation>`.

Additional Resources
####################

* :ref:`Using Async API and running multiple inference requests in parallel to leverage throughput <throughput_app_design>`.
* :doc:`The throughput approach implementation details for specific devices <optimize-inference/optimizing-low-level-implementation>`
* :doc:`Details on throughput <optimize-inference/optimizing-throughput>`
* :doc:`Details on latency <optimize-inference/optimizing-latency>`
* :doc:`API examples and details <optimize-inference/high-level-performance-hints>`

