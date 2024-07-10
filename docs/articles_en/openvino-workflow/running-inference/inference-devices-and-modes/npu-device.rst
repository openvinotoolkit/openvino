NPU Device
==========

.. meta::
   :description: OpenVINO™ supports the Neural Processing Unit,
                 a low-power processing device dedicated to running AI inference.


The Neural Processing Unit is a low-power hardware solution, introduced with the
Intel® Core™ Ultra generation of CPUs (formerly known as Meteor Lake). It enables
you to offload certain neural network computation tasks from other devices,
for more streamlined resource management.

NPU Plugin is now available through all relevant OpenVINO distribution channels.

| **Supported Platforms:**
|   Host: Intel® Core™ Ultra (former Meteor Lake)
|   NPU device: NPU 3720
|   OS: Ubuntu* 22.04 64-bit (with Linux kernel 6.6+), MS Windows* 11 64-bit (22H2, 23H2)

NPU Plugin needs an NPU Driver to be installed on the system for both compiling and executing a model.
Follow the instructions below to install the latest NPU drivers:

* `Windows driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__
* `Linux driver <https://github.com/intel/linux-npu-driver/releases>`__


The plugin uses the graph extension API exposed by the driver to convert the OpenVINO specific representation
of the model into a proprietary format. The compiler included in the user mode driver (UMD) performs
platform specific optimizations in order to efficiently schedule the execution of network layers and
memory transactions on various NPU hardware submodules.

To use NPU for inference, pass the device name to the ``ov::Core::compile_model()`` method:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/articles_en/assets/snippets/compile_model_npu.py
         :language: py
         :fragment: [compile_model_default_npu]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/articles_en/assets/snippets/compile_model_npu.cpp
         :language: cpp
         :fragment: [compile_model_default_npu]


Model Caching
#############################

Model Caching helps reduce application startup delays by exporting and reusing the compiled
model automatically. The following two compilation-related metrics are crucial in this area:

| **First Ever Inference Latency (FEIL)**
|   Measures all steps required to compile and execute a model on the device for the
    first time. It includes model compilation time, the time required to load and
    initialize the model on the device and the first inference execution.
| **First Inference Latency (FIL)**
|   Measures the time required to load and initialize the pre-compiled model on the
    device and the first inference execution.


UMD Dynamic Model Caching
+++++++++++++++++++++++++++++

UMD model caching is a solution enabled by default in the current NPU driver.
It improves time to first inference (FIL) by storing the model in the cache
after the compilation (included in FEIL), based on a hash key. The process
may be summarized in three stages:

1. UMD generates the key from the input IR model and build arguments
2. UMD requests the DirectX Shader cache session to store the model
   with the computed key.
3. All subsequent requests to compile the same IR model with the same arguments
   use the pre-compiled model, reading it from the cache instead of recompiling.


OpenVINO Model Caching
+++++++++++++++++++++++++++++

OpenVINO Model Caching is a common mechanism for all OpenVINO device plugins and
can be enabled by setting the ``ov::cache_dir`` property. This way, the UMD model
caching is automatically bypassed by the NPU plugin, which means the model
will only be stored in the OpenVINO cache after compilation. When a cache hit
occurs for subsequent compilation requests, the plugin will import the model
instead of recompiling it.

For more details about OpenVINO model caching, see the
:doc:`Model Caching Overview <../optimize-inference/optimizing-latency/model-caching-overview>`.

| **Supported Inference Data Types**
| The NPU plugin supports the following data types as inference precision of internal primitives:
|    Floating-point data types: F32, F16
|    Quantized data types: U8 (quantized models may be INT8 or mixed FP16-INT8)
|    Computation precision for the HW is FP16.
|
| For more details on how to get a quantized model, refer to the
  :doc:`Model Optimization guide <../../model-optimization>` and
  :doc:`NNCF tool quantization guide <../../model-optimization-guide/quantizing-models-post-training/basic-quantization-flow>`.

Supported Features and properties
#######################################

The NPU device is currently supported by AUTO inference modes
(HETERO execution is partially supported, for certain models).

The NPU support in OpenVINO is still under active development and may
offer a limited set of supported OpenVINO features.

**Supported Properties:**

.. tab-set::

   .. tab-item:: Read-write properties

      .. code-block::

         ov::device::id
         ov::log::level
         ov::hint::enable_cpu_pinning
         ov::hint::inference_precision
         ov::hint::model_priority
         ov::hint::num_requests
         ov::hint::performance_mode
         ov::cache_dir
         ov::compilation_num_threads
         ov::enable_profiling

   .. tab-item:: Read-only properties

      .. code-block::

         ov::supported_properties
         ov::available_devices
         ov::optimal_number_of_infer_requests
         ov::range_for_async_infer_requests
         ov::range_for_streams
         ov::num_streams
         ov::device::architecture
         ov::device::capabilities
         ov::device::full_name
         ov::device::uuid
         ov::intel_npu::device_alloc_mem_size
         ov::intel_npu::device_total_mem_size
         ov::intel_npu::driver_version


.. note::

   The optimum number of inference requests returned by the plugin
   based on the performance mode is **4 for THROUGHPUT** and **1 for LATENCY**.
   The default mode for the NPU device is LATENCY.


Limitations
#############################

* Currently, only models with static shapes are supported on NPU.

**Import/Export:**

Offline compilation and blob import is supported only for development purposes.
Pre-compiled models (blobs) are not recommended to be used in production.
Blob compatibility across different OpenVINO / NPU Driver versions is not
guaranteed.

Additional Resources
#############################

* `Working with NPUs in OpenVINO™ Notebook <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/hello-npu/hello-npu.ipynb>`__
* `Vision colorization Notebook <./../../../notebooks/vision-image-colorization-with-output.html>`__
* `Classification Benchmark C++ Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/classification_benchmark_demo/cpp>`__
* `3D Human Pose Estimation Python Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/3d_segmentation_demo/python>`__
* `Object Detection C++ Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/object_detection_demo/cpp>`__
* `Object Detection Python Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/object_detection_demo/python>`__
