.. {#openvino_docs_OV_UG_supported_plugins_NPU}

NPU Device
==========

.. meta::
   :description: OpenVINO™ supports the Neural Processing Unit,
                 a low-power processing device dedicated to running AI inference.


The Neural Processing Unit is a low-power hardware solution, introduced with the
Intel® Core™ Ultra generation of CPUs (formerly known as Meteor Lake). It enables
you to offload certain neural network computation tasks from other devices,
for more streamlined resource management.

Note that the NPU plugin is currently available only with the Archive distribution of OpenVINO™
and you need to :doc:`install a proper NPU driver <openvino_docs_install_guides_configurations_for_intel_npu>`
to use it successfully. For an in-depth description of the NPU plugin, see:

* `NPU plugin developer documentation <https://github.com/openvinotoolkit/npu_plugin/blob/develop/docs/VPUX_DG/index.md>`__
* `OpenVINO Runtime NPU plugin source files <https://github.com/openvinotoolkit/npu_plugin>`__


| **Supported Platforms:**
|   Host: Intel® Core™ Ultra (former Meteor Lake)
|   NPU device: NPU 3720
|   OS: Ubuntu* 20, MS Windows* 11 (both 64-bit)


| **Supported Inference Data Types**
| The NPU plugin supports the following data types as inference precision of internal primitives:
|    Floating-point data types: f32, f16O
|    Quantized data types: u8 (quantized models may be int8 or mixed fp16-int8)
|    Computation precision for the HW is fp16.
|
| For more details on how to get a quantized model, refer to the
  :doc:`Model Optimization guide <openvino_docs_model_optimization_guide>`, and
  :doc:`NNCF tool quantization guide <basic_quantization_flow>`.



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
:doc:`Model Caching Overview <openvino_docs_OV_UG_Model_caching_overview>`.


Supported Features and properties
#######################################

The NPU device is currently supported by AUTO and MULTI inference modes.

The NPU support in OpenVINO is still under active development and may
offer a limited set of supported OpenVINO features.

**Supported Properties:**

.. tab-set::

   .. tab-item:: Read-write properties

      .. code-block::

         ov::caching_properties
         ov::enable_profiling
         ov::hint::performance_mode
         ov::hint::num_requests
         ov::hint::model_priority
         ov::hint::enable_cpu_pinning
         ov::log::level
         ov::device::id
         ov::cache_dir
         ov::internal::exclusive_async_requests
         ov::intel_vpux::dpu_groups
         ov::intel_vpux::dma_engines
         ov::intel_vpux::compilation_mode
         ov::intel_vpux::compilation_mode_params
         ov::intel_vpux::print_profiling
         ov::intel_vpux::profiling_output_file
         ov::intel_vpux::vpux_platform
         ov::intel_vpux::use_elf_compiler_backend

   .. tab-item:: Read-only properties

      .. code-block::

         ov::supported_properties
         ov::streams::num
         ov::optimal_number_of_infer_requests
         ov::range_for_async_infer_requests
         ov::range_for_streams
         ov::available_devices
         ov::device::uuid
         ov::device::architecture
         ov::device::full_name
         ov::intel_vpux::device_total_mem_size
         ov::intel_vpux::driver_version

.. note::

   The optimum number of inference requests returned by the plugin
   based on the performance mode is **4 for THROUGHPUT** and **1 for LATENCY**.
   The default mode for the NPU device is LATENCY.


Limitations
#############################

* Currently, only the models with static shapes are supported on NPU.
* If the path to the model file includes non-Unicode symbols, such as in Chinese,
  the model cannot be used for inference on NPU. It will return an error.
* Running the Alexnet model with NPU may result in a drop in accuracy.
  At this moment, the googlenet-v4 model is recommended for classification tasks.


Additional Resources
#############################

* `Vision colorization Notebook <notebooks/222-vision-image-colorization-with-output.html>`__
* `Classification Benchmark C++ Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/classification_benchmark_demo/cpp>`__
* `3D Human Pose Estimation Python Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/3d_segmentation_demo/python>`__
* `Object Detection C++ Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/object_detection_demo/cpp>`__
* `Object Detection Python Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/object_detection_demo/python>`__
* `POT-specific sample with sparse resnet-50 generation <https://github.com/openvinotoolkit/openvino/tree/master/tools/pot/openvino/tools/pot/api/samples/prune_and_quantize>`__
