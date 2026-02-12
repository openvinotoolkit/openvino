NPU Device
==========

.. meta::
   :description: OpenVINO™ supports the Neural Processing Unit,
                 a low-power processing device dedicated to running AI inference.


.. toctree::
   :maxdepth: 1
   :hidden:

   npu-device/remote-tensor-api-npu-plugin
   npu-device/batching-on-npu-plugin


The Neural Processing Unit is a low-power hardware solution, introduced with the
Intel® Core™ Ultra generation of CPUs (formerly known as Meteor Lake). It enables
you to offload certain neural network computation tasks
for more streamlined resource management.

NPU Plugin is now available through all relevant OpenVINO distribution channels.

| **Supported Platforms:**
|   Host: Intel® Core™ Ultra series
|   NPU device: NPU 3720
|   OS: Ubuntu* 22.04 64-bit (with Linux kernel 6.6+), MS Windows* 11 64-bit (22H2, 23H2)

NPU Plugin needs an NPU Driver to be installed on the system to execute a model.
Follow the instructions below to install the latest NPU drivers:

* `Windows driver <https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html>`__
* `Linux driver <https://github.com/intel/linux-npu-driver/releases>`__

Starting with the 2026.0 release, the compiler library is available in the OpenVINO package as a preview feature (``Compiler-In-Plugin``).
The default compiler type remains ``Compiler-In-Driver`` (the compiler library included in the driver package).
Users can override the default compiler selection by setting ``ov::intel_npu::compiler_type``.

The plugin uses either the NPU compiler library included in the driver or
the compiler library included in the OpenVINO package to convert the OpenVINO specific
representation of the model into a proprietary format. The compiler performs platform specific
optimizations in order to efficiently schedule the execution of layers and memory transactions
on various NPU hardware submodules.

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

UMD Dynamic Model Caching can be bypassed for given model by setting boolean property
ov::intel_npu::bypass_umd_caching (NPU_BYPASS_UMD_CACHE) to true at compilation. (default value is false)


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
         ov::hint::execution_mode
         ov::cache_dir
         ov::compilation_num_threads
         ov::enable_profiling
         ov::workload_type
         ov::intel_npu::compilation_mode_params
         ov::intel_npu::compiler_dynamic_quantization
         ov::intel_npu::qdq_optimization
         ov::intel_npu::qdq_optimization_aggressive
         ov::intel_npu::turbo
         ov::intel_npu::platform
         ov::intel_npu::tiles
         ov::intel_npu::max_tiles
         ov::intel_npu::bypass_umd_caching
         ov::intel_npu::defer_weights_load
         ov::intel_npu::run_inferences_sequentially
         ov::intel_npu::disable_idle_memory_prunning
         ov::intel_npu::compiler_type
         ov::intel_npu::enable_strides_for

   .. tab-item:: Read-only properties

      .. code-block::

         ov::supported_properties
         ov::available_devices
         ov::optimal_number_of_infer_requests
         ov::range_for_async_infer_requests
         ov::range_for_streams
         ov::num_streams
         ov::execution_devices
         ov::device::architecture
         ov::device::capabilities
         ov::device::full_name
         ov::device::uuid
         ov::device::luid (windows only)
         ov::device::pci_info
         ov::device::gops
         ov::device::type
         ov::intel_npu::device_alloc_mem_size
         ov::intel_npu::device_total_mem_size
         ov::intel_npu::driver_version
         ov::intel_npu::compiler_version


.. note::

   The optimum number of inference requests returned by the plugin
   based on the performance mode is **4 for THROUGHPUT** and **1 for LATENCY**.
   The default mode for the NPU device is LATENCY.

**ov::intel_npu::compilation_mode_params**

``ov::intel_npu::compilation_mode_params`` is an NPU-specific property that allows
control of model compilation for NPU.

.. note::

   The functionality is in experimental stage currently, can be a subject for
   deprecation and may be replaced with generic OV API in future OV releases.

Following configuration options are supported:

**optimization-level**

Defines an optimization effort hint to the compiler.

.. list-table::
   :widths: 10 200
   :header-rows: 1

   * - **Value**
     - **Description**
   * - 0
     - Reduced subset of optimization passes. May result in smaller compile time.
   * - 1
     - **Default.** Balanced performance/compile time.
   * - 2
     - Prioritize performance over compile time that may be an issue.

**performance-hint-override**

The LATENCY mode can be overridden by specifying ``ov::hint::performance_mode``
Has no effect for other ``ov::hint::PerformanceMode`` hints.

.. list-table::
   :widths: 10 200
   :header-rows: 1

   * - **Value**
     - **Description**
   * - efficiency
     - **Default.** Balanced performance and power consumption.
   * - latency
     - Prioritize performance over power efficiency.

Usage example:

.. code-block::

   map<str, str> config = {ov::intel_npu::compilation_mode_params.name(), ov::Any("optimization-level=1 performance-hint-override=latency")};

   compile_model(model, config);

**ov::intel_npu::turbo**

The turbo mode, where available, provides a hint to the system to maintain the
maximum NPU frequency and memory throughput within the platform TDP limits.
The turbo mode is not recommended for sustainable workloads due to higher power
consumption and potential impact on other compute resources.

.. code-block::

   core.set_property("NPU", ov::intel_npu::turbo(true));

or

.. code-block::

   core.compile_model(ov_model, "NPU", {ov::intel_npu::turbo(true)});

.. note::

   NPU_TURBO usage may cause higher compile time, memory footprint,
   affect workload latency and compatibility issues with older NPU drivers

**ov::intel_npu::max_tiles and ov::intel_npu::tiles**

For on-device compilation, the plugin queries the driver for the available number of tiles and sets ``ov::intel_npu::max_tiles``.
``ov::intel_npu::max_tiles`` is a read-write property to allow users to set it during offline compilation.
Note that ``ov::intel_npu::max_tiles`` represents the maximum number of tiles available,
but the compiler may target a lower number of tiles depending on other properties.
Users can set ``ov::intel_npu::tiles`` to override the number of tiles selected by the compiler based on other properties.

.. note::

   When setting ``ov::intel_npu::tiles``, users must ensure that the value does not
   exceed ``ov::intel_npu::max_tiles``. Any tile count other than 1 may impact
   cross-device compatibility if it is not explicitly validated against the target
   devices's `ov::intel_npu::max_tiles` value.

**ov::intel_npu::compiler_type**

This property allows users to override the default compiler type selected by the plugin.
To use ``Compiler-In-Plugin`` whenever possible, users can set the property to ``PREFER_PLUGIN``.
This instructs the plugin to use the integrated compiler when all the following conditions are met:
- The library is present
- The compiler supports the current platform ``or`` there is no platform detected (offline compilation)
- Compatibility is maintained between the current compiler version and all drivers released for the platform ``or`` there is no platform detected (offline compilation)
- Note: On Meteor Lake (3720), when the property is set to ``PREFER_PLUGIN``, the plugin will fall back to ``Compiler-in-Driver`` because
the compiler library integrated in the plugin may not be compatible with driver versions lower than v2565.
Users can set ``ov::intel_npu::compiler_type`` to ``PLUGIN`` to force ``Compiler-in-Plugin``, but the blob will fail to execute on incompatible drivers.

If any condition is not met, the plugin automatically falls back to using ``Compiler-In-Driver``.
The compiler type used to compile a model can be queried from the resulting ``CompiledModel`` by reading the ``ov::intel_npu::compiler_type`` property.

.. note::

Notes regarding on-device vs offline compilation:
- For on-device compilation with ``Compiler-In-Plugin``, the plugin is responsible for querying the platform information
from the driver and for passing the mandatory configs to the compiler (platform ID, available number of tiles, stepping information).
Such compiled models are compatible with all the drivers released for that platform.
- For offline compilation, users must explicitly set the ``ov::intel_npu::platform`` property to one of the supported values (see table above).
Setting extra properties during offline compilation may result in compiled models that cannot be executed
on SKUs with fewer resources or on drivers that do not support those features.
Example: Setting ``performance-hint-override=latency`` through ``ov::intel_npu::compilation_mode_params`` instructs the compiler
to use all available resources for the given platform. If ``ov::intel_npu::max_tiles`` is not provided,
the compiler falls back to a fixed lookup table embedded in the library to determine available resources, which might not be representative of all SKUs.

Limitations
#############################

* Currently, only models with static shapes are supported on NPU.

**Import/Export:**

Offline compilation and blob import is supported only for development purposes.
Pre-compiled models (blobs) are not recommended to be used in production.
Blob compatibility across different OpenVINO versions is not guaranteed.

Additional Resources
#############################

* `Working with NPUs in OpenVINO™ Notebook <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-npu>`__
* `Vision colorization Notebook <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/ddcolor-image-colorization>`__
