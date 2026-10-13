---
sidebar_position: 3
sidebar_label: "NPU Device"
description: "OpenVINO™ supports the Neural Processing Unit, a low-power processing device dedicated to running AI inference."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# NPU Device


The Neural Processing Unit is a low-power hardware solution, introduced with the
Intel® Core™ Ultra generation of CPUs (formerly known as Meteor Lake). It enables
you to offload certain neural network computation tasks
for more streamlined resource management.

NPU Plugin is now available through all relevant OpenVINO distribution channels.

NPU Plugin needs an NPU Driver to be installed on the system to execute a model.
Follow the instructions below to install the latest NPU drivers:

- [Windows driver](https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html)
- [Linux driver](https://github.com/intel/linux-npu-driver/releases)

:::note
The NPU compiler library was first introduced in the OpenVINO 2026.0 release package as a preview feature (Compiler-In-Plugin).
Starting with the 2026.1 release, Compiler-In-Plugin becomes the preferred compiler type used by the NPU Plugin.
Users can override the default compiler selection by setting `ov::intel_npu::compiler_type`.
:::

The plugin uses either the NPU compiler library included in the driver or
the compiler library included in the OpenVINO package to convert the OpenVINO specific
representation of the model into a proprietary format. The compiler performs platform specific
optimizations in order to efficiently schedule the execution of layers and memory transactions
on various NPU hardware submodules.

To use NPU for inference, pass the device name to the `ov::Core::compile_model()` method:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()
      compiled_model = core.compile_model(model, "NPU")
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
      ov::Core core;
      auto model = core.read_model("model.xml");
      auto compiled_model = core.compile_model(model, "NPU");
  ```
  </TabItem>
</Tabs>

## Model Caching

Model Caching helps reduce application startup delays by exporting and reusing the compiled
model automatically. The following two compilation-related metrics are crucial in this area:

**First Ever Inference Latency (FEIL)**  
Measures all steps required to compile and execute a model on the device for the
first time. It includes model compilation time, the time required to load and
initialize the model on the device and the first inference execution.  
**First Inference Latency (FIL)**  
Measures the time required to load and initialize the pre-compiled model on the
device and the first inference execution.

### UMD Dynamic Model Caching

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
ov::intel\_npu::bypass\_umd\_caching (NPU\_BYPASS\_UMD\_CACHE) to true at compilation. (default value is false)

### OpenVINO Model Caching

OpenVINO Model Caching is a common mechanism for all OpenVINO device plugins and
can be enabled by setting the `ov::cache_dir` property. This way, the UMD model
caching is automatically bypassed by the NPU plugin, which means the model
will only be stored in the OpenVINO cache after compilation. When a cache hit
occurs for subsequent compilation requests, the plugin will import the model
instead of recompiling it.

For more details about OpenVINO model caching, see the
[Model Caching Overview](../../optimize-inference/optimizing-latency/model-caching-overview.md).

**Supported Inference Data Types**  
The NPU plugin supports the following data types as inference precision of internal primitives:  
Floating-point data types: F32, F16  
Quantized data types: U8 (quantized models may be INT8 or mixed FP16-INT8)  
Computation precision for the HW is FP16.  
  
For more details on how to get a quantized model, refer to the
[Model Optimization guide](../../../model-optimization/index.md) and
[NNCF tool quantization guide](../../../model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.md).

## Supported Features and properties

The NPU device is currently supported by AUTO inference modes
(HETERO execution is partially supported, for certain models).

**Supported Properties:**

<Tabs>
  <TabItem value="read-write-properties" label="Read-write properties">
  ```text
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
  ov::intel_npu::bypass_umd_caching
  ov::intel_npu::defer_weights_load
  ov::intel_npu::run_inferences_sequentially
  ov::intel_npu::disable_idle_memory_prunning
  ov::intel_npu::compiler_type
  ov::intel_npu::enable_strides_for
  ```
  </TabItem>
  <TabItem value="read-only-properties" label="Read-only properties">
  ```text
  ov::supported_properties
  ov::available_devices
  ov::optimal_number_of_infer_requests
  ov::range_for_async_infer_requests
  ov::range_for_streams
  ov::num_streams
  ov::execution_devices
  ov::runtime_requirements
  ov::compatibility_check
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
  ov::intel_npu::max_tiles
  ```
  </TabItem>
  <TabItem value="write-only-properties" label="Write-only properties">
  ```text
  ov::cache_encryption_callbacks
  ```
  </TabItem>
</Tabs>

:::note
The optimum number of inference requests returned by the plugin
based on the performance mode is **4 for THROUGHPUT** and **1 for LATENCY**.
The default mode for the NPU device is LATENCY.
:::

**ov::intel\_npu::compilation\_mode\_params**

`ov::intel_npu::compilation_mode_params` is an NPU-specific property that allows
control of model compilation for NPU.

:::note
The functionality is in experimental stage currently, can be a subject for
deprecation and may be replaced with generic OV API in future OV releases.
:::

Following configuration options are supported:

**optimization-level**

Defines an optimization effort hint to the compiler.

| **Value** | **Description** |
| --- | --- |
| 0 | Reduced subset of optimization passes. May result in smaller compile time. |
| 1 | **Default.** Balanced performance/compile time. |
| 2 | Prioritize performance over compile time that may be an issue. |

**performance-hint-override**

The LATENCY mode can be overridden by specifying `ov::hint::performance_mode`
Has no effect for other `ov::hint::PerformanceMode` hints.

| **Value** | **Description** |
| --- | --- |
| efficiency | **Default.** Balanced performance and power consumption. |
| latency | Prioritize performance over power efficiency. |

Usage example:

```text
map<str, str> config = {ov::intel_npu::compilation_mode_params.name(), ov::Any("optimization-level=1 performance-hint-override=latency")};

compile_model(model, config);
```

**ov::intel\_npu::turbo**

The turbo mode, where available, provides a hint to the system to maintain the
maximum NPU frequency and memory throughput within the platform TDP limits.
The turbo mode is not recommended for sustainable workloads due to higher power
consumption and potential impact on other compute resources.

```text
core.set_property("NPU", ov::intel_npu::turbo(true));
```

or

```text
core.compile_model(ov_model, "NPU", {ov::intel_npu::turbo(true)});
```

:::note
NPU\_TURBO usage may cause higher compile time, memory footprint,
affect workload latency and compatibility issues with older NPU drivers
:::

**ov::intel\_npu::max\_tiles and ov::intel\_npu::tiles**

For on-device compilation, the plugin queries the driver for the available number of tiles and sets `ov::intel_npu::max_tiles`.

`ov::intel_npu::max_tiles` is a read-only property, and it will not be listed as supported in cases where no device is present.

Note that `ov::intel_npu::max_tiles` represents the maximum number of tiles available,
but the compiler may target a lower number of tiles depending on other properties.
Users can set `ov::intel_npu::tiles` to override the number of tiles selected by the compiler based on other properties.

:::note
When setting `ov::intel_npu::tiles`, users must ensure that the value does not
exceed `ov::intel_npu::max_tiles`. Any tile count other than 1 may impact
cross-device compatibility if it is not explicitly validated against the target
device's `ov::intel_npu::max_tiles` value.
:::

**ov::intel\_npu::compiler\_type**

This property allows users to override the default compiler type selected by the plugin.
By default, the NPU Plugin behavior corresponds to the `PREFER_PLUGIN` setting.
In this mode, the integrated compiler (`Compiler-In-Plugin`) is used when all the following conditions are met:

- The library is present
- The compiler supports the current platform `or` there is no platform detected (offline compilation)
- Compatibility is maintained between the current compiler version and all drivers released for the platform `or` there is no platform detected (offline compilation)

Note: On Meteor Lake (3720), when the property is set to `PREFER_PLUGIN` (by default), the plugin will fall back to `Compiler-in-Driver` because
the compiler library integrated in the plugin may not be compatible with driver versions lower than v2565.
Users can set `ov::intel_npu::compiler_type` to `PLUGIN` to force `Compiler-in-Plugin`, but the blob will fail to execute on incompatible drivers.

If any condition is not met, the plugin automatically falls back to using `Compiler-In-Driver`.
The compiler type used to compile a model can be queried from the resulting `CompiledModel` by reading the `ov::intel_npu::compiler_type` property.

:::note
For on-device compilation with `Compiler-In-Plugin`, the plugin is responsible for querying the platform information
from the driver and for passing the mandatory configs to the compiler (platform ID, available number of tiles, stepping information).
Such compiled models are compatible with all the drivers released for that platform.

For offline compilation, users must explicitly set the `ov::intel_npu::platform` property to one of the supported values (see table above).
Setting extra properties during offline compilation may result in compiled models that cannot be executed on SKUs with fewer resources or on drivers that do not support those features.
:::

Example: Setting `performance-hint-override=latency` through `ov::intel_npu::compilation_mode_params` instructs the compiler
to use all available resources for the given platform. If `ov::intel_npu::max_tiles` is not provided,
the compiler falls back to a fixed lookup table embedded in the library to determine available resources, which might not be representative of all SKUs.

**ov::cache\_encryption\_callbacks**

Enables blob encryption and decryption using user-provided callbacks. This is a write-only property that accepts a struct with encryption and decryption callback functions.
Can be used for both model caching and user requested export or import of compiled models (blobs) scenarios.

:::note
If user compiles a model using `Compiler-In-Driver`, currently there is no secure compilation available in driver until `32.0.100.4724` (inclusively) and an exception will be thrown
for older drivers during compilation if encryption callbacks were set.
:::

Usage example:

:::note
Will set OV provided `codec_xor` utility encryption and decryption at plugin level, every compiled model created will inherit these callbacks if not changed explicitly.
:::

```text
core.set_property("NPU", ov::cache_encryption_callbacks(ov::EncryptionCallbacks(ov::util::codec_xor, ov::util::codec_xor)));

core.compile_model(ov_model, "NPU").export_model(encrypted_blob_stream);

core.import_model(encrypted_blob_stream, "NPU");
```

:::note
Also force decryption function by providing it as a property to `import_model` API
:::

```text
core.import_model(encrypted_blob_stream, "NPU", {ov::cache_encryption_callbacks(ov::EncryptionCallbacks{nullptr, ov::util::codec_xor})});
```

:::note
Force other encryption and decryption callbacks for compiled model below
:::

```text
auto compile_model = core.compile_model(ov_model, "NPU", {ov::cache_encryption_callbacks(ov::EncryptionCallbacks{[](const std::string& str) {
    auto custom_encryption = str;
    // user encryption logic here...
    return custom_encryption;
}, nullptr})});

compiled_model.export_model(custom_encrypted_blob_stream);
```

or

```text
auto compile_model = core.compile_model(ov_model, "NPU");

compiled_model.set_property(ov::cache_encryption_callbacks(ov::EncryptionCallbacks{[](const std::string& str) {
    auto custom_encryption = str;
    // user encryption logic here...
    return custom_encryption;
}, nullptr}));

compiled_model.export_model(custom_encrypted_blob_stream);
```

**ov::runtime\_requirements**

This property returns a plugin-specific requirements string from a compiled model.

```text
auto requirements = compiled_model.get_property(ov::runtime_requirements);
```

:::note
Particular models might not support the `ov::runtime_requirements` property. For example,
models compiled with a driver (compiler-in-driver) that does not support Level Zero
version 1.16, or models compiled with `ENABLE_WEIGHTLESS=true`, which is only a
preview feature for now. Before reading the property, the application is recommended
to check whether it is present in the list of supported properties for that model.
Reading an unsupported property will throw.
:::

**ov::compatibility\_check**

This property can be queried with a requirements string to check whether
the described model is compatible with the current runtime.

```text
auto compat = core.get_property("NPU", ov::compatibility_check, {{ov::runtime_requirements.name(), requirements}});
```


## Dynamic Shapes {#npu-dynamic-shapes}

Dynamic shape support on NPU is a preview (experimental) feature. The general description of what dynamic
shapes are and how they are used can be found in the
[dynamic shapes guide](../../model-input-output/dynamic-shapes/index.md).

Dynamic shapes are supported starting from the NPU40XX (Lunar Lake generation) platform and require
[Windows NPU driver 32.0.100.4621](https://www.intel.com/content/www/us/en/download/794734/914683/intel-npu-driver-windows.html)
or [Linux NPU driver 1.30](https://github.com/intel/linux-npu-driver/releases#release-v1.30.0) (or newer).
Only bounded dynamic shapes are supported, which means each dynamic dimension must specify an upper bound,
for example `ov::Dimension(1, 512)`.

Dynamic shapes are supported only with the `Compiler-In-Plugin` compiler type; the `Compiler-In-Driver`
path does not support dynamic shapes. See the `ov::intel_npu::compiler_type` property description above for
details on selecting the compiler type. The driver versions listed above are required because dynamic shape
support depends on the driver's support for IO strides, which is available only starting from those versions.

:::note
As a preview feature, dynamic shape support currently has the following limitations:

- Support is limited to vision models and has been validated on super resolution models (e.g., ESPCN).
- Only bounded dynamic shapes are supported at this time. Full dynamic shape support, including
  unbounded shapes and broader model coverage, is planned for future releases.
:::

## Limitations

- Dynamic shape support is a preview feature. See the [Dynamic Shapes](#npu-dynamic-shapes) section
  for supported platforms, drivers, and constraints.

**Import/Export:**

Offline compilation and blob import is supported only for development purposes.
Pre-compiled models (blobs) are not recommended to be used in production.
Blob compatibility across different OpenVINO versions is not guaranteed.

## Additional Resources

- [Working with NPUs in OpenVINO™ Notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-npu)
- [Vision colorization Notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/ddcolor-image-colorization)
