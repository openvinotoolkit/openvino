# CPU device {#openvino_docs_OV_UG_supported_plugins_CPU}

The CPU plugin is developed to achieve high performance inference of neural networks on Intel® x86-64 CPUs.
The plugin extensively use optimized DL operation implementations from the Intel® oneAPI Deep Neural Network Library (Intel® [oneDNN](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onednn.html)).
For an in-depth description of CPU plugin, see

- [CPU plugin developers documentation](https://github.com/openvinotoolkit/openvino/wiki/CPUPluginDevelopersDocs)

- [OpenVINO Runtime CPU plugin source files](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu/)


The CPU plugin is a part of the Intel® Distribution of OpenVINO™ toolkit.

## Device name
For the CPU plugin `"CPU"` device name is used. On multi-socket platforms, load balancing and memory usage distribution between NUMA nodes are handled automatically.   
In order to use CPU for inference the device name should be passed to `ov::Core::compile_model()` method:

@snippet snippets/cpu/compile_model.cpp compile_model_default

## Supported inference data types
CPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:
  - f32
  - i32
  - bf16
- Quantized data types:
  - u8
  - i8
  - u1

Selected precision of each primitive depends on the operation precision in IR, quantization primitives, and available hardware capabilities.
u1/u8/i8 data types are used for quantized operations only, i.e. those are not selected automatically for non-quantized operations.

See [low-precision optimization guide](@ref pot_docs_LowPrecisionOptimizationGuide) for more details on how to get quantized model.

> **NOTE**: Calculation results in u8/i8 precisions may be different between platforms with and without Intel® AVX512-VNNI extension support. Platforms that do not support VNNI have a known [saturation (overflow) issue](@ref pot_saturation_issue), which in some cases leads to reduced computational accuracy.

Default floating-point precision of a CPU primitive is f32, but on platforms that natively support bfloat16 calculations (have AVX512_BF16 extension) bfloat16 type is automatically used to achieve better performance (for details see [Using Bfloat16 Inference](../Bfloat16Inference.md)).
This means that to infer a model with bfloat16 precision no special actions is required, only CPU with native support for bfloat16.

[Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md) can be used to print out supported data types for all detected devices.
  
## Supported features

### Multi-device execution
If a machine has OpenVINO supported devices other than CPU (for example integrated GPU), then any supported model can be executed on CPU and all the other devices simultaneously.
This can be achieved by specifying `"MULTI:CPU,GPU.0"` as a target device in case of simultaneous usage of CPU and GPU.

@snippet snippets/cpu/compile_model.cpp compile_model_multi

See [Multi-device execution page](../multi_device.md) for more details.

### Multi-stream execution
If either `ov::num_streams(n_streams)` with `n_streams > 1` or `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` property is set for CPU plugin,
then multiple streams are created for the model. In case of CPU plugin each stream has its own host thread which means that incoming infer requests can be processed simultaneously.
Each stream is pinned to its own group of physical cores with respect to NUMA nodes physical memory usage to minimize overhead on data transfer between NUMA nodes.

See [optimization guide](@ref openvino_docs_deployment_optimization_guide_dldt_optimization_guide) for more details.

> **NOTE**: When it comes to latency, one needs to keep in mind that running only one stream on multi-socket platform may introduce additional overheads on data transfer between NUMA nodes.
> In that case it is better to run inference on one socket (please see [deployment optimization guide (additional configurations)](@ref openvino_docs_deployment_optimization_guide_dldt_optimization_guide_additional) for details).

### Dynamic shapes
CPU plugin provides full support for models with dynamic shapes. 
But from the performance standpoint, it should be understood that the more degrees of freedom we have, the more difficult it is to achieve the best performance.
The most flexible configuration is the fully undefined shape, when we do not apply any constraints to the shape dimensions. 

@snippet snippets/cpu/dynamic_shape.cpp undefined_shape

In such configuration we will have high memory consumption since we can not estimate the total memory amount that can be allocated in advance and effectively reused.
To reduce memory consumption through memory reuse, and as a result achieve better cache locality, which in its turn leads to better inference performance, it is better to use dynamic shapes with defined upper bounds.

@snippet snippets/cpu/dynamic_shape.cpp defined_upper_bound

> **NOTE**: Some runtime optimizations works better if the model shapes are known in advance. This means that for the best performance, it is better to use static shapes, of course if it is applicable to the specific problem.

See [dynamic shapes guide](../ov_dynamic_shapes.md) for more details.

### Preprocessing acceleration
CPU plugin supports a full set of the preprocessing operations, providing high performance implementations for them.

See [preprocessing API guide](../preprocessing_overview.md) for more details.

The CPU plugin precision conversion operation implementation supports the following element types:
- bf16
- f16
- f32
- f64
- i8
- i16
- i32
- i64
- u8
- u16
- u32
- u64
- boolean

### Models caching
CPU plugin supports Import/Export network capability. If the model caching is enabled via common OpenVINO `ov::cache_dir` property, the plugin will automatically create a cached blob inside the specified directory during model compilation.
This cached blob contains some intermediate representation of the network that it has after common runtime optimizations and low precision transformations.
The next time the model is compiled, the cached representation will be loaded to the plugin instead of the initial IR, so the aforementioned transformation steps will be skipped.
These transformations take a significant amount of time during model compilation, so caching this representation reduces time spent for subsequent compilations of the model,
thereby reducing first inference latency (FIL).

See [model caching overview](@ref openvino_docs_IE_DG_Model_caching_overview) for more details.

### Extensibility
CPU plugin supports fallback on `ov::Op` reference implementation if the plugin do not have its own implementation for such operation.
That means that [OpenVINO™ Extensibility Mechanism](@ref openvino_docs_Extensibility_UG_Intro) can be used for the plugin extension as well.
To enable fallback on a custom operation implementation, one have to re-implement `ov::Op::evaluate` method in the derived operation class (see [custom OpenVINO™ operations](@ref openvino_docs_Extensibility_UG_add_openvino_ops) for details).

### Stateful models
CPU plugin supports stateful models without any limitations.

See [stateful models guide](@ref openvino_docs_IE_DG_network_state_intro) for details.

## Supported properties
The plugin supports the properties listed below.

### Read-write properties
All parameters must be set before calling `ov::Core::compile_model()` in order to take effect or passed as additional argument to `ov::Core::compile_model()`

- ov::enable_profiling
- ov::hint::inference_precision
- ov::hint::performance_mode
- ov::hint::num_request
- ov::num_streams
- ov::affinity
- ov::inference_num_threads


### Read-only properties
- ov::cache_dir
- ov::supported_properties
- ov::available_devices
- ov::range_for_async_infer_requests
- ov::range_for_streams
- ov::device::full_name
- ov::device::capabilities

## See Also
* [Supported Devices](Supported_Devices.md)
* [Optimization guide](@ref openvino_docs_optimization_guide_dldt_optimization_guide)
* [СPU plugin developers documentation](https://github.com/openvinotoolkit/openvino/wiki/CPUPluginDevelopersDocs)

