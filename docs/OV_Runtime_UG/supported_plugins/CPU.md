# CPU device {#openvino_docs_OV_UG_supported_plugins_CPU}

The CPU plugin is developed to achieve high performance inference of neural networks on Intel® x86-64 CPUs.
For an in-depth description of CPU plugin, see

- [CPU plugin developers documentation](https://github.com/openvinotoolkit/openvino/wiki/CPUPluginDevelopersDocs)

- [OpenVINO Runtime CPU plugin source files](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_cpu/)


The CPU plugin is a part of the Intel® Distribution of OpenVINO™ toolkit.

## Device name
For the CPU plugin `"CPU"` device name is used, and even though there can be more than one socket on a platform, from the plugin's point of view, there is only one `"CPU"` device.
On multi-socket platforms, load balancing and memory usage distribution between NUMA nodes are handled automatically.   
In order to use CPU for inference the device name should be passed to `ov::Core::compile_model()` method:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/compile_model.cpp compile_model_default
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/compile_model.py compile_model_default
@endsphinxtab

@endsphinxtabset

## Supported inference data types
CPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:
  - f32
  - bf16
- Integer data types:
  - i32
- Quantized data types:
  - u8
  - i8
  - u1
  
[Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md) can be used to print out supported data types for all detected devices.

### Quantized data types specifics

Selected precision of each primitive depends on the operation precision in IR, quantization primitives, and available hardware capabilities.
u1/u8/i8 data types are used for quantized operations only, i.e. those are not selected automatically for non-quantized operations.

See [low-precision optimization guide](@ref pot_docs_LowPrecisionOptimizationGuide) for more details on how to get quantized model.

> **NOTE**: Platforms that do not support Intel® AVX512-VNNI have a known "saturation issue" which in some cases leads to reduced computational accuracy for u8/i8 precision calculations.
> See [saturation (overflow) issue section](@ref pot_saturation_issue) to get more information on how to detect such issues and possible workarounds.

### Floating point data types specifics

Default floating-point precision of a CPU primitive is f32. To support f16 IRs the plugin internally converts all the f16 values to f32 and all the calculations are performed using native f32 precision.
On platforms that natively support bfloat16 calculations (have AVX512_BF16 extension) bf16 type is automatically used instead of f32 to achieve better performance, thus no special steps are required to run a model with bf16 precision.
See the [BFLOAT16 – Hardware Numerics Definition white paper](https://software.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf) for more details about bfloat16 format.

Using bf16 precision provides the following performance benefits:

- Faster multiplication of two bfloat16 numbers because of shorter mantissa of the bfloat16 data.
- Reduced memory consumption since bfloat16 data size is two times less than 32-bit float. 

To check if the CPU device can support the bfloat16 data type use the [query device properties interface](./config_properties.md) to query ov::device::capabilities property, which should contain `BF16` in the list of CPU capabilities:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/Bfloat16Inference0.cpp part0
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/Bfloat16Inference.py part0
@endsphinxtab

@endsphinxtabset

In case if the model was converted to bf16, ov::hint::inference_precision is set to ov::element::bf16 and can be checked via ov::CompiledModel::get_property call. The code below demonstrates how to get the element type:

@snippet snippets/cpu/Bfloat16Inference1.cpp part1

To infer the model in f32 precision instead of bf16 on targets with native bf16 support, set the ov::hint::inference_precision to ov::element::f32.

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/Bfloat16Inference2.cpp part2
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/Bfloat16Inference.py part2
@endsphinxtab

@endsphinxtabset

Bfloat16 software simulation mode is available on CPUs with Intel® AVX-512 instruction set that do not support the native `avx512_bf16` instruction. This mode is used for development purposes and it does not guarantee good performance.
To enable the simulation, one have to explicitly set ov::hint::inference_precision to ov::element::bf16.

> **NOTE**: An exception is thrown in case of setting ov::hint::inference_precision to ov::element::bf16 on CPU without native bfloat16 support or bfloat16 simulation mode.

> **NOTE**: Due to the reduced mantissa size of the bfloat16 data type, the resulting bf16 inference accuracy may differ from the f32 inference, especially for models that were not trained using the bfloat16 data type. If the bf16 inference accuracy is not acceptable, it is recommended to switch to the f32 precision.
  
## Supported features

### Multi-device execution
If a machine has OpenVINO supported devices other than CPU (for example integrated GPU), then any supported model can be executed on CPU and all the other devices simultaneously.
This can be achieved by specifying `"MULTI:CPU,GPU.0"` as a target device in case of simultaneous usage of CPU and GPU.

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/compile_model.cpp compile_model_multi
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/compile_model.py compile_model_multi
@endsphinxtab

@endsphinxtabset

See [Multi-device execution page](../multi_device.md) for more details.

### Multi-stream execution
If either `ov::num_streams(n_streams)` with `n_streams > 1` or `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` property is set for CPU plugin,
then multiple streams are created for the model. In case of CPU plugin each stream has its own host thread which means that incoming infer requests can be processed simultaneously.
Each stream is pinned to its own group of physical cores with respect to NUMA nodes physical memory usage to minimize overhead on data transfer between NUMA nodes.

See [optimization guide](@ref openvino_docs_deployment_optimization_guide_dldt_optimization_guide) for more details.

> **NOTE**: When it comes to latency, one needs to keep in mind that running only one stream on multi-socket platform may introduce additional overheads on data transfer between NUMA nodes.
> In that case it is better to use ov::hint::PerformanceMode::LATENCY performance hint (please see [performance hints overview](@ref openvino_docs_OV_UG_Performance_Hints) for details).

### Dynamic shapes
CPU plugin provides full functional support for models with dynamic shapes in terms of the opset coverage.

> **NOTE**: CPU plugin does not support tensors with dynamically changing rank. In case of an attempt to infer a model with such tensors, an exception will be thrown.

Dynamic shapes support introduce some additional overheads on memory management and may limit internal runtime optimizations.
The more degrees of freedom we have, the more difficult it is to achieve the best performance.
The most flexible configuration is the fully undefined shape, when we do not apply any constraints to the shape dimensions, which is the most convenient approach.
But reducing the level of uncertainty will bring performance gains.
We can reduce memory consumption through memory reuse, and as a result achieve better cache locality, which in its turn leads to better inference performance, if we explicitly set dynamic shapes with defined upper bounds.

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/dynamic_shape.cpp defined_upper_bound
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/dynamic_shape.py defined_upper_bound
@endsphinxtab

@endsphinxtabset

> **NOTE**: Using fully undefined shapes may result in significantly higher memory consumption compared to inferring the same model with static shapes.
> If the memory consumption is unacceptable but dynamic shapes are still required, one can reshape the model using shapes with defined upper bound to reduce memory footprint.

Some runtime optimizations works better if the model shapes are known in advance.
Therefore, if the input data shape is not changed between inference calls, it is recommended to use a model with static shapes or reshape the existing model with the static input shape to get the best performance.

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/dynamic_shape.cpp static_shape
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/dynamic_shape.py static_shape
@endsphinxtab

@endsphinxtabset

See [dynamic shapes guide](../ov_dynamic_shapes.md) for more details.

### Preprocessing acceleration
CPU plugin supports a full set of the preprocessing operations, providing high performance implementations for them.

See [preprocessing API guide](../preprocessing_overview.md) for more details.

@sphinxdirective
.. dropdown:: The CPU plugin support for handling tensor precision conversion is limited to the following ov::element types:

    * bf16
    * f16
    * f32
    * f64
    * i8
    * i16
    * i32
    * i64
    * u8
    * u16
    * u32
    * u64
    * boolean
@endsphinxdirective

### Models caching
CPU plugin supports Import/Export network capability. If the model caching is enabled via common OpenVINO™ `ov::cache_dir` property, the plugin will automatically create a cached blob inside the specified directory during model compilation.
This cached blob contains some intermediate representation of the network that it has after common runtime optimizations and low precision transformations.
The next time the model is compiled, the cached representation will be loaded to the plugin instead of the initial IR, so the aforementioned transformation steps will be skipped.
These transformations take a significant amount of time during model compilation, so caching this representation reduces time spent for subsequent compilations of the model,
thereby reducing first inference latency (FIL).

See [model caching overview](@ref openvino_docs_IE_DG_Model_caching_overview) for more details.

### Extensibility
CPU plugin supports fallback on `ov::Op` reference implementation if the plugin do not have its own implementation for such operation.
That means that [OpenVINO™ Extensibility Mechanism](@ref openvino_docs_Extensibility_UG_Intro) can be used for the plugin extension as well.
To enable fallback on a custom operation implementation, one have to override `ov::Op::evaluate` method in the derived operation class (see [custom OpenVINO™ operations](@ref openvino_docs_Extensibility_UG_add_openvino_ops) for details).

> **NOTE**: At the moment, custom operations with internal dynamism (when the output tensor shape can only be determined as a result of performing the operation) are not supported by the plugin.

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

## External dependencies
For some performance-critical DL operations, the CPU plugin uses optimized implementations from the oneAPI Deep Neural Network Library ([oneDNN](https://github.com/oneapi-src/oneDNN)).

@sphinxdirective
.. dropdown:: The following operations are implemented using primitives from the OneDNN library:

    * AvgPool
    * Concat
    * Convolution
    * ConvolutionBackpropData
    * GroupConvolution
    * GroupConvolutionBackpropData
    * GRUCell
    * GRUSequence
    * LRN
    * LSTMCell
    * LSTMSequence
    * MatMul
    * MaxPool
    * RNNCell
    * RNNSequence
    * SoftMax
@endsphinxdirective

## See Also
* [Supported Devices](Supported_Devices.md)
* [Optimization guide](@ref openvino_docs_optimization_guide_dldt_optimization_guide)
* [СPU plugin developers documentation](https://github.com/openvinotoolkit/openvino/wiki/CPUPluginDevelopersDocs)
