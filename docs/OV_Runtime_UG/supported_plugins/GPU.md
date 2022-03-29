# GPU device {#openvino_docs_OV_UG_supported_plugins_GPU}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API

@endsphinxdirective

The GPU plugin is OpenCL based plugin for inference of deep neural networks on Intel GPUs including integrated and discrete ones.
For an in-depth description of GPU plugin, see
- [GPU plugin developers documentation](https://github.com/openvinotoolkit/openvino/wiki/GPUPluginDevelopersDocs)
- [OpenVINO Runtime GPU plugin source files](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu/)
- [Accelerate Deep Learning Inference with Intel® Processor Graphics](https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics).

The GPU plugin is a part of the Intel® Distribution of OpenVINO™ toolkit.

See [GPU configuration page](@ref openvino_docs_install_guides_configurations_for_intel_gpu) for more details on how to configure machine to use GPU plugin.

## Device Naming Convention
* Devices are enumerated as `"GPU.X"` where `X={0, 1, 2,...}`. Only Intel® GPU devices are considered.
* If the system has an integrated GPU, it always has id=0 (`"GPU.0"`).
* Other GPUs have undefined order that depends on the GPU driver.
* `"GPU"` is an alias for `"GPU.0"`
* If the system doesn't have an integrated GPU, then devices are enumerated starting from 0.
* For GPUs with multi-tile architecture (multiple sub-devices in OpenCL terms) specific tile may be addresed as `"GPU.X.Y"` where `X,Y={0, 1, 2,...}`, `X` - id of the GPU device, `Y` - id of the tile within device `X`

For demonstration purposes, see the [Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md) that can print out the list of available devices with associated indices. Below is an example output (truncated to the device names only):

```sh
./hello_query_device
Available devices:
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
...
    Device: HDDL
```

Then device name can be passed to `ov::Core::compile_model()` method:

@sphinxtabset

@sphinxtab{Running on default device}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_default_gpu
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_default_gpu
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Running on specific GPU}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_gpu_with_id
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_gpu_with_id
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Running on specific tile}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_gpu_with_id_and_tile
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_gpu_with_id_and_tile
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

## Supported inference data types
GPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:
  - f32
  - f16
- Quantized data types:
  - u8
  - i8
  - u1

Selected precision of each primitive depends on the operation precision in IR, quantization primitives, and available hardware capabilities.
u1/u8/i8 data types are used for quantized operations only, i.e. those are not selected automatically for non-quantized operations.
See [low-precision optimization guide](@ref pot_docs_LowPrecisionOptimizationGuide) for more details on how to get quantized model.

Floating-point precision of a GPU primitive is selected based on operation precision in IR except [compressed f16 IR form](../../MO_DG/prepare_model/FP16_Compression.md) which is executed in f16 precision.

> **NOTE**: Hardware acceleration for i8/u8 precision may be unavailable on some platforms. In that case model is executed in floating-point precision taken from IR. Hardware support of u8/i8 acceleration can be queried via `ov::device::capabilities` property.

[Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md) can be used to print out supported data types for all detected devices.

## Supported features

### Multi-device execution
If a machine has multiple GPUs (for example integrated GPU and discrete Intel GPU), then any supported model can be executed on all GPUs simultaneously.
This can be achieved by specifying `"MULTI:GPU.1,GPU.0"` as a target device.

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_multi
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_multi
@endsphinxtab

@endsphinxtabset

See [Multi-device execution page](../multi_device.md) for more details.

### Automatic batching
GPU plugin is capable of reporting `ov::max_batch_size` and `ov::optimal_batch_size` metrics with respect to the current hardware platform and model,
thus automatic batching is automatically enabled when `ov::optimal_batch_size` is > 1 and `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` is set.
Alternatively it can be enabled explicitly via the device notion, e.g. `"BATCH:GPU"`.

@sphinxtabset

@sphinxtab{Batching via BATCH plugin}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_batch_plugin
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_batch_plugin
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Bacthing via throughput hint}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_auto_batch
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_auto_batch
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

See [Automatic batching page](../automatic_batching.md) for more details.

### Multi-stream execution
If either `ov::num_streams(n_streams)` with `n_streams > 1` or `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` property is set for GPU plugin,
then multiple streams are created for the model. In case of GPU plugin each stream has its own host thread and associated OpenCL queue
which means that incoming infer requests can be processed simultaneously.

> **NOTE**: Simultaneous scheduling of kernels to different queues doesn't mean that the kernels are actually executed in parallel on GPU device. The actual behavior depends on the hardware architecture, and in some cases the execution may be serialized inside the GPU driver.

When multiple inferences of the same model need to be executed in parallel, multi-stream feature is preferrable over multiple instances of the model or application,
since implementation of streams in GPU plugin supports weights memory sharing across streams, thus memory consumption may be less comparing to the other approaches.

See [optimization guide](@ref openvino_docs_deployment_optimization_guide_dldt_optimization_guide) for more details.

### Dynamic shapes
GPU plugin supports dynamic shapes for batch dimension only (specified as 'N' in the [layouts terms](../layout_overview.md)) with fixed upper bound. Any other dynamic dimensions are unsupported. Internally GPU plugin creates
`log2(N)` (`N` - is an upper bound for batch dimension here) low-level execution graphs for batch sizes equal to powers of 2 to emulate dynamic behavior, so that incoming infer request with specific batch size is executed via minimal combination of internal networks.
For example, batch size 33 may be executed via 2 internal networks with batch size 32 and 1.

> **NOTE**: Such approach requires much more memory and overall model compilation time is significantly bigger comparing to static batch scenario.

The code snippet below demonstrates how to use dynamic batch in simple scenarios:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/dynamic_batch.cpp dynamic_batch
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/dynamic_batch.py dynamic_batch
@endsphinxtab

@endsphinxtabset

See [dynamic shapes guide](../ov_dynamic_shapes.md) for more details.

### Preprocessing acceleration
GPU plugin has the following additional preprocessing options:
- `ov::intel_gpu::memory_type::surface` and `ov::intel_gpu::memory_type::buffer` values for `ov::preprocess::InputTensorInfo::set_memory_type()` preprocessing method. These values are intended to be used to provide a hint for the plugin on the type of input Tensors that will be set in runtime to generate proper kernels.

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/preprocessing.cpp init_preproc
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/preprocessing.py init_preproc
@endsphinxtab

@endsphinxtabset

With such preprocessing GPU plugin will expect `ov::intel_gpu::ocl::ClImage2DTensor` (or derived) to be passed for each NV12 plane via `ov::InferRequest::set_tensor()` or `ov::InferRequest::set_tensors()` methods.

Refer to [RemoteTensor API](./GPU_RemoteTensor_API.md) for usage examples.

See [preprocessing API guide](../preprocessing_overview.md) for more details.

### Models caching
Cache for GPU plugin may be enabled via common OpenVINO `ov::cache_dir` property. GPU plugin implementation supports only compiled kernels caching,
thus all plugin specific model transformations are executed on each `ov::Core::compile_model()` call regardless `cache_dir` option, but since
the kernels compilation is a bottleneck in the model loading process, significant load time reduction can be achieved with `ov::cache_dir` property enabled.

See [Model caching overview page](../Model_caching_overview.md) for more details.

### Extensibility
See [GPU Extensibility](@ref openvino_docs_Extensibility_UG_GPU) page.

### GPU context and memory sharing via RemoteTensor API
See [RemoteTensor API of GPU Plugin](GPU_RemoteTensor_API.md).


## Supported properties
The plugin supports the properties listed below.

### Read-write properties
All parameters must be set before calling `ov::Core::compile_model()` in order to take effect or passed as additional argument to `ov::Core::compile_model()`

- ov::cache_dir
- ov::enable_profiling
- ov::hint::model_priority
- ov::hint::performance_mode
- ov::hint::num_requests
- ov::num_streams
- ov::compilation_num_threads
- ov::device::id
- ov::intel_gpu::hint::host_task_priority
- ov::intel_gpu::hint::queue_priority
- ov::intel_gpu::hint::queue_throttle
- ov::intel_gpu::enable_loop_unrolling

### Read-only properties
- ov::supported_properties
- ov::available_devices
- ov::range_for_async_infer_requests
- ov::range_for_streams
- ov::optimal_batch_size
- ov::max_batch_size
- ov::device::full_name
- ov::device::type
- ov::device::gops
- ov::device::capabilities
- ov::intel_gpu::device_total_mem_size
- ov::intel_gpu::uarch_version
- ov::intel_gpu::execution_units_count
- ov::intel_gpu::memory_statistics

## Limitations
In some cases GPU plugin may implicitly execute several primitives on CPU using internal implementations which may lead to increase of CPU utilization.
Below is the list of such operations:
- Proposal
- NonMaxSuppression
- DetectionOutput

The behavior depends on specific parameters of the operations and hardware configuration.


## GPU Performance Checklist: Summary <a name="gpu-checklist"></a>
Since the OpenVINO relies on the OpenCL&trade; kernels for the GPU implementation. Thus, many general OpenCL tips apply:
-	Prefer `FP16` inference precision over `FP32`, as the Model Optimizer can generate both variants and the `FP32` is default. Also, consider [int8 inference](../Int8Inference.md)
- 	Try to group individual infer jobs by using [automatic batching](../automatic_batching.md)
-	Consider [caching](../Model_caching_overview.md) to minimize model load time
-	If your application is simultaneously using the inference on the CPU or otherwise loads the host heavily, make sure that the OpenCL driver threads do not starve. You can use [CPU configuration options](./CPU.md) to limit number of inference threads for the CPU plugin.
-	Even in the GPU-only scenario, a GPU driver might occupy a CPU core with spin-looped polling for completion. If the _CPU_ utilization is a concern, consider the dedicated referenced in this document. Notice that this option might increase the inference latency, so consider combining with multiple GPU streams or [throughput performance hints](../performance_hints.md).
- When operating media inputs consider [remote tensors API of the GPU Plugin](./GPU_RemoteTensor_API.md).


## See Also
* [Supported Devices](Supported_Devices.md)
* [Optimization guide](@ref openvino_docs_optimization_guide_dldt_optimization_guide)
* [GPU plugin developers documentation](https://github.com/openvinotoolkit/openvino/wiki/GPUPluginDevelopersDocs)
