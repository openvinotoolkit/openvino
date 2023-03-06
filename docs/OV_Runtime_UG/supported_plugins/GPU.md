# GPU Device {#openvino_docs_OV_UG_supported_plugins_GPU}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API

@endsphinxdirective

The GPU plugin is an OpenCL based plugin for inference of deep neural networks on Intel GPUs, both integrated and discrete ones.
For an in-depth description of the GPU plugin, see:
- [GPU plugin developers documentation](https://github.com/openvinotoolkit/openvino/wiki/GPUPluginDevelopersDocs)
- [OpenVINO Runtime GPU plugin source files](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu/)
- [Accelerate Deep Learning Inference with Intel® Processor Graphics](https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics).

The GPU plugin is a part of the Intel® Distribution of OpenVINO™ toolkit. For more information on how to configure a system to use it, see the [GPU configuration](@ref openvino_docs_install_guides_configurations_for_intel_gpu).

## Device Naming Convention
* Devices are enumerated as `GPU.X`, where `X={0, 1, 2,...}` (only Intel® GPU devices are considered).
* If the system has an integrated GPU, its `id` is always 0 (`GPU.0`).
* The order of other GPUs is not predefined and depends on the GPU driver.
* The `GPU` is an alias for `GPU.0`.
* If the system does not have an integrated GPU, devices are enumerated, starting from 0.
* For GPUs with multi-tile architecture (multiple sub-devices in OpenCL terms), a specific tile may be addressed as `GPU.X.Y`, where `X,Y={0, 1, 2,...}`, `X` - id of the GPU device, `Y` - id of the tile within device `X`

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
    Device: GNA
```

Then, device name can be passed to the `ov::Core::compile_model()` method:

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

## Supported Inference Data Types
The GPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:
  - f32
  - f16
- Quantized data types:
  - u8
  - i8
  - u1

Selected precision of each primitive depends on the operation precision in IR, quantization primitives, and available hardware capabilities.
The `u1`/`u8`/`i8` data types are used for quantized operations only, which means that they are not selected automatically for non-quantized operations.
For more details on how to get a quantized model, refer to the [Model Optimization guide](@ref openvino_docs_model_optimization_guide).

Floating-point precision of a GPU primitive is selected based on operation precision in the OpenVINO IR, except for the [compressed f16 OpenVINO IR form](../../MO_DG/prepare_model/FP16_Compression.md), which is executed in the `f16` precision.

> **NOTE**: Hardware acceleration for `i8`/`u8` precision may be unavailable on some platforms. In such cases, a model is executed in the floating-point precision taken from IR. Hardware support of `u8`/`i8` acceleration can be queried via the `ov::device::capabilities` property.

[Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md) can be used to print out the supported data types for all detected devices.

## Supported Features

The GPU plugin supports the following features:

### Multi-device Execution
If a system has multiple GPUs (for example, an integrated and a discrete Intel GPU), then any supported model can be executed on all GPUs simultaneously.
It is done by specifying `MULTI:GPU.1,GPU.0` as a target device.

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_multi
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_multi
@endsphinxtab

@endsphinxtabset

For more details, see the [Multi-device execution](../multi_device.md).

### Automatic Batching
The GPU plugin is capable of reporting `ov::max_batch_size` and `ov::optimal_batch_size` metrics with respect to the current hardware
platform and model. Therefore, automatic batching is enabled by default when `ov::optimal_batch_size` is `> 1` and `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` is set.
Alternatively, it can be enabled explicitly via the device notion, for example `BATCH:GPU`.

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

@sphinxtab{Batching via throughput hint}

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

For more details, see the [Automatic batching](../automatic_batching.md).

### Multi-stream Execution
If either the `ov::num_streams(n_streams)` with `n_streams > 1` or the `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` property is set for the GPU plugin,
multiple streams are created for the model. In the case of GPU plugin each stream has its own host thread and an associated OpenCL queue
which means that the incoming infer requests can be processed simultaneously.

> **NOTE**: Simultaneous scheduling of kernels to different queues does not mean that the kernels are actually executed in parallel on the GPU device. The actual behavior depends on the hardware architecture and in some cases the execution may be serialized inside the GPU driver.

When multiple inferences of the same model need to be executed in parallel, the multi-stream feature is preferred to multiple instances of the model or application.
The reason for this is that the implementation of streams in the GPU plugin supports weight memory sharing across streams, thus, memory consumption may be lower, compared to the other approaches.

For more details, see the [optimization guide](@ref openvino_docs_deployment_optimization_guide_dldt_optimization_guide).

### Dynamic Shapes
The GPU plugin supports dynamic shapes for batch dimension only (specified as `N` in the [layouts terms](../layout_overview.md)) with a fixed upper bound. Any other dynamic dimensions are unsupported. Internally, GPU plugin creates
`log2(N)` (`N` - is an upper bound for batch dimension here) low-level execution graphs for batch sizes equal to powers of 2 to emulate dynamic behavior, so that incoming infer request with a specific batch size is executed via a minimal combination of internal networks.
For example, batch size 33 may be executed via 2 internal networks with batch size 32 and 1.

> **NOTE**: Such approach requires much more memory and the overall model compilation time is significantly longer, compared to the static batch scenario.

The code snippet below demonstrates how to use dynamic batching in simple scenarios:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/dynamic_batch.cpp dynamic_batch
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/dynamic_batch.py dynamic_batch
@endsphinxtab

@endsphinxtabset

For more details, see the [dynamic shapes guide](../ov_dynamic_shapes.md).

### Preprocessing Acceleration
The GPU plugin has the following additional preprocessing options:
- The `ov::intel_gpu::memory_type::surface` and `ov::intel_gpu::memory_type::buffer` values for the `ov::preprocess::InputTensorInfo::set_memory_type()` preprocessing method. These values are intended to be used to provide a hint for the plugin on the type of input Tensors that will be set in runtime to generate proper kernels.

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/preprocessing_nv12_two_planes.cpp init_preproc
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/preprocessing_nv12_two_planes.py init_preproc
@endsphinxtab

@endsphinxtabset

With such preprocessing, GPU plugin will expect `ov::intel_gpu::ocl::ClImage2DTensor` (or derived) to be passed for each NV12 plane via `ov::InferRequest::set_tensor()` or `ov::InferRequest::set_tensors()` methods.

For usage examples, refer to the [RemoteTensor API](./GPU_RemoteTensor_API.md).

For more details, see the [preprocessing API](../preprocessing_overview.md).

### Model Caching
Cache for the GPU plugin may be enabled via the common OpenVINO `ov::cache_dir` property. GPU plugin implementation supports only caching of compiled kernels, so all plugin-specific model transformations are executed on each `ov::Core::compile_model()` call regardless of the `cache_dir` option.
Still, since kernel compilation is a bottleneck in the model loading process, a significant load time reduction can be achieved with the `ov::cache_dir` property enabled.

> **NOTE**: Full model caching support is currently implemented as a preview feature. To activate it, set the OV_GPU_CACHE_MODEL environment variable to 1.

For more details, see the [Model caching overview](../Model_caching_overview.md).

### Extensibility
For information on this subject, see the [GPU Extensibility](@ref openvino_docs_Extensibility_UG_GPU).

### GPU Context and Memory Sharing via RemoteTensor API
For information on this subject, see the [RemoteTensor API of GPU Plugin](GPU_RemoteTensor_API.md).


## Supported Properties
The plugin supports the properties listed below.

### Read-write properties
All parameters must be set before calling `ov::Core::compile_model()` in order to take effect or passed as additional argument to `ov::Core::compile_model()`.

- ov::cache_dir
- ov::enable_profiling
- ov::hint::model_priority
- ov::hint::performance_mode
- ov::hint::execution_mode
- ov::hint::num_requests
- ov::inference_precision
- ov::num_streams
- ov::compilation_num_threads
- ov::device::id
- ov::intel_gpu::hint::host_task_priority
- ov::intel_gpu::hint::queue_priority
- ov::intel_gpu::hint::queue_throttle
- ov::intel_gpu::enable_loop_unrolling

### Read-only Properties
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
In some cases, the GPU plugin may implicitly execute several primitives on CPU using internal implementations, which may lead to an increase in CPU utilization.
Below is a list of such operations:
- Proposal
- NonMaxSuppression
- DetectionOutput

The behavior depends on specific parameters of the operations and hardware configuration.

## GPU Performance Checklist: Summary <a name="gpu-checklist"></a>
Since OpenVINO relies on the OpenCL kernels for the GPU implementation, many general OpenCL tips apply:
-	Prefer `FP16` inference precision over `FP32`, as Model Optimizer can generate both variants, and the `FP32` is the default. Also, consider using the [Post-training Optimization Tool](https://docs.openvino.ai/latest/pot_introduction.html).
- Try to group individual infer jobs by using [automatic batching](../automatic_batching.md).
-	Consider [caching](../Model_caching_overview.md) to minimize model load time.
-	If your application performs inference on the CPU alongside the GPU, or otherwise loads the host heavily, make sure that the OpenCL driver threads do not starve. [CPU configuration options](./CPU.md) can be used to limit the number of inference threads for the CPU plugin.
-	Even in the GPU-only scenario, a GPU driver might occupy a CPU core with spin-loop polling for completion. If CPU load is a concern, consider the dedicated `queue_throttle` property mentioned previously. Note that this option may increase inference latency, so consider combining it with multiple GPU streams or [throughput performance hints](../performance_hints.md).
- When operating media inputs, consider [remote tensors API of the GPU Plugin](./GPU_RemoteTensor_API.md).


## Additional Resources
* [Supported Devices](Supported_Devices.md)
* [Optimization guide](@ref openvino_docs_deployment_optimization_guide_dldt_optimization_guide)
* [GPU plugin developers documentation](https://github.com/openvinotoolkit/openvino/wiki/GPUPluginDevelopersDocs)
