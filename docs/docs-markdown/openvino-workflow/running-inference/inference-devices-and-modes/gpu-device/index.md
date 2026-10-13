---
sidebar_position: 2
sidebar_label: "GPU Device"
description: "The GPU plugin in the Intel® Distribution of OpenVINO™ toolkit is an OpenCL based plugin for inference of deep neural networks on Intel® GPus."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# GPU Device


The GPU plugin is an OpenCL based plugin for inference of deep neural networks on Intel GPUs, both integrated and discrete ones.
For an in-depth description of the GPU plugin, see:

- [GPU plugin developer documentation](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/README.md)
- [OpenVINO Runtime GPU plugin source files](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_gpu/)
- [Start AI Development with Intel](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/overview.html)

The GPU plugin is a part of the Intel® Distribution of OpenVINO™ toolkit. For more information on how to configure a system to use it, see the [GPU configuration](../../../../get-started/install-openvino/configurations/configurations-intel-gpu.md).

## Device Naming Convention

- Devices are enumerated as `GPU.X`, where `X={0, 1, 2,...}` (both Intel® and non-Intel GPUs are considered).
- The order of GPUs is predefined: integrated Intel® GPUs first, followed by discrete Intel® GPUs, and then non-Intel® GPUs. The order of non-Intel® GPUs is not guaranteed.
- A physical GPU exposed through multiple OpenCL® runtime platforms may appear as more than one device entry. When the same GPU is exposed by both the Intel® and another OpenCL® platform, the Intel®-platform entry is enumerated first. The order of non-Intel® OpenCL® platforms is not guaranteed.
- If the system has an integrated Intel® GPU, its `id` is always 0 (`GPU.0`).
- If the system does not have an integrated Intel® GPU, devices are enumerated, starting from 0.
- The `GPU` is an alias for `GPU.0`.
- For GPUs with multi-tile architecture (multiple sub-devices in OpenCL terms), a specific tile may be addressed as `GPU.X.Y`, where `X,Y={0, 1, 2,...}`, `X` - id of the GPU device, `Y` - id of the tile within device `X`

For demonstration purposes, see the [Hello Query Device C++ Sample](../../../../get-started/learn-openvino/openvino-samples/hello-query-device.md) that can print out the list of available devices with associated indices. Below is an example output (truncated to the device names only):

```bash
./hello_query_device
Available devices:
    Device: CPU
...
    Device: GPU.0
...
    Device: GPU.1
```

Then, the device name can be passed to the `ov::Core::compile_model()` method, running on:

<Tabs>
  <TabItem value="default-device" label="default device">
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
        core = ov.Core()
        compiled_model = core.compile_model(model, "GPU")
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
        ov::Core core;
        auto model = core.read_model("model.xml");
        auto compiled_model = core.compile_model(model, "GPU");
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="specific-gpu" label="specific GPU">
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
        core = ov.Core()
        compiled_model = core.compile_model(model, "GPU.1")
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
        ov::Core core;
        auto model = core.read_model("model.xml");
        auto compiled_model = core.compile_model(model, "GPU.1");
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="specific-tile" label="specific tile">
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
        core = ov.Core()
        compiled_model = core.compile_model(model, "GPU.1.0")
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
        ov::Core core;
        auto model = core.read_model("model.xml");
        auto compiled_model = core.compile_model(model, "GPU.1.0");
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

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
For more details on how to get a quantized model, refer to the [Model Optimization guide](../../../model-optimization/index.md).

Floating-point precision of a GPU primitive is selected based on operation precision in the OpenVINO IR, except for the [&lt;compressed f16 OpenVINO IR form](../../../model-preparation/conversion-parameters.md), which is executed in the `f16` precision.

:::note
The newer generation Intel Iris Xe and Xe MAX GPUs provide accelerated performance for i8/u8 models. Hardware acceleration for `i8`/`u8` precision may be unavailable on older generation platforms. In such cases, a model is executed in the floating-point precision taken from IR.
Hardware support of `u8`/`i8` acceleration can be queried via the `ov::device::capabilities` property.
:::

[Hello Query Device C++ Sample](../../../../get-started/learn-openvino/openvino-samples/hello-query-device.md) can be used to print out the supported data types for all detected devices.

## Supported Features

The GPU plugin supports the following features:

### Automatic Device Selection

If a system has multiple GPUs (for example, an integrated and a discrete Intel GPU), then any supported model can be executed on all GPUs simultaneously.
It is done by specifying `AUTO:GPU.1,GPU.0` as a target device, and adding the `CUMULATIVE_THROUGHPUT` parameter.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()
      compiled_model = core.compile_model(model, "AUTO:GPU.1,CPU.0", {hints.performance_mode: hints.PerformanceMode.CUMULATIVE_THROUGHPUT})
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
      ov::Core core;
      auto model = core.read_model("model.xml");
      auto compiled_model = core.compile_model(model, "AUTO:GPU.1,CPU.0", ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
  ```
  </TabItem>
</Tabs>

For more details, see the [Automatic Device Selection](../auto-device-selection/index.md).

### Automatic Batching

The GPU plugin is capable of reporting `ov::max_batch_size` and `ov::optimal_batch_size` metrics with respect to the current hardware
platform and model. Therefore, automatic batching is enabled by default when `ov::optimal_batch_size` is `> 1` and `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` is set.
Alternatively, it can be enabled explicitly via the device notion, for example `BATCH:GPU`.

<Tabs>
  <TabItem value="batching-via-batch-plugin" label="Batching via BATCH plugin">
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
        core = ov.Core()
        compiled_model = core.compile_model(model, "BATCH:GPU")
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
        ov::Core core;
        auto model = core.read_model("model.xml");
        auto compiled_model = core.compile_model(model, "BATCH:GPU");
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="batching-via-throughput-hint" label="Batching via throughput hint">
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
        import openvino.properties.hint as hints

        core = ov.Core()
        compiled_model = core.compile_model(
            model,
            "GPU",
            {
                hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
            },
        )
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
        ov::Core core;
        auto model = core.read_model("model.xml");
        auto compiled_model = core.compile_model(model, "GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

For more details, see the [Automatic batching](../automatic-batching.md).

### Multi-stream Execution

If either the `ov::num_streams(n_streams)` with `n_streams > 1` or the `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` property is set for the GPU plugin,
multiple streams are created for the model. In the case of GPU plugin each stream has its own host thread and an associated OpenCL queue
which means that the incoming infer requests can be processed simultaneously.

:::note
Simultaneous scheduling of kernels to different queues does not mean that the kernels are actually executed in parallel on the GPU device.
The actual behavior depends on the hardware architecture and in some cases the execution may be serialized inside the GPU driver.
:::

When multiple inferences of the same model need to be executed in parallel, the multi-stream feature is preferred to multiple instances of the model or application.
The reason for this is that the implementation of streams in the GPU plugin supports weight memory sharing across streams, thus, memory consumption may be lower, compared to the other approaches.

For more details, see the [optimization guide](../../optimize-inference/index.md).

### Dynamic Shapes

:::note
Currently, dynamic shape support for GPU is a preview feature and has the following limitations:

- It mainly supports NLP models (Natural Language Processing). Not all operations and optimization passes support dynamic shapes.
  As a result, a given model may crash or experience significant performance drops.
- Due to the dominant runtime overhead on the host device, dynamic shapes may perform worse than static shapes on a discrete GPU.
- Dynamic rank is not supported.
:::

The general description of what dynamic shapes are and how they are used can be found in
[dynamic shapes guide](../../model-input-output/dynamic-shapes/index.md).
To support dynamic shape execution, the following basic infrastructures are implemented:

- Runtime shape inference: infers output shapes of each primitive for a new input shape at runtime.
- Shape agnostic kernels: new kernels that can run arbitrary shapes. If a shape-agnostic kernel is not available,
  the required kernel is compiled at runtime for each shape.
- Asynchronous kernel compilation: even when a shape-agnostic kernel is available,
  the GPU plugin compiles an optimal kernel for the given shape and preserves it in the in-memory cache for future use.
- In-memory cache: preserves kernels compiled at runtime and weights reordered for the specific kernels.

#### Bounded dynamic batch

It is worth noting that the internal behavior differs in the case of bounded-batch dynamic shapes,
which means that only the batch dimension is dynamic and it has a fixed upper bound.

While general dynamic shapes can run on one compiled model, for the bounded dynamic batch the GPU plugin creates `log2(N)`
low-level execution graphs in batch sizes equal to the powers of 2, to emulate the dynamic behavior (`N` - is the upper bound for the batch dimension here).
As a result, the incoming infer request with a specific batch size is executed via the minimal combination of internal networks.
For example, a batch size of 33 may be executed via two internal networks with batch sizes of 32 and 1.
This approach is adopted for performance reasons, but it requires more memory and increased compilation time for multiple copies of internal networks.

The code snippet below demonstrates examples of a bounded dynamic batch:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      core = ov.Core()

      C = 3
      H = 224
      W = 224

      model.reshape([(1, 10), C, H, W])

      # compile model and create infer request
      compiled_model = core.compile_model(model, "GPU")
      infer_request = compiled_model.create_infer_request()

      # create input tensor with specific batch size
      input_tensor = ov.Tensor(model.input().element_type, [2, C, H, W])

      # ...

      results = infer_request.infer([input_tensor])
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp

  // Read model
  ov::Core core;
  auto model = core.read_model("model.xml");

  model->reshape({{ov::Dimension(1, 10), ov::Dimension(C), ov::Dimension(H), ov::Dimension(W)}});  // {1..10, C, H, W}

  // compile model and create infer request
  auto compiled_model = core.compile_model(model, "GPU");
  auto infer_request = compiled_model.create_infer_request();
  auto input = model->get_parameters().at(0);

  // ...

  // create input tensor with specific batch size
  ov::Tensor input_tensor(input->get_element_type(), {2, C, H, W});

  // ...

  infer_request.set_tensor(input, input_tensor);
  infer_request.infer();
  ```
  </TabItem>
</Tabs>

#### Notes for performance and memory consumption in dynamic shapes

- Extra CPU utilization during inference:
  
  - Shape inference for new input shapes
  - Kernel compilation in runtime for optimal kernel
  - Unfusion of the fused subgraph when fusing is not allowed for a runtime shape
- Higher memory consumption for in-memory cache
  
  - Optimal kernels and weights from the previously used shapes are preserved in in-memory cache for future use

#### Recommendations for performance improvement

- Use static shapes whenever possible
  
  - Static models can benefit from more aggressive optimizations, such as, constant propagation, fusing, and reorder optimization.
    If the same shape is used for a dynamic and a static model, performance is worse in the dynamic one.
    It is, therefore, recommended to reshape dynamic models to static ones, if the scenario allows.
- Use bounded dynamic shapes whenever possible
  
  - The GPU plugin needs to reallocate memory if the current shape is larger than the maximum of the previous shapes, which causes additional overhead.
  - Using a bounded dynamic shape will help to reduce such overhead. For example, use `{ov::Dimension(1, 10), ov::Dimension(1, 384)}`
    instead of `{ov::Dimension(-1), ov::Dimension(-1)}`.
  - Note that a bounded dynamic *batch* is handled differently as mentioned above.
- Use permanent cache, e.g., OpenVino model\_cache, to reduce the runtime re-compilation overhead
  
  - GPU plugin deploys in-memory cache to store compiled kernels for previously used shapes,
    but the size of such an in-memory cache is limited. Therefore, it is recommended to use
    a permanent cache such as OpenVino model\_cache. For more details, See
    [Model caching overview](../../optimize-inference/optimizing-latency/model-caching-overview.md).
- The longer the inference sequence, the better throughput can be obtained, because it can
  leverage more compilation time during inference.
  
  - If the primitive has a shape-agnostic kernel and the static shape kernel for the current
    shape does not exist in the in-memory cache, the shape-agnostic kernel is used. Then, as
    mentioned above, optimal kernels for the current shapes are also asynchronously compiled
    in parallel for future use. If the application process removes the CompiledModel object
    and the GPU plugin is unusable, any not-yet-started compilation tasks for optimal kernels
    will be canceled. However, if the application process allows enough time for the enqueued
    asynchronous compilation tasks, the more optimal kernels become available, enabling better
    throughput. For example, running 200 inputs of
    `{[1, 1], ..., [1, 50], [1, 1], ... , [1, 50], [1, 1], ..., [1, 50], [1, 1], ..., [1, 50]}`
    may achieve better throughput than running 100 inputs of `{[1, 1], ..., [1, 50], [1, 1], ... , [1,50]}`.

### Preprocessing Acceleration

The GPU plugin has the following additional preprocessing options:

- The `ov::intel_gpu::memory_type::surface` and `ov::intel_gpu::memory_type::buffer` values for the `ov::preprocess::InputTensorInfo::set_memory_type()` preprocessing method. These values are intended to be used to provide a hint for the plugin on the type of input Tensors that will be set in runtime to generate proper kernels.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      model = get_model()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
      using namespace ov::preprocess;
      auto p = PrePostProcessor(model);
      p.input().tensor().set_element_type(ov::element::u8)
                        .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                        .set_memory_type(ov::intel_gpu::memory_type::surface);
      p.input().preprocess().convert_color(ov::preprocess::ColorFormat::BGR);
      p.input().model().set_layout("NCHW");
      auto model_with_preproc = p.build();
  ```
  </TabItem>
</Tabs>

With such preprocessing, GPU plugin will expect `ov::intel_gpu::ocl::ClImage2DTensor` (or derived) to be passed for each NV12 plane via `ov::InferRequest::set_tensor()` or `ov::InferRequest::set_tensors()` methods.

For usage examples, refer to the [RemoteTensor API](remote-tensor-api-gpu-plugin.md).

For more details, see the [preprocessing API](../../optimize-inference/optimize-preprocessing/index.md).

### Model Caching

Model Caching helps reduce application startup delays by exporting and reusing
the compiled model automatically. The cache for the GPU plugin may be enabled
via the common OpenVINO `ov::cache_dir` property.

This means that all plugin-specific model transformations are executed on each `ov::Core::compile_model()`
call, regardless of the `ov::cache_dir` option. Still, since kernel compilation is a bottleneck in the model
loading process, a significant load time reduction can be achieved.
Currently, GPU plugin implementation fully supports static models only. For dynamic models,
kernel caching is used instead and multiple ‘.cl\_cache’ files are generated along with the ‘.blob’ file.

For more details, see the [Model caching overview](../../optimize-inference/optimizing-latency/model-caching-overview.md).

### Extensibility

For information on this subject, see the [GPU Extensibility](../../../../documentation/openvino-extensibility/custom-gpu-operations.md).

### GPU Context and Memory Sharing via RemoteTensor API

For information on this subject, see the [RemoteTensor API of GPU Plugin](remote-tensor-api-gpu-plugin.md).

## Supported Properties

The plugin supports the properties listed below.

- For more details on the properties, see the [Device Properties](https://docs.openvino.ai/2026/api/c_cpp_api/group__ov__runtime__cpp__prop__api.html).
- For more details on usage of the properties, see the [Query Device Properties - Configuration](../query-device-properties.md).
- To check the exact list of available properties on target device, you can run [Hello Query Device C++ Sample](../../../../get-started/learn-openvino/openvino-samples/hello-query-device.md).

### Read-write properties

All parameters must be set before calling `ov::Core::compile_model()` in order to take effect or passed as additional argument to `ov::Core::compile_model()`.

- `ov::cache_dir`
- `ov::cache_encryption_callbacks`
- `ov::cache_mode`
- `ov::compilation_num_threads`
- `ov::device::id`
- `ov::enable_profiling`
- `ov::num_streams`
- `ov::weights_path`
- `ov::hint::activations_scale_factor`
- `ov::hint::dynamic_quantization_group_size`
- `ov::hint::enable_cpu_pinning`
- `ov::hint::enable_cpu_reservation`
- `ov::hint::execution_mode`
- `ov::hint::inference_precision`
- `ov::hint::kv_cache_precision`
- `ov::hint::model`
- `ov::hint::model_priority`
- `ov::hint::num_requests`
- `ov::hint::performance_mode`
- `ov::intel_gpu::disable_winograd_convolution`
- `ov::intel_gpu::enable_loop_unrolling`
- `ov::intel_gpu::hint::enable_sdpa_optimization`
- `ov::intel_gpu::hint::host_task_priority`
- `ov::intel_gpu::hint::queue_priority`
- `ov::intel_gpu::hint::queue_throttle`

### Read-only Properties

- `ov::available_devices`
- `ov::max_batch_size`
- `ov::optimal_batch_size`
- `ov::range_for_async_infer_requests`
- `ov::range_for_streams`
- `ov::supported_properties`
- `ov::device::architecture`
- `ov::device::capabilities`
- `ov::device::full_name`
- `ov::device::gops`
- `ov::device::luid`
- `ov::device::type`
- `ov::device::uuid`
- `ov::intel_gpu::device_total_mem_size`
- `ov::intel_gpu::device_max_alloc_mem_size`
- `ov::intel_gpu::execution_units_count`
- `ov::intel_gpu::memory_statistics`
- `ov::intel_gpu::uarch_version`

## Limitations

In some cases, the GPU plugin may implicitly execute several primitives on CPU using internal implementations, which may lead to an increase in CPU utilization.
Below is a list of such operations:

- Proposal
- NonMaxSuppression
- DetectionOutput

The behavior depends on specific parameters of the operations and hardware configuration.

:::info[Important]
While working on a fine tuned model, inference may give an inaccuracy and performance drop
on GPU if winograd convolutions are selected. This issue can be fixed by disabling winograd
convolutions:

```text
compiled_model = core.compile_model(ov_model, device_name=devStr1, config={ "GPU_DISABLE_WINOGRAD_CONVOLUTION": True })
```
:::

## GPU Performance Checklist: Summary

Since OpenVINO relies on the OpenCL kernels for the GPU implementation, many general OpenCL tips apply:

- Prefer `FP16` inference precision over `FP32`, as Model Conversion API can generate both variants, and the `FP32` is the default. To learn about optimization options, see [Optimization Guide](../../../model-optimization/index.md).
- Try to group individual infer jobs by using [automatic batching](../automatic-batching.md).
- Consider [caching](../../optimize-inference/optimizing-latency/model-caching-overview.md) to minimize model load time.
- If your application performs inference on the CPU alongside the GPU, or otherwise loads the host heavily, make sure that the OpenCL driver threads do not starve. [CPU configuration options](../cpu-device/index.md) can be used to limit the number of inference threads for the CPU plugin.
- Even in the GPU-only scenario, a GPU driver might occupy a CPU core with spin-loop polling for completion. If CPU load is a concern, consider the dedicated `queue_throttle` property mentioned previously. Note that this option may increase inference latency, so consider combining it with multiple GPU streams or [throughput performance hints](../../optimize-inference/high-level-performance-hints.md).
- When operating media inputs, consider [remote tensors API of the GPU Plugin](remote-tensor-api-gpu-plugin.md).

## Additional Resources

- [Working with GPUs in OpenVINO™ Notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/gpu-device/)
- [Inference Devices and Modes](../index.md).
- [Optimization guide](../../optimize-inference/index.md).
- [GPU plugin developer documentation](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_gpu/README.md)
