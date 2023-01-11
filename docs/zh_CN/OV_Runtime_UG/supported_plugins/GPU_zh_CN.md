# GPU 设备 {#openvino_docs_OV_UG_supported_plugins_GPU_zh_CN}


GPU 插件是一个基于 OpenCL 的插件，用于在英特尔 GPU 上推理深度神经网络，包括集成 GPU 和独立 GPU。
有关 GPU 插件的深入描述，请参见：
- [GPU 插件开发人员文档](https://github.com/openvinotoolkit/openvino/wiki/GPUPluginDevelopersDocs)
- [OpenVINO™ 运行时 GPU 插件源文件](https://github.com/openvinotoolkit/openvino/tree/releases/2022/2/src/plugins/intel_gpu/)
- [通过英特尔® 处理器显卡加速深度学习推理](https://software.intel.com/en-us/articles/accelerating-deep-learning-inference-with-intel-processor-graphics)。

GPU 插件是英特尔® 发行版 OpenVINO™ 工具套件的一部分。有关如何配置系统以便使用它的更多详细信息，请参见 [GPU 配置](@ref openvino_docs_install_guides_configurations_for_intel_gpu)。

## 设备命名约定
* 设备被枚举为 `GPU.X`。其中 `X={0, 1, 2,...}`（仅考虑英特尔® GPU 设备）。
* 如果系统具有集成 GPU，则其 `id` 始终为 0 (`GPU.0`)。
* 其他 GPU 的顺序不是预定义的，并且取决于 GPU 驱动程序。
* `GPU` 是 `GPU.0` 的别名。
* 如果系统没有集成 GPU，则从 0 开始枚举设备。
* 对于具有多块架构的 GPU（用 OpenCL 术语来说指的是多个子设备），特定块可以作为 `GPU.X.Y` 进行寻址。其中 `X,Y={0, 1, 2,...}`，`X` 是 GPU 设备的 ID，`Y` 是设备 `X` 内块的 ID

为了演示目的，请参见 [Hello 查询设备 C++ 样本](../../../../samples/cpp/hello_query_device/README.md)，使用该样本可以打印出具有关联索引的可用设备列表。下面是一个示例输出（仅截断为设备名称）：

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

然后，设备名称可以传递到 `ov::Core::compile_model()` 方法：

@sphinxtabset

@sphinxtab{在默认设备上运行}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_default_gpu
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_default_gpu
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{在特定 GPU 上运行}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_gpu_with_id
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_gpu_with_id
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{在特定块上运行}

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

## 支持的推理数据类型
GPU 插件支持以下数据类型作为内部基元的推理精度：

- 浮点数据类型：
  - f32
  - f16
- 量化数据类型：
  - u8
  - i8
  - u1

每个基元的所选精度取决于 IR 中的操作精度、量化基元和可用的硬件功能。
`u1`/`u8`/`i8` 数据类型仅用于量化操作，这意味着不会为非量化操作自动选择它们。
有关如何获得量化模型的更多详细信息，请参阅[模型优化指南](@ref openvino_docs_model_optimization_guide_zh_CN)。

GPU 基元的浮点精度是根据 OpenVINO™ IR 中的操作精度选择的，但[压缩的 f16 OpenVINO™ IR 格式](../../../MO_DG/prepare_model/FP16_Compression.md)除外。该格式以 `f16` 精度执行。

> **NOTE**: `i8`/`u8` 精度的硬件加速在某些平台上可能不可用。在这种情况下，以从 IR 获取的浮点精度执行模型。可以通过 `ov::device::capabilities` 属性查询支持 `u8`/`i8` 加速的硬件。

[Hello 查询设备 C++ 样本](../../../../samples/cpp/hello_query_device/README.md)可以用于打印出所有检测到的设备支持的数据类型。

## 支持的功能

GPU 插件支持下列功能：

### 多设备执行
如果系统具有多个 GPU（例如，集成的英特尔 GPU 和单独的英特尔 GPU），则任何支持的模型都可以同时在所有 GPU 上执行。
通过指定 `MULTI:GPU.1,GPU.0` 为目标设备来完成。

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_multi
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_multi
@endsphinxtab

@endsphinxtabset

有关更多详细信息，请参见[多设备执行](../../../OV_Runtime_UG/multi_device.md)。

### 自动批处理
GPU 插件能够报告与当前硬件平台和模型相关的 `ov::max_batch_size` 和 `ov::optimal_batch_size` 指标。因此，当 `ov::optimal_batch_size` 为 `> 1` 且设置 `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` 时，默认情况下会启用自动批处理。或者，可以通过设备概念明确启用它，例如 `BATCH:GPU`。

@sphinxtabset

@sphinxtab{通过 BATCH 插件进行批处理}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/compile_model.cpp compile_model_batch_plugin
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/compile_model.py compile_model_batch_plugin
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{通过吞吐量提示进行批处理}

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

有关更多详细信息，请参见[自动批处理](../../../OV_Runtime_UG/automatic_batching.md)。

### 多流执行
如果为 GPU 插件设置 `ov::num_streams(n_streams)` (`n_streams > 1`) 或 `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` 属性，则可以为模型创建多个流。如果是 CPU 插件，每个流都有自己的主机线程和相关的 OpenCL 队列，这意味着可以同时处理传入的推理请求。

> **NOTE**: 将内核同时调度到不同的队列并不意味着内核实际上是在 GPU 设备上并行执行的。实际行为取决于硬件架构，并且在某些情况下，执行可能会在 GPU 驱动程序中序列化。

当需要并行执行同一模型的多个推理时，多流功能优先于模型或应用的多个实例。
其原因是 GPU 插件中的流实现支持所有流中共享权重内存。因此，与其他方法相比，内存消耗可能更低。

有关更多详细信息，请参见[优化指南](@ref openvino_docs_deployment_optimization_guide_dldt_optimization_guide_zh_CN)。

### 动态输入
GPU 插件仅支持具有固定上限的批处理维度的动态形状（在[布局术语](../../../OV_Runtime_UG/layout_overview.md)中指定为 `N`）。不支持任何其他动态维度。在内部，GPU 插件为等于 2 的幂数的批次大小创建 `log2(N)`（`N` - 此处是批处理维度的上限）低级执行图来模拟动态行为，以便通过内部网络的最小组合来执行具有特定批次大小的传入推理请求。
例如，可以通过批次大小为 32 和 1 的 2 个内部网络执行批次大小 33。

> **NOTE**: 与静态批处理场景相比，这种方法需要更多内存，并且整个模型的编译时间明显更长。

以下代码片段演示了如何在简单场景中使用动态批处理：

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/dynamic_batch.cpp dynamic_batch
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/dynamic_batch.py dynamic_batch
@endsphinxtab

@endsphinxtabset

有关更多详细信息，请参见[动态形状指南](../ov_dynamic_shapes_zh_CN.md)。

### 预处理加速
GPU 插件具有以下附加预处理选项：
- 用于 `ov::preprocess::InputTensorInfo::set_memory_type()` 预处理方法的 `ov::intel_gpu::memory_type::surface` 和 `ov::intel_gpu::memory_type::buffer` 值。这些值旨在用于为插件提供输入张量类型相关的提示，这些张量将在运行时设置，以生成适当的内核。

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/gpu/preprocessing.cpp init_preproc
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/gpu/preprocessing.py init_preproc
@endsphinxtab

@endsphinxtabset

通过这种预处理，GPU 插件将期望通过 `ov::InferRequest::set_tensor()` 或 `ov::InferRequest::set_tensors()` 方法为每个 NV12 平面传递 `ov::intel_gpu::ocl::ClImage2DTensor`（或派生）。

有关使用示例，请参阅 [RemoteTensor API](../../../OV_Runtime_UG/supported_plugins/GPU_RemoteTensor_API.md)。

有关更多详细信息，请参见[预处理 API](../../../OV_Runtime_UG/preprocessing_overview.md)。

### 模型缓存
可以通过通用 OpenVINO™ `ov::cache_dir` 属性启用 GPU 插件的缓存。GPU 插件实现仅支持缓存已编译内核。因此无论 `cache_dir` 选项如何，所有插件特定的模型转换都会在每个 `ov::Core::compile_model()` 调用时执行。
尽管如此，由于内核编译是模型加载过程中的瓶颈。因此启用 `ov::cache_dir` 属性可以显著减少加载时间。

有关更多详细信息，请参见[模型缓存概述](../../../OV_Runtime_UG/Model_caching_overview.md)。

### 扩展性
有关此主题的信息，请参见 [GPU 扩展性](@ref openvino_docs_Extensibility_UG_GPU).

### GPU 上下文和内存通过 RemoteTensor API 共享
有关此主题的信息，请参见 [GPU 插件的 RemoteTensor API](../../../OV_Runtime_UG/supported_plugins/GPU_RemoteTensor_API.md)。


## 支持的属性
插件支持以下所列属性。

### 读写属性
在调用 `ov::Core::compile_model()` 之前必须设置所有参数才能生效或作为附加参数传递给 `ov::Core::compile_model()`。

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

### 只读属性
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

## 限制
在某些情况下，GPU 插件可能会使用内部实现在 CPU 上隐式地执行多个基元，这可能会导致 CPU 利用率增加。
以下是此类操作的列表：
- 提案
- NonMaxSuppression
- DetectionOutput

行为取决于操作的特定参数和硬件配置。

## GPU 性能清单：摘要<a name="gpu-checklist"></a>
由于 OpenVINO™ 依赖 OpenCL 内核进行 GPU 实现。因此许多通用 OpenCL 提示都适用：
- `FP16` 推理精度优于 `FP32`，因为模型优化器可以生成两个变体，并且 `FP32` 是默认值。此外，请考虑使用[训练后优化工具](https://docs.openvino.ai/2022.2/pot_introduction.html)。
- 尝试使用[自动批处理](../../../OV_Runtime_UG/automatic_batching.md)对各个推理作业进行分组。
- 考虑[缓存](../../../OV_Runtime_UG/Model_caching_overview.md)，以尽量减少模型加载时间。
- 如果您的应用在 CPU 和 GPU 上执行推理，或者以其他方式重载主机，请确保 OpenCL 驱动程序线程不会停顿。[CPU 配置选项](./CPU_zh_CN.md)可以用于限制 CPU 插件的推理线程数量。
- 即使仅在 GPU 上执行推理，GPU 驱动程序可能会占用 CPU 核心，并通过自旋循环轮询来完成。如果 CPU 负载是一个问题，请考虑前面提到的专用 `queue_throttle` 属性。请注意，此选项可能会增加推理延迟。因此请考虑将其与多个 GPU 流或[吞吐量性能提示](../performance_hints_zh_CN.md)结合使用。
- 操作媒体输入时，请考虑 [GPU 插件的远程张量 API](../../../OV_Runtime_UG/supported_plugins/GPU_RemoteTensor_API.md)。


## 其他资源
* [支持的设备](../../../OV_Runtime_UG/supported_plugins/Supported_Devices.md)
* [优化指南](@ref openvino_docs_optimization_guide_dldt_optimization_guide_zh_CN)
* [GPU 插件开发人员文档](https://github.com/openvinotoolkit/openvino/wiki/GPUPluginDevelopersDocs)
