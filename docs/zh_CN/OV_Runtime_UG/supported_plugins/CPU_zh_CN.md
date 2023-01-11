# CPU 设备 {#openvino_docs_OV_UG_supported_plugins_CPU_zh_CN}

CPU 插件是英特尔® 发行版 OpenVINO™ 工具套件的一部分。其开发目的是为了实现英特尔® x86-64 CPU 上神经网络的高性能推理。
有关 CPU 插件的深入描述，请参见：

- [CPU 插件开发人员文档](https://github.com/openvinotoolkit/openvino/wiki/CPUPluginDevelopersDocs)。

- [OpenVINO™ 运行时 CPU 插件源文件](https://github.com/openvinotoolkit/openvino/tree/releases/2022/2/src/plugins/intel_cpu/)。


## 设备名称
`CPU` 设备名称用于 CPU 插件。即使平台上可以有多个物理套接字，但 OpenVINO™ 也只列出了一个此类设备。
在多套接字平台上，会自动处理 NUMA 节点之间的负载平衡和内存使用分配情况。   
为了将 CPU 用于推理，应将设备名称传递到 `ov::Core::compile_model()` 方法：

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/compile_model.cpp compile_model_default
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/compile_model.py compile_model_default
@endsphinxtab

@endsphinxtabset

## 支持的推理数据类型
CPU 插件支持以下数据类型作为内部基元的推理精度：

- 浮点数据类型：
  - f32
  - bf16
- 整数数据类型：
  - i32
- 量化数据类型：
  - u8
  - i8
  - u1
  
[Hello 查询设备 C++ 样本](../../../../samples/cpp/hello_query_device/README.md)可以用于打印出所有检测到的设备支持的数据类型。

### 量化数据类型细节

每个基元的所选精度取决于 IR 中的操作精度、量化基元和可用的硬件功能。
`u1/u8/i8` 数据类型仅用于量化操作，即不会自动为非量化操作选择的数据类型。

有关如何获得量化模型的更多详细信息，请参见[低精度优化指南](@ref openvino_docs_model_optimization_guide_zh_CN)。

> **NOTE**: 不支持英特尔® AVX512-VNNI 的平台有一个已知的“饱和问题”。该问题可能会导致 `u8/i8` 精度计算的计算精度降低。
> 请参见[饱和（溢出）问题部分](@ref pot_saturation_issue)以获取有关如何检测此类问题和可能的解决方法的更多信息。

### 浮点数据类型细节

CPU 基元的默认浮点精度为 `f32`。如需支持 `f16` OpenVINO™ IR，插件要在内部将所有 `f16` 值转换为 `f32`，并且所有计算都使用 `f32` 的原生精度执行。
在本地支持 `bfloat16` 计算的平台上（具有 `AVX512_BF16` 扩展），会自动使用 `bf16` 类型，而不会使用 `f32`，以获得更高性能。因此，运行 `bf16` 模型不需要采取特殊步骤。
有关 `bfloat16` 格式的更多详细信息，请参见 [BFLOAT16 – 硬件数字定义白皮书](https://software.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf)。

使用 `bf16` 精度提供以下性能优势：

- 加快两个 `bfloat16` 数字的乘法，因为 `bfloat16` 数据的尾数较短。
- 内存消耗减少，因为 `bfloat16` 数据大小是 32 位浮点大小的一半。

如需检查 CPU 设备是否支持 `bfloat16` 数据类型，请使用[查询设备属性接口](../../../OV_Runtime_UG/supported_plugins/config_properties.md)查询应在 CPU 功能列表中包含 `BF16` 的 `ov::device::capabilities` 属性：

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/Bfloat16Inference0.cpp part0
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/Bfloat16Inference.py part0
@endsphinxtab

@endsphinxtabset

如果模型已转换为 `bf16`，则 `ov::hint::inference_precision` 将设置为 `ov::element::bf16` 并且可以通过 `ov::CompiledModel::get_property` 调用进行检查。以下代码显示如何获得元件类型：

@snippet snippets/cpu/Bfloat16Inference1.cpp part1

如需在具有原生 `bf16` 支持的目标上以 `f32` 精度推理模型，而不是使用 `bf16`，请将 `ov::hint::inference_precision` 设置为 `ov::element::f32`。

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/Bfloat16Inference2.cpp part2
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/Bfloat16Inference.py part2
@endsphinxtab

@endsphinxtabset

`Bfloat16` 软件模拟模式适用于不支持本机 `avx512_bf16` 指令、采用英特尔® AVX-512 指令集的 CPU。此模式用于开发目的，无法保证良好的性能。
如需启用模拟，必须明确将 `ov::hint::inference_precision` 设置为 `ov::element::bf16`。

> **NOTE**: 如果在不支持本机 bfloat16 或 bfloat16 模拟模式的 CPU 上将 ov::hint::inference_precision 设置为 ov::element::bf16，会引发异常。

> **NOTE**: 由于 `bfloat16` 数据类型的尾数大小减小。因此生成的 `bf16` 推理精度可能与 `f32` 推理不同，特别是对于未使用 `bfloat16` 数据类型进行训练的模型而言。如果 `bf16` 推理精度不可接受，建议切换到 `f32` 精度。
  
## 支持的功能

### 多设备执行
除 CPU 以外，如果系统包含 OpenVINO™ 支持的设备（例如集成 GPU），则任何支持的模型都可以同时在所有设备上执行。
这可以通过在同时使用 CPU 和 GPU 的情况下将 `MULTI:CPU,GPU.0` 指定为目标设备来实现。

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/compile_model.cpp compile_model_multi
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/compile_model.py compile_model_multi
@endsphinxtab

@endsphinxtabset

有关更多详细信息，请参见[多设备执行](../../../OV_Runtime_UG/multi_device.md)一文。

### 多流执行
如果可以为 CPU 插件设置具有 `n_streams > 1` 或 `ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)` 属性的 `ov::num_streams(n_streams)`，则可以为模型创建多个流。如果是 CPU 插件，每个流都有自己的主机线程，这意味着可以同时处理传入的推理请求。
就 NUMA 节点的物理内存使用情况而言，每个流都固定到其自己的物理核心组，以最大程度地减少 NUMA 节点之间数据传输的开销。

有关更多详细信息，请参见[优化指南](@ref openvino_docs_deployment_optimization_guide_dldt_optimization_guide_zh_CN)。

> **NOTE**: 在延迟方面，请注意，在多套接字平台上仅运行一个流可能会在 NUMA 节点之间的数据传输中引入额外开销。在这种情况下，最好使用 `ov::hint::PerformanceMode::LATENCY` 性能提示。有关更多详细信息，请参见[性能提示](@ref openvino_docs_OV_UG_Performance_Hints_zh_CN)概述。

### 动态输入
在操作集覆盖范围方面，CPU 为具有动态形状的模型提供完整的功能支持。

> **NOTE**: CPU 插件不支持动态更改等级的张量。如果尝试使用此种张量推理模型，则会引发异常。

动态形状支持会给内存管理带来额外开销，并且可能会限制内部运行时优化。
使用的自由度越多，实现最佳性能的难度就越大。
最灵活的配置和最方便的方法是完全未定义的形状，这意味着不会应用形状维度的约束。
但是，降低不确定性水平会使性能提升。
您可以通过内存重用来减少内存消耗，从而实现更好的缓存本地性并提高推理性能。为此，请以定义的上界为限，明确设置动态形状。

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/dynamic_shape.cpp static_shape
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/dynamic_shape.py static_shape
@endsphinxtab

@endsphinxtabset

> **NOTE**: 与使用静态形状推理同一模型相比，使用完全未定义的形状可能会导致内存消耗显著增加。
> 如果内存消耗不可接受，但仍需要动态形状，则可以使用具有定义上限的形状重塑模型，以减少内存占用空间。

如果提前知道模型形状，则某些运行时优化效果会更好。
因此，如果在推理调用之间不更改输入数据形状，建议使用具有静态形状的模型或使用静态输入形状重塑现有模型，以获得最佳性能。

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/cpu/dynamic_shape.cpp static_shape
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/cpu/dynamic_shape.py static_shape
@endsphinxtab

@endsphinxtabset

有关更多详细信息，请参见[动态形状指南](../ov_dynamic_shapes_zh_CN.md)。

### 预处理加速
CPU 插件支持全套预处理操作，并能高性能实现这些操作。

有关更多详细信息，请参阅[预处理 API 指南](../../../OV_Runtime_UG/preprocessing_overview.md)。

@sphinxdirective
.. dropdown:: 支持处理张量精度转换的 CPU 插件仅限用于以下 ov::element 类型：

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

### 模型缓存
CPU 支持导入/导出网络功能。如果通过通用 OpenVINO™ `ov::cache_dir` 属性启用模型缓存，则插件会在模型编译期间自动在指定目录中创建缓存的 blob。
此缓存的 blob 包含网络的部分表示形式，可执行常见的运行时优化和进行低精度转换。
下次编译模型时，缓存的表示将加载到插件中，而不是初始 OpenVINO™ IR 中。因此将跳过上述转换步骤。
这些转换在模型编译期间会花费大量时间。因此缓存此表示可减少模型后续编译所花费的时间，从而减少首次推理延迟 (FIL)。

有关更多详细信息，请参见[模型缓存](@ref openvino_docs_OV_UG_Model_caching_overview)概述。

### 扩展性
如果 CPU 插件无法实现自己的此类操作，则支持 `ov::Op` 参考实现的回退。
那意味着 [OpenVINO™ 扩展性机制](@ref openvino_docs_Extensibility_UG_Intro_zh_CN)也可用于插件扩展。
通过重写派生操作类中的 `ov::Op::evaluate` 方法，可以启用自定义操作实现的回退（请参见[自定义 OpenVINO™ 操作](@ref openvino_docs_Extensibility_UG_add_openvino_ops) 了解详细信息）。

> **NOTE**: 目前，插件不支持具有内部动态的自定义操作（当输出张量形状只能确定作为执行操作的结果时）。

### 有状态模型
CPU 插件支持有状态模型，且没有任何限制。

有关详细信息，请参见[有状态模型指南](@ref openvino_docs_OV_UG_network_state_intro)。

## 支持的属性
插件支持以下属性：

### 读写属性
在调用 `ov::Core::compile_model()` 之前必须设置所有参数才能生效或作为附加参数传递给 `ov::Core::compile_model()`

- ov::enable_profiling
- ov::hint::inference_precision
- ov::hint::performance_mode
- ov::hint::num_request
- ov::num_streams
- ov::affinity
- ov::inference_num_threads
- ov::intel_cpu::denormals_optimization



### 只读属性
- ov::cache_dir
- ov::supported_properties
- ov::available_devices
- ov::range_for_async_infer_requests
- ov::range_for_streams
- ov::device::full_name
- ov::device::capabilities


## 外部依赖包
对于某些性能关键深度学习操作，CPU 插件使用 oneAPI Deep Neural Network Library ([oneDNN](https://github.com/oneapi-src/oneDNN)) 中的优化实现。

@sphinxdirective
.. dropdown:: 使用 OneDNN 库中的基元实现以下操作：

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

## 优化指南

### 反向规格化数字优化
反向规格化数字是非常接近零的非零有限浮点数字，即 (0, 1.17549e-38) 和 (0, -1.17549e-38) 中的数字。在这种情况下，规格化数字编码格式无法对数字进行编码，并且会发生下溢。在很多硬件上，涉及此类数字的计算极其缓慢。

由于反向规格化数字非常接近于零。因此将反向规格化数字直接视为零是优化反向规格化数字计算的一种直接又简单的方法。由于此优化并不符合 IEEE 754 标准，如果它造成不可接受的精度下降，则可以引入属性 (`ov::intel_cpu::denormals_optimization`) 来控制此行为。如果用例中存在反向规格化数字，并且未看到精度下降或下降幅度可忽略不计，则可以将该属性设置为 `YES` 以提高性能，否则将其设置为 `NO`。如果属性未显式设置，并且应用程序也不执行任何反向规格化数字优化，则默认情况下禁用此优化。启用此属性后，OpenVINO 将在适用的所有平台上提供跨操作系统/编译器的安全优化。

在某些情况下，使用 OpenVINO 的应用程序也可以执行这种低级别反向规格化数字优化。如果通过在调用 OpenVINO 的线程开头的 MXCSR 寄存器中设置 FTZ (Flush-To-Zero) 和 DAZ (Denormals-As-Zero) 标志进行优化，则 OpenVINO 将在同一线程及子线程中继承此设置，因此无需通过属性设置。在这种情况下，应由应用程序用户负责设置的有效性和安全性。

还需指出的一点是，此属性必须在调用 `compile_model()` 前进行设置。

要启用反向规格化数字优化，应用必须将 `ov::denormals_optimization` 属性设置为 `true`：


@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_denormals.cpp
         :language: cpp
         :fragment: [ov:intel_cpu:denormals_optimization:part0]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_denormals.py
         :language: python
         :fragment: [ov:intel_cpu:denormals_optimization:part0]

@endsphinxdirective

## 另请参阅
* [支持的设备](../../../OV_Runtime_UG/supported_plugins/Supported_Devices.md)
* [优化指南](@ref openvino_docs_optimization_guide_dldt_optimization_guide_zh_CN)
* [CPU 插件开发人员文档](https://github.com/openvinotoolkit/openvino/wiki/CPUPluginDevelopersDocs)
