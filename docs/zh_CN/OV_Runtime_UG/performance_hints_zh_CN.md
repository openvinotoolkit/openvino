# 高级别性能提示 {#openvino_docs_OV_UG_Performance_Hints_zh_CN}

尽管 OpenVINO™ 中所有[支持的设备](supported_plugins/Device_Plugins_zh_CN.md)都提供低级别性能设置，但除了极少数情况，其他情况不建议使用它们。
在 OpenVINO™ 运行时配置性能的首选方式是使用性能提示。此解决方案面向未来且与[自动设备选择推理模式](../../OV_Runtime_UG/auto_device_selection.md)完全兼容，并且在设计时考虑了*移动性*。

这些提示还以正确的顺序设置配置的方向。这些提示没有将应用需求映射到低级别性能设置，也没有保持相关的应用逻辑来单独配置每个可能的设备，而是用单个配置键来表达目标场景，并让*设备*对自身进行配置作为响应。

以前，一定级别的自动配置是参数的*默认*值的结果。例如，在设置 `ov::streams::AUTO`（在 API 2.0 之前的版本中称为 `CPU_THROUGHPUT_AUTO`）时，CPU 流的数量是从 CPU 核心数量推导而来。然而，由此产生的流数量并未考虑要推理的模型的实际计算要求。
相反，这些提示会考虑实际模型。因此会针对每个模型单独计算最佳吞吐量的参数（基于计算与内存带宽要求和设备的功能）。

## 性能提示：延迟和吞吐量
如[优化指南](../optimization_guide/dldt_optimization_guide_zh_CN.md)中所论，有一些不同的指标与推理速度相关联。
吞吐量和延迟是衡量整体性能使用的最广泛的应用指标。

因此，为了简化设备的配置，OpenVINO™ 提供两个专用提示，即 `ov::hint::PerformanceMode::THROUGHPUT` 和 `ov::hint::PerformanceMode::LATENCY`。
特殊的 `ov::hint::PerformanceMode::UNDEFINED` 提示的作用等同于不指定任何提示。

有关使用 `benchmark_app` 进行性能测量的详细信息，请参见本文档的最后一节。

请记住，与 `ov::hint::PerformanceMode::LATENCY` 相比，典型模型在 `ov::hint::PerformanceMode::THROUGHPUT` 下的加载时间可能要长得多，并且会消耗更多内存。

## 性能提示：工作原理
在内部，每个设备都把提示的值“转换”成实际性能设置。
例如，`ov::hint::PerformanceMode::THROUGHPUT` 选择 CPU 或 GPU 流的数量。
此外，选择 GPU 的最佳批次大小，并且尽可能应用[自动批处理](../../OV_Runtime_UG/automatic_batching.md)。如需检查设备是否支持性能提示，请参见[设备/功能支持表](./supported_plugins/Device_Plugins_zh_CN.md)一文。

所得到的（设备特定的）设置可以从 `ov:Compiled_Model` 的实例中查询。  
请注意，`benchmark_app` 输出 `THROUGHPUT` 提示的实际设置。请参见以下输出示例：

   ```
    $benchmark_app -hint tput -d CPU -m 'path to your favorite model'
    ...
    [Step 8/11] Setting optimal runtime parameters
    [ INFO ] Device: CPU
    [ INFO ]   { PERFORMANCE_HINT , THROUGHPUT }
    ...
    [ INFO ]   { OPTIMAL_NUMBER_OF_INFER_REQUESTS , 4 }
    [ INFO ]   { NUM_STREAMS , 4 }
    ...
   ```

## 使用性能提示：基本 API
在下面的示例代码片段中，为 `compile_model` 的 `ov::hint::performance_mode` 属性指定了 `ov::hint::PerformanceMode::THROUGHPUT`：
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [compile_model]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [compile_model]

@endsphinxdirective

## 来自应用的其他（可选）提示
对于处理 4 个视频流的应用，传达并行闲置限制最经得起未来考验的方法是为性能提示配备可选的 `ov::hint::num_requests` 配置键，并将该键设置为 4。
如前所述，这将限制 GPU 的批次大小和 CPU 的推理流数量。因此，每个设备都使用 `ov::hint::num_requests` 将提示转换为实际设备配置选项：
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [hint_num_requests]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [hint_num_requests]

@endsphinxdirective

## 最佳推理请求数
这些提示用于假定应用查询 `ov::optimal_number_of_infer_requests` 以同时创建和运行返回的请求数：
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [query_optimal_num_requests]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [query_optimal_num_requests]

@endsphinxdirective

虽然应用可以根据需要（例如为了支持异步输入填充）自由创建更多请求，**至少要并行运行推理请求的 `ov::optimal_number_of_infer_requests`，这一点非常重要**。出于效率或设备利用率原因，建议这样做。

请记住，`ov::hint::PerformanceMode::LATENCY` 并不一定意味着使用单个推理请求。例如，多插槽 CPU 可以以与系统中的 NUMA 节点数相同的最小延迟提供尽可能多的请求。
如需使应用完全可扩展，请务必直接查询 `ov::optimal_number_of_infer_requests`。

## 倾向异步 API
推理请求的 API 提供同步执行和异步执行。`ov::InferRequest::infer()` 本质上是同步的，并且易于操作（因为它会序列化当前应用线程中的执行流）。异步将 `infer()`“拆分”成 `ov::InferRequest::start_async()` 和 `ov::InferRequest::wait()`（或回调）。如需获取更多信息，请参阅 [API 示例](../../OV_Runtime_UG/ov_infer_request.md)。
尽管同步 API 在某种程度上可能更容易上手，但建议在生产代码中使用异步（基于回调的）API。这是任何可能数量的请求（以及延迟和吞吐量情况）实现流控制的最通用和可扩展的方法。
 
## 结合提示和各个低级别设置
虽然在某种程度上牺牲了可移植性，但是可以将提示与单个设备特定的设置结合起来。
例如，使用 `ov::hint::PerformanceMode::THROUGHPUT` 准备常规配置并覆盖其任何特定值：  
@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/ov_auto_batching.cpp
       :language: cpp
       :fragment: [hint_plus_low_level]

.. tab:: Python

    .. doxygensnippet:: docs/snippets/ov_auto_batching.py
       :language: python
       :fragment: [hint_plus_low_level]


@endsphinxdirective

## 利用 Benchmark_App 测试提示性能
`benchmark_app` 具有 [C++](../../../samples/cpp/benchmark_app/README.md) 和 [Python](../../../tools/benchmark_tool/README.md) 两种版本，是评估特定设备的性能提示功能的最佳方式：
 - benchmark_app **-hint tput** -d 'device' -m 'path to your model'
 - benchmark_app **-hint latency** -d 'device' -m 'path to your model'
- 禁用提示以模拟提示出现以前的时期（强烈建议在尝试各个低级别设置之前禁用，如下面的流数量、线程等）：
  - benchmark_app **-hint none -nstreams 1**  -d 'device' -m 'path to your model'
 

### 其他资源
[支持的设备](../../OV_Runtime_UG/supported_plugins/Supported_Devices.md)
