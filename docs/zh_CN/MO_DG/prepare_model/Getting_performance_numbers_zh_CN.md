# 获得性能数据 {#openvino_docs_MO_DG_Getting_Performance_Numbers_zh_CN}

本指南介绍了需要注意的事项以及如何使用 benchmark_app 获取性能数据。它还解释了如何通过内部推理性能计数器和执行图来反映性能数据。最后一节包含有关使用 ITT 和英特尔® VTune™ Profiler 获取性能洞察的信息。

## 提示 1：请选择适当的操作集进行测量

使用 OpenVINO™ 运行时评估模型的性能时，需要测量适当的操作集。请记住以下提示：
- 避免包括模型加载等一次性成本。

- 单独跟踪发生在 OpenVINO™ 运行时之外的操作（例如视频解码）。

> **NOTE**: 一些图像预处理可以融入 OpenVINO™ IR 中并相应地加速。如需了解更多信息，请参阅[嵌入预处理](../../../MO_DG/prepare_model/Additional_Optimizations.md)和[通用运行时优化](../../../optimization_guide/dldt_deployment_optimization_common.md)。

## 提示 2：尝试获取可靠的数据

性能结论应基于可复制的数据。性能测量应该通过对同一例程的大量调用来完成。由于第一次迭代几乎总是比后续迭代慢得多。因此最终预测的执行时间可以使用合计值：

- 如果热身运行没有帮助或执行时间仍然变化，您可以尝试运行大量迭代，然后对结果求平均值。
- 如果时间值范围太大，请考虑几何平均值。
- 请注意过热降频和其他功率异常。设备可以处于几种不同电源状态中的一种。优化模型时，请考虑固定设备频率以获得更好的性能数据重现性。然而，端到端（应用）基准测试也应在实际操作条件下执行。

## 使用 benchmark_app 测量参考性能数据

要获取性能数据，请使用专用的 [OpenVINO™ 基准测试应用](../../../../samples/cpp/benchmark_app/README.md)样本。该样本是生成性能参考的最推荐的解决方案。
它包括许多设备特定的旋钮，但主要用途与以下用于测量 GPU 上模型的性能的命令一样简单：
```bash
$ ./benchmark_app –d GPU –m <model> -i <input>
```
测量 GPU 上的模型的性能。
或者
```bash
$ ./benchmark_app –d CPU –m <model> -i <input>
```
改为在 CPU 上执行。

每个 [OpenVINO™ 支持的设备](../../../OV_Runtime_UG/supported_plugins/Supported_Devices.md)都提供性能设置，这些设置在[基准测试应用](../../../../samples/cpp/benchmark_app/README.md)中包含命令行等效项。
虽然这些设置提供非常低级别的控制，并支持利用_特定_设备上的最佳模型性能，但建议始终首先使用 [OpenVINO™ 高级性能提示](../../OV_Runtime_UG/performance_hints_zh_CN.md)开始进行性能评估：
 - benchmark_app **-hint tput** -d 'device' -m 'path to your model'
 - benchmark_app **-hint latency** -d 'device' -m 'path to your model'

## 注意将性能与原生/框架代码进行比较

在将 OpenVINO™ 运行时性能与框架或其他参考代码进行比较时，请确保两个版本尽可能相似：

- 包装确切的推理执行（有关示例，请参阅[基准测试应用](../../../../samples/cpp/benchmark_app/README.md)）。
- 不要包括模型加载时间。
- 确保 OpenVINO™ 运行时和框架的输入相同。例如，请注意可用于填充输入的随机值。
- 在应单独跟踪任何用户端预处理的情况下，请考虑[图像预处理和转换](../../../OV_Runtime_UG/preprocessing_overview.md)。
- 适用时，请利用[动态形状支持](../../OV_Runtime_UG/ov_dynamic_shapes_zh_CN.md)。
- 如果可能，应要求达到相同的精度。例如，TensorFlow 支持执行 `FP16`。因此在进行比较时，确保也使用 `FP16` 来测试 OpenVINO™ 运行时。

## 来自内部推理性能计数器和执行图的数据<a name="performance-counters"></a>
可以通过设备特定的性能计数器和/或执行图获得有关推理性能分解的更详细的洞察。
[C++](../../../../samples/cpp/benchmark_app/README.md) 和 [Python](../../../../tools/benchmark_tool/README.md) 版本的 `benchmark_app` 都支持输出内部执行分解的 `-pc` 命令行参数。

例如，下表是在 [CPU 插件](../../OV_Runtime_UG/supported_plugins/CPU_zh_CN.md)上量化的 [ResNet-50 的 TensorFlow 实现](https://github.com/openvinotoolkit/open_model_zoo/tree/releases/2022/2/models/public/resnet-50-tf)的性能计数器的一部分。
请记住，由于设备是 CPU。因此 `realTime` 挂钟和 `cpu` 时间层相同。有关层精度的信息也存储在性能计数器中。

| layerName                                                 | execStatus | layerType    | execType             | realTime (ms) | cpuTime (ms) |
| --------------------------------------------------------- | ---------- | ------------ | -------------------- | ------------- | ------------ |
| resnet\_model/batch\_normalization\_15/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_1x1\_I8 | 0.377         | 0.377        |
| resnet\_model/conv2d\_16/Conv2D/fq\_input\_0              | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/batch\_normalization\_16/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_I8      | 0.499         | 0.499        |
| resnet\_model/conv2d\_17/Conv2D/fq\_input\_0              | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/batch\_normalization\_17/FusedBatchNorm/Add | EXECUTED   | Convolution  | jit\_avx512\_1x1\_I8 | 0.399         | 0.399        |
| resnet\_model/add\_4/fq\_input\_0                         | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |
| resnet\_model/add\_4                                      | NOT\_RUN   | Eltwise      | undef                | 0             | 0            |
| resnet\_model/add\_5/fq\_input\_1                         | NOT\_RUN   | FakeQuantize | undef                | 0             | 0            |


此表的 `exeStatus` 列包括以下可能的值：
- `EXECUTED` - 层通过独立的基元执行。
- `NOT_RUN` - 层不是通过独立基元执行的，或者与其他操作融合并在另一个层基元中执行。
   
此表的 `execType` 列包括具有特定后缀的推理基元。层可以具有以下标记：
* `I8` 后缀适用于具有 8 位数据类型输入并以 8 位精度计算的层。
* `FP32` 后缀适用于以 32 位精度计算的层。

所有 `Convolution` 层都以 `int8` 精度执行。其余层使用操作后优化融合到卷积中，如 [CPU 设备](../../OV_Runtime_UG/supported_plugins/CPU_zh_CN.md)中所述。
这包含层名称（如 OpenVINO™ IR 中所示）、层类型和执行统计信息。

此外，两个 `benchmark_app` 版本都支持 `exec_graph_path` 命令行选项。它要求 OpenVINO™ 在每层输出相同的执行统计信息，但以插件特定的 [Netron 可查看](https://netron.app/)图形的形式输出到指定文件中。

特别是在对[延迟](../../../optimization_guide/dldt_deployment_optimization_latency.md)进行性能调试时，请注意计数器不反映在 `plugin/device/driver/etc` 队列中花费的时间。如果计数器之和与推理请求的延迟差异太大，请考虑使用较少的推理请求进行测试。例如，运行具有多个请求的单个 [OpenVINO™ 流](../../../optimization_guide/dldt_deployment_optimization_tput.md)将产生与运行单个推理请求几乎相同的计数器，而实际延迟可能大不相同。

最后，性能计数器和执行图的性能统计信息是平均值。因此应仔细测量[动态形状输入](../../OV_Runtime_UG/ov_dynamic_shapes_zh_CN.md)的此类数据，最好通过隔离特定形状并循环多次执行，以收集可靠的数据。

## 使用 ITT 获得性能洞察

一般来说，OpenVINO™ 及其各个插件都大量使用英特尔® 检测和跟踪技术 (ITT)。因此，您还可以在启用 ITT 的情况下从源代码编译 OpenVINO，并使用[英特尔® VTune™ Profiler](https://software.intel.com/en-us/vtune) 等工具在时间线视图上获取详细的推理性能分解和应用级性能的其他洞察。
