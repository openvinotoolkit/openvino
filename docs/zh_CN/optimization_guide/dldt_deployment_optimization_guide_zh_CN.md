# 运行时推理优化 {#openvino_docs_deployment_optimization_guide_dldt_optimization_guide_zh_CN}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_deployment_optimization_guide_common
   openvino_docs_deployment_optimization_guide_latency
   openvino_docs_deployment_optimization_guide_tput
   openvino_docs_deployment_optimization_guide_tput_advanced
   openvino_docs_deployment_optimization_guide_internals

@endsphinxdirective

运行时优化（也称为部署优化）侧重于调优推理参数和执行方法（例如，同时执行最佳数量的请求）。与模型级优化不同，它们高度依赖于自身针对的硬件和用例，并往往需要付出代价。
`ov::hint::inference_precision` 是一种“典型的运行时配置”，它会以准确性换取性能，允许在量化原始 `fp32` 模型后对仍处于 `fp32` 的层执行 `fp16/bf16`。

因此，优化应从定义用例开始。例如，如果数据中心在通宵任务中处理数百万个样本，则吞吐量可以优先于延迟。另一方面，在实际使用情况下，可能会降低吞吐量，换取以最低延迟来交付结果。还可能存在组合场景，其目标是尽可能提高吞吐量，同时保持特定的延迟阈值。

了解全堆栈应用如何以“端到端”的方式使用推理组件也至关重要。例如，了解需要编排哪些阶段以节省专门用于提取和准备输入数据的工作负载。

有关此主题的更多信息，请参阅以下文章：
* [支持的功能（按设备）](@ref features_support_matrix_zh_CN)、
* [使用 OpenVINO™ 对输入进行预处理](@ref inputs_pre_processing)。
* [异步 API](@ref async_api)。
* [惯用语“get_tensor”](@ref tensor_idiom)。
* 对于大小可变的输入，应考虑使用[动态形状](../OV_Runtime_UG/ov_dynamic_shapes_zh_CN.md)。

请参阅[延迟](../../optimization_guide/dldt_deployment_optimization_latency.md)和[吞吐量](../../optimization_guide/dldt_deployment_optimization_tput.md)优化指南了解**特定于用例的优化**

## 编写性能可移植推理应用
虽然在 OpenVINO™ 运行时执行的推理可配置为采用众多低级别性能设置，但在多数情况下不建议这样做。首先，通过此类调整实现最佳性能需要深入了解设备架构和推理引擎。


其次，这类优化可能并不适用于其他设备-模型组合。换言之，在不同条件下使用时，一组执行参数很可能会产生不同的性能。例如：
* CPU 和 GPU 都支持[流](../../optimization_guide/dldt_deployment_optimization_tput_advanced.md)的概念，但它们推断出的最佳数量截然不同。
* 即使在相同类型的设备中，也可能会将不同的执行配置视为最优，例如 CPU 的指令集或核心数量以及 GPU 的批次大小。
* 考虑到计算与内存带宽对比、推理精度和可能的模型量化等因素，不同模型会具有不同的最优参数配置。
* 执行“调度”会极大地影响性能并高度依赖于设备，例如，批处理、组合多个输入以实现最佳吞吐量等面向 GPU 的优化[并不总是完全适用于 CPU](../../optimization_guide/dldt_deployment_optimization_internals.md)。
 
 
为进一步简化配置流程并提高其性能优化的可移植性，已引入了[性能提示](../OV_Runtime_UG/performance_hints_zh_CN.md)选项。它包括两个专门针对**延迟**或**吞吐量**的高级“预设”，而且从本质上来说，它使执行细节变得无关紧要。

“性能提示”功能将使配置对应用透明，例如，预计是否需要显式（应用端）批处理或流，并帮助为不同输入源并行处理单独的推理请求


## 其他资源

* [使用异步 API 且并行运行多个推理请求以利用吞吐量](@ref throughput_app_design)。
* [特定设备的吞吐量方法实现细节](../../optimization_guide/dldt_deployment_optimization_internals.md)
* [有关吞吐量的详细信息](../../optimization_guide/dldt_deployment_optimization_tput.md)
* [有关延迟的详细信息](../../optimization_guide/dldt_deployment_optimization_latency.md)
* [API 示例和详细信息](../OV_Runtime_UG/performance_hints_zh_CN.md)。
