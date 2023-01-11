# 推理引擎插件库概述 {#openvino_docs_ie_plugin_dg_overview_zh_CN}


推理引擎的插件架构支持开发和插入不同设备专用的独立推理解决方案。从物理上讲，插件用导出支持创建新插件实例的单个 `CreatePluginEngine` 函数的动态库来表示。

推理引擎插件库
-----------------------

推理引擎插件动态库包含几个主要组件：

1. [插件类](@ref openvino_docs_ie_plugin_dg_plugin)：
   - 提供有关特定类型设备的信息。
   - 可以创建[可执行网络](@ref openvino_docs_ie_plugin_dg_executable_network)实例。该实例表示特定设备的神经网络后端特定图形结构，与独立于后端的 InferenceEngine::ICNNNetwork 接口相反。
   - 可以将已编译图形结构从输入流导入到[可执行网络](@ref openvino_docs_ie_plugin_dg_executable_network)对象。
2. [可执行网络类](@ref openvino_docs_ie_plugin_dg_executable_network)：
   - 是为特定设备编译的执行配置，并考虑该设备的功能。
   - 保存对特定设备的引用和此设备的任务执行器。
   - 可以创建几个[推理请求](@ref openvino_docs_ie_plugin_dg_infer_request)实例。
   - 可以将特定于内部后端的图形结构导出到输出流。
3. [推理请求类](@ref openvino_docs_ie_plugin_dg_infer_request)：
   - 串行运行推理管道。
   - 可以提取性能计数器，以便分析推理管道的执行性能。
4. [异步推理请求类](@ref openvino_docs_ie_plugin_dg_async_infer_request)：
   - 包装[推理请求](@ref openvino_docs_ie_plugin_dg_infer_request)类，并根据特定于设备的管道结构在几个任务执行器上并行运行管道阶段。

> **NOTE**: 本文档基于 `Template` 插件编写，它演示了插件开发细节。在 `<dldt source dir>/docs/template_plugin` 查找 `Template` 的完整代码，它完全兼容并保持最新。

详细指南
-----------------------

* 使用 CMake [构建](@ref openvino_docs_ie_plugin_dg_plugin_build)插件库
* 插件及其组件[测试](@ref openvino_docs_ie_plugin_dg_plugin_testing)
* [量化网络](@ref openvino_docs_ie_plugin_dg_quantized_networks)
* [低精度转换](@ref openvino_docs_OV_UG_lpt)指南
* [编写 OpenVINO™ 转换](@ref openvino_docs_transformations_zh_CN)指南

API 参考
-----------------------

* [推理引擎插件 API](@ref ie_dev_api)
* [推理引擎转换 API](@ref ie_transformation_api)
