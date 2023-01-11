# 推理设备支持 {#openvino_docs_OV_UG_Working_with_devices_zh_CN}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_supported_plugins_CPU_zh_CN
   openvino_docs_OV_UG_supported_plugins_GPU_zh_CN

@endsphinxdirective

OpenVINO™ 运行时可以使用以下设备类型来推理深度学习模型：

* [CPU](CPU_zh_CN.md)    
* [GPU](GPU_zh_CN.md)    
* [VPU](@ref openvino_docs_OV_UG_supported_plugins_VPU)   
* [GNA](@ref openvino_docs_OV_UG_supported_plugins_GNA)   
* [Arm® CPU](@ref openvino_docs_OV_UG_supported_plugins_ARM_CPU)     

有关更详细的硬件列表，请参见[支持的设备](../../../OV_Runtime_UG/supported_plugins/Supported_Devices.md)

对于与我们用于基准测试的设备类似的设备，可以使用[英特尔® DevCloud for the Edge](https://devcloud.intel.com/edge/)（一种可以访问英特尔® 硬件的远程开发环境）和最新版本的英特尔® 发行版 OpenVINO™ 工具套件进行访问。[了解更多信息](https://devcloud.intel.com/edge/get_started/devcloud/)或[在此处注册](https://inteliot.force.com/DevcloudForEdge/s/)。


@anchor features_support_matrix_zh_CN
## 功能支持表
下表展示了 OpenVINO™ 器件插件支持的关键功能。

| 功能 | [CPU](CPU_zh_CN.md) | [GPU](GPU_zh_CN.md) | [GNA](@ref openvino_docs_OV_UG_supported_plugins_GNA) | [Arm® CPU](@ref openvino_docs_OV_UG_supported_plugins_ARM_CPU) |
| ---------- | --- | --- | --- | --- |
| [异构执行](../../../OV_Runtime_UG/hetero_execution.md) | 是 | 是 | 否 | 是 |
| [多设备执行](../../../OV_Runtime_UG/multi_device.md) | 是 | 是 | 部分 | 是 |
| [自动批处理](../../../OV_Runtime_UG/automatic_batching.md) | 否 | 是 | 否 | 否 |
| [多流执行](../../../optimization_guide/dldt_deployment_optimization_tput.md) | 是 | 是 | 否 | 是 |
| [模型缓存](../../../OV_Runtime_UG/Model_caching_overview.md) | 是 | 部分 | 是 | 否 |
| [动态形状](../ov_dynamic_shapes_zh_CN.md) | 是 | 部分 | 否 | 否 |
| [导入/导出](../../compile_tool/README_zh_CN.md) | 是 | 否 | 是 | 否 |
| [预处理加速](../../../OV_Runtime_UG/preprocessing_overview.md) | 是 | 是 | 否 | 部分 |
| [有状态模型](../../../OV_Runtime_UG/network_state_intro.md) | 是 | 否 | 是 | 否 |
| [扩展性](@ref openvino_docs_Extensibility_UG_Intro_zh_CN) | 是 | 是 | 否 | 否 |

有关插件特定功能限制的更多详细信息，请参见相应的插件页面。

## 枚举可用设备
OpenVINO™ 运行时 API 具有枚举设备及其功能的专用方法。请参阅 [Hello 查询设备 C++ 样本](../../../../samples/cpp/hello_query_device/README.md)。这是样本的示例输出（仅截断为设备名称）：

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

枚举设备并与多设备配合使用的简单编程方式如下：

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: ../../../snippets/MULTI2.cpp
       :language: cpp
       :fragment: [part2]

@endsphinxdirective

除了典型的“CPU”、“GPU”、“HDDL”等之外，当设备的多个实例可用时，名称会更有限定性。例如，在 hello_query_sample 中这样枚举两个英特尔® Movidius™ Myriad™ X 电脑棒。
```
...
    Device: MYRIAD.1.2-ma2480
...
    Device: MYRIAD.1.4-ma2480
```

因此，使用这两者的显式配置将是“MULTI:MYRIAD.1.2-ma2480,MYRIAD.1.4-ma2480”。因此，循环遍历“MYRIAD”类型的所有可用设备的代码如下：

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: ../../../snippets/MULTI3.cpp
       :language: cpp
       :fragment: [part3]

@endsphinxdirective



