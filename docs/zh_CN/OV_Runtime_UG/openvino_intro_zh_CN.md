# 利用 OpenVINO™ 运行时展开推理 {#openvino_docs_OV_UG_OV_Runtime_User_Guide_zh_CN}

@sphinxdirective

.. _deep learning openvino runtime:

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Working_with_devices_zh_CN
   openvino_docs_OV_UG_DynamicShapes_zh_CN
   openvino_docs_OV_UG_Performance_Hints_zh_CN
   
@endsphinxdirective

OpenVINO™ 运行时是一组包含 C 和 Python 绑定的 C++ 库，提供通用 API，以在您选择的平台上提供推理解决方案。使用 OpenVINO™ 运行时 API 读取中间表示 (IR)、ONNX 或 PaddlePaddle 模型并在首选设备上执行。

OpenVINO™ 运行时使用插件架构。它的插件是软件组件，可完全实现对特定英特尔® 硬件设备（CPU、GPU、VPU 等）进行推理。每个插件都实现统一的 API，并提供额外的硬件特定 API，用于配置设备或在 OpenVINO™ 运行时和底层插件后端之间实现 API 互操作性。
 
下面的方案展示了部署经过训练的深度学习模型的典型工作流程：

<!-- TODO: need to update the picture below with PDPD files -->
![](../../img/BASIC_FLOW_IE_C.svg)


