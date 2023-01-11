# OpenVINO™ 深度学习工作台概述 {#workbench_docs_Workbench_DG_Introduction_zh_CN}

深度学习工作台 (DL Workbench) 是官方的 OpenVINO™ 图形界面，旨在使预训练深度学习计算机视觉和自然语言处理模型的生成变得更加容易。

在您的浏览器中最大限度地缩短神经模型的推理到部署工作流程时间：导入模型，分析其性能和准确性，可视化输出，优化并在几分钟内使最终模型部署就绪。深度学习工作台将带您了解完整的 OpenVINO™ 工作流程，让您有机会了解各种工具套件组件。
 
![](../../Documentation/img/openvino_dl_wb.png)


@sphinxdirective

.. link-button:: workbench_docs_Workbench_DG_Start_DL_Workbench_in_DevCloud
    :type: ref
    :text: 在英特尔® DevCloud 中运行深度学习工作台
    :classes: btn-primary btn-block

@endsphinxdirective

深度学习工作台使您能够获得详细的性能评估，探索推理配置并获得优化模型以准备部署在各种英特尔® 配置上，例如客户端和服务器 CPU、英特尔® 处理器显卡 (GPU)、英特尔® Movidius™ 神经电脑棒 2 (NCS 2) 和采用英特尔® Movidius™ 视觉处理器的英特尔® Vision Accelerator Design。

DL Workbench 还提供 [JupyterLab 环境](https://docs.openvino.ai/2022.2/workbench_docs_Workbench_DG_Jupyter_Notebooks.html#doxid-workbench-docs-workbench-d-g-jupyter-notebooks)，帮助您快速开始使用 OpenVINO™ API 和命令行界面 (CLI)。按照为您的模型创建的完整 OpenVINO 工作流程进行操作，并了解不同的工具套件组件。

## 用户目标

深度学习工作台根据您的深度学习之旅的阶段帮助您实现目标。

如果您是深度学习领域的初学者，深度学习工作台为您提供学习机会：
* 了解什么是神经网络、神经网络工作原理以及如何检查其架构。
* 在生产前学习神经网络分析和优化的基础知识。
* 熟悉 OpenVINO™ 生态系统及其主要组件，而无需在您的系统上安装。

如果您在神经网络方面有足够的经验，深度学习工作台会为您提供一个方便的 web 界面来优化您的模型，并为生产做好准备：
* 测量和解释模型性能。
* 调整模型以增强性能。
* 分析模型的质量并可视化输出。

## 通用工作流程

下图说明了典型的深度学习工作台的工作流程。单击以查看全尺寸图像：

![](../../img/openvino_dl_wb_diagram_overview.svg)

在深度学习工作台用户界面中快速了解工作流程：

![](../../img/openvino_dl_wb_workflow.gif)

## OpenVINO™ 工具套件组件

深度学习工作台基于网页的直观交互界面让您轻松使用 OpenVINO™ 工具套件的各组件：

| 组件 | 描述 |
|------------------|------------------|
| [Open Model Zoo](https://docs.openvino.ai/2022.2/omz_tools_downloader.html) | 访问一系列高质量预训练深度学习[公共](https://docs.openvino.ai/2022.2/omz_models_group_public.html)和[英特尔训练的](https://docs.openvino.ai/2022.2/omz_models_group_intel.html)模型（经过训练以解决各种不同的任务）。 |
| [模型优化器](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide_zh_CN) | 将在支持的框架中训练的模型优化和转换为 IR 格式。<br>支持的框架包括 TensorFlow\*、Caffe\*、Kaldi\*、MXNet\* 和 ONNX\* 格式。 |
| [基准测试工具](https://docs.openvino.ai/2022.2/openvino_inference_engine_tools_benchmark_tool_README.html) | 估计支持设备上的深度学习模型推理性能。 |
| [精度检查器](https://docs.openvino.ai/2022.2/omz_tools_accuracy_checker.html) | 通过收集一个或多个指标值来评估模型的精度。 |
| [训练后优化工具](https://docs.openvino.ai/2022.2/pot_README.html) | 优化预训练模型，将模型精度从浮点精度（FP32 或 FP16）降低到整数精度 (INT8)，无需重新训练或微调模型。 |


@sphinxdirective

.. link-button:: workbench_docs_Workbench_DG_Start_DL_Workbench_in_DevCloud
    :type: ref
    :text: 在英特尔® DevCloud 中运行深度学习工作台
    :classes: btn-outline-primary 

@endsphinxdirective

## 联系我们

* [深度学习工作台 GitHub 存储库](https://github.com/openvinotoolkit/workbench)

* [英特尔社区论坛上的深度学习工作台](https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/bd-p/distribution-openvino-toolkit)

* [深度学习工作台网格聊天](https://gitter.im/dl-workbench/general?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&content=body)
