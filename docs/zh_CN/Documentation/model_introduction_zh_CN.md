# 模型处理简介 {#openvino_docs_model_processing_introduction_zh_CN}

每个深度学习工作流程都始于获取模型。您可以选择准备自定义模型，使用现成的解决方案并根据需要进行调整，或者从 OpenVINO™ 的 [Open Model Zoo](../../model_zoo.md) 等在线数据库下载并运行预训练网络。

[OpenVINO™ 支持多种模型格式](../MO_DG/prepare_model/convert_model/supported_model_formats_zh_CN.md)并可将其转换为它自己的 OpenVINO™ IR 格式，同时提供一个专门执行此任务的工具。

[模型优化器](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide_zh_CN.md)会读取原始模型并创建 OpenVINO™ IR 模型（.xml 和 .bin 文件），以便最终执行推理，而不会由于格式转换而造成延迟。根据需要，模型优化器可以调整模型以使其更适合推理，例如，通过[交替输入形状](../../MO_DG/prepare_model/convert_model/Converting_Model.md)、[嵌入预处理](../../MO_DG/prepare_model/Additional_Optimizations.md)和[切除训练部分](../../MO_DG/prepare_model/convert_model/Cutting_Model.md)。

完全转换模型的方法被视为默认选择，因为它支持 OpenVINO™ 的全部功能。训练后优化工具等其他转换和准备工具会使用 OpenVINO™ IR 模型格式，以进一步优化已转换模型。

ONNX 和 PaddlePaddle 模型无需转换，因为 OpenVINO™ 提供 C++ 和 Python API，可用于将这些模型直接导入 OpenVINO™ 运行时。它提供了一种可在推理应用中快速从基于框架的代码切换为基于 OpenVINO™ 的代码的便捷方法。

本节介绍如何获取和准备模型，以便使用 OpenVINO™ 获得最佳推理结果：
* [查看支持的格式及如何在项目中使用这些格式](../MO_DG/prepare_model/convert_model/supported_model_formats_zh_CN.md)。
* [将不同模型格式转换为 OpenVINO™ IR 格式](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide_zh_CN.md)。
* [通过模型下载程序和其他 OMZ 工具自动执行模型相关任务](https://docs.openvino.ai/2022.2/omz_tools_downloader.html)。

首先，您可能需要[浏览在您的项目中使用的模型数据库](../../model_zoo.md)。




