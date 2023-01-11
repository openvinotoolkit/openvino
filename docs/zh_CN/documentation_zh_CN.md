# 文档 {#documentation_zh_CN}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :caption: 转换和准备模型
   :hidden:

   openvino_docs_model_processing_introduction_zh_CN
   Supported_Model_Formats_zh_CN
   openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide_zh_CN


.. toctree::
   :maxdepth: 1
   :caption: 部署推理
   :hidden:

   openvino_docs_OV_UG_OV_Runtime_User_Guide_zh_CN
   openvino_2_0_transition_guide_zh_CN
   openvino_docs_deployment_guide_introduction_zh_CN
   openvino_deployment_guide_zh_CN
   openvino_inference_engine_tools_compile_tool_README_zh_CN


.. toctree::
   :maxdepth: 1
   :caption: 性能调优
   :hidden:

   openvino_docs_optimization_guide_dldt_optimization_guide_zh_CN
   openvino_docs_MO_DG_Getting_Performance_Numbers_zh_CN
   openvino_docs_model_optimization_guide_zh_CN
   openvino_docs_deployment_optimization_guide_dldt_optimization_guide_zh_CN
   openvino_docs_performance_benchmarks_zh_CN


.. toctree::
   :maxdepth: 1
   :caption: OpenVINO™ 工具套件的 Web 图形界面  
   :hidden:

   workbench_docs_Workbench_DG_Introduction_zh_CN


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 媒体处理与计算机视觉库

   英特尔® Deep Learning Streamer <openvino_docs_dlstreamer_zh_CN>
   openvino_docs_gapi_gapi_intro_zh_CN


.. toctree::
   :maxdepth: 1
   :caption: 插件
   :hidden:

   openvino_ecosystem_zh_CN
   ovms_what_is_openvino_model_server_zh_CN
   ovtf_integration_zh_CN
   ote_documentation_zh_CN
   ovsa_get_started_zh_CN

.. toctree::
   :maxdepth: 1
   :caption: OpenVINO 可扩展性
   :hidden:

   openvino_docs_Extensibility_UG_Intro_zh_CN
   openvino_docs_transformations_zh_CN
   OpenVINO 插件开发人员指南 <openvino_docs_ie_plugin_dg_overview_zh_CN>
   
.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 安全使用 OpenVINO™ 工具套件
   
   openvino_docs_security_guide_introduction_zh_CN
   openvino_docs_security_guide_workbench_zh_CN
   openvino_docs_OV_UG_protecting_model_guide_zh_CN
   ovsa_get_started_zh_CN

@endsphinxdirective

本节提供参考文档，指导您使用 OpenVINO™ 工具套件开发您自己的深度学习应用。为了使这些文档充分发挥作用，请先阅读[入门](../get_started.md)指南。

## 转换和准备模型
在[模型下载器](@ref omz_tools_downloader)和[模型优化器](MO_DG/Deep_Learning_Model_Optimizer_DevGuide_zh_CN.md)指南中，您将学习如何下载预训练模型并对其进行转换，以便与 OpenVINO™ 工具套件配合使用。您可以提供自己的模型，也可以从 [Open Model Zoo](../model_zoo.md) 中提供的众多模型中选择公共模型或英特尔模型。

## 部署推理
[OpenVINO™ 运行时用户指南](../OV_Runtime_UG/openvino_intro.md)解释了创建您自己的应用以使用 OpenVINO™ 工具套件运行推理的过程。[API 参考](../api_references.html)定义了面向 Python、C++ 和 C 的 OpenVINO 运行时 API。OpenVINO 运行时 API 可用于创建 OpenVINO™ 推理应用，使用增强的操作集和其他功能。编写应用后，您可以根据[使用 OpenVINO 进行部署](./OV_Runtime_UG/deployment/deployment_intro_zh_CN.md)，将其部署到目标设备。

## 性能调优
该工具套件提供[性能优化指南](optimization_guide/dldt_optimization_guide_zh_CN.md)和实用程序，帮助您的应用获得卓越性能，包括[精度检查器](@ref omz_tools_accuracy_checker)、[训练后优化工具](@ref pot_introduction)和其他用于精度测量、性能基准测试和应用调优的工具。

## OpenVINO™ 工具套件的 Web 图形界面
您可以选择使用 [OpenVINO™ 深度学习工作台](@ref workbench_docs_Workbench_DG_Introduction_zh_CN)。这是一个基于 Web 的工具，可指导您执行转换、测量、优化和部署模型的过程。此工具还可作为工具套件的低门槛入门工具，并提供各种有用的交互图表，以便了解性能。

## 媒体处理与计算机视觉库

OpenVINO™ 工具套件还可与以下媒体处理框架和库配合使用：

* [英特尔® Deep Learning Streamer（英特尔® DL Streamer）](@ref openvino_docs_dlstreamer_zh_CN)- 基于 GStreamer 的流媒体分析框架，用于创建针对英特尔硬件平台进行优化的复杂媒体分析管道。请访问英特尔® DL Streamer [文档](https://dlstreamer.github.io/)网站了解更多信息。
* [英特尔® oneAPI Video Processing Library (OneVPL)](https://www.intel.com/content/www/us/en/develop/documentation/oneapi-programming-guide/top/api-based-programming/intel-oneapi-video-processing-library-onevpl.html) - 用于进行视频解码、编码和处理，以在 CPU、GPU 和其他加速器上构建可移植媒体管道的编程接口。

您还可以使用优化版本的 [OpenCV](https://opencv.org/) 在您的应用中添加计算机视觉功能。
