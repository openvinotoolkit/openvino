.. _documentation_zh_CN:

文档
====================

.. toctree::
   :maxdepth: 1
   :caption: 转换和准备模型
   :hidden:


.. toctree::
   :maxdepth: 1
   :caption: 部署推理
   :hidden:

   openvino_intro_zh_CN

   deployment_guide_introduction_zh_CN
   deployment_guide_zh_CN

.. toctree::
   :maxdepth: 1
   :caption: 性能调优
   :hidden:

.. toctree::
   :maxdepth: 1
   :caption: OpenVINO™ 工具套件的 Web 图形界面  
   :hidden:


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 媒体处理与计算机视觉库

   英特尔® Deep Learning Streamer <dlstreamer_zh_CN>

.. toctree::
   :maxdepth: 1
   :caption: 插件
   :hidden:

   ovtf_integration_zh_CN

.. toctree::
   :maxdepth: 1
   :caption: OpenVINO 可扩展性
   :hidden:


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: 安全使用 OpenVINO™ 工具套件






本节提供参考文档，指导您使用 OpenVINO™ 工具套件开发您自己的深度学习应用。为了使这些文档充分发挥作用，请先阅读 
`入门 <https://docs.openvino.ai/2022.2/get_started.html>`_ 指南。

转换和准备模型
--------------------

在 `模型下载器 <https://docs.openvino.ai/2022.2/omz_tools_downloader.html>`_ 和
:doc:`模型优化器 <MO_DG/Deep_Learning_Model_Optimizer_DevGuide_zh_CN.rst>`指南中，
您将学习如何下载预训练模型并对其进行转换，以便与 OpenVINO™ 工具套件配合使用。您可以提供自己的模型，
也可以从 [Open Model Zoo](https://docs.openvino.ai/2022.2/model_zoo.html) 中提供的众多模型中选择公共模型或英特尔模型。

部署推理
--------------------

`OpenVINO™ 运行时用户指南 <https://docs.openvino.ai/2022.2/openvino_docs_OV_UG_OV_Runtime_User_Guide.html>`_ 解释了创建您自己的应用以使用 
OpenVINO™ 工具套件运行推理的过程。`API 参考 <https://docs.openvino.ai/2022.2/api/api_reference.html>`_ 定义了面向 Python、C++ 和 C 的 
OpenVINO 运行时 API。OpenVINO 运行时 API 可用于创建 OpenVINO™ 推理应用，使用增强的操作集和其他功能。编写应用后，您可以根据
:doc:`使用 OpenVINO 进行部署 <deployment_guide_zh_CN.rst>`，将其部署到目标设备。








`API 参考 <https://docs.openvino.ai/2022.2/api/api_reference.html>`_


:doc:`使用 OpenVINO 进行部署 <deployment_intro_zh_CN.rst>`