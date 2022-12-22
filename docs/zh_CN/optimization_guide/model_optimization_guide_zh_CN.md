# 模型优化指南 {#openvino_docs_model_optimization_guide_zh_CN}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   pot_introduction
   tmo_introduction
  （试验性）保护模型 <pot_ranger_README>

@endsphinxdirective

模型优化是一个可选的离线步骤，通过应用量化、修剪、预处理优化等特殊优化方法来提升模型的最终性能。OpenVINO™ 在模型开发的不同步骤中提供了几种工具来优化模型：
@sphinxdirective

- :ref:`Model Optimizer <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide_zh_CN>`在默认情况下可以对模型实现大多数优化参数。但您可以自由配置均值/标度值、批次大小、RGB 和 BGR 输入通道对比以及其他参数，加快模型的预处理 (:ref:`Embedding Preprocessing Computation <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>`)。

- :ref:`Post-training Optimization w/ POT <pot_introduction>` 旨在通过应用不需要模型重新训练或微调的训练后方法（如训练后 8 位量化）来优化深度学习模型的推理。

- :ref:`Training-time Optimization w/ NNCF <tmo_introduction>`是一套用于在 PyTorch 和 TensorFlow 2.x 等深度学习框架内优化训练时间模型的高级方法。它支持量化感知训练和过滤器修剪等方法。经过 NNCF 优化的模型可以利用所有可用工作流程使用 OpenVINO 进行推理。

@endsphinxdirective


## 详细工作流程：
要了解您需要哪个开发优化工具，请参阅下图：

![](../../img/DEVELOPMENT_FLOW_V3_crunch.svg)

训练后方法在优化模型时可实现的精度和性能兼顾水平有限。在这种情况下，可以选择使用 NNCF 进行训练时间优化。

使用上述工具优化模型之后，就可以通过常规 OpenVINO 推理工作流程使用该模型进行推理。不需要更改推理代码。

![](../../img/WHAT_TO_USE.svg)

训练后方法可实现的精度有限，在某些情况下精度可能会下降。  在这种情况下，使用 NNCF 优化训练时间可能会得到更好的结果。

使用上述工具优化模型之后，即可通过常规 OpenVINO™ 推理工作流程使用该模型进行推理。不需要更改代码。

如果您不熟悉模型优化方法，请参阅[训练后方法](@ref pot_introduction)。

## 其他资源
- [部署优化](./dldt_deployment_optimization_guide_zh_CN.md)