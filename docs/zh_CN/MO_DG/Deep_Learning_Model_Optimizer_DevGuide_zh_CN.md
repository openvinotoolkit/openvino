# 模型优化器用法 {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide_zh_CN}

@sphinxdirective

.. _deep learning model optimizer:

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_model_inputs_outputs
   openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model
   openvino_docs_MO_DG_prepare_model_Model_Optimization_Techniques 
   openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model
   openvino_docs_MO_DG_Additional_Optimization_Use_Cases
   openvino_docs_MO_DG_FP16_Compression
   openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ

@endsphinxdirective

模型优化器是一个跨平台命令行工具，有助于在训练与部署环境之间进行迁移，执行静态模型分析，并调整深度学习模型，以在端点目标设备上优化执行。

要使用该优化器，您需要采用以下支持格式之一的预训练深度学习模型：TensorFlow、PyTorch、PaddlePaddle、MXNet、Caffe、Kaldi 或 ONNX。模型优化器将模型转换为 OpenVINO™ 中间表示 (IR) 格式，您之后可以通过 [OpenVINO™ 运行时](../OV_Runtime_UG/openvino_intro_zh_CN.md)对其进行推理。

请注意，模型优化器不对模型进行推理。

下图展示了部署已训练深度学习模型的典型工作流程：

![](../../img/BASIC_FLOW_MO_simplified.svg)

其中 IR 是一对描述模型的文件：

* <code>.xml</code> - 描述网络拓扑。

* <code>.bin</code> - 包含权重和偏移二进制数据。

OpenVINO™ IR 可通过[训练后优化](../../../tools/pot/docs/Introduction.md)进一步优化，
> 以便于推理。

> **TIP**: 您还可以在 OpenVINO™ [深度学习工作台](@ ref workbench_docs_Workbench_DG_Introduction_zh_CN)中使用模型优化器。该工作台是一个采用 GUI 的基于 Web 的工具，用于对深度学习模型的性能进行优化、调优、分析、可视化和比较。

## 如何运行模型优化器

要将模型转换为 IR，您可以通过以下命令运行优化器：

```sh
mo --input_model INPUT_MODEL
```

如果开箱即用的转换（仅指定 `--input_model` 参数）未成功，请使用下文提到的参数覆盖输入形状并切割模型：

- 模型优化器提供两个参数来覆盖原始输入形状以便转换模型：`--input` 和 `--input_shape`。
   有关这些参数的更多信息，请参阅[设置输入形状](../../MO_DG/prepare_model/convert_model/Converting_Model.md)指南。

- 要切除模型中不需要的部分（如不支持的操作和训练子图），请使用 `--input` 和 `--output` 参数定义已转换模型的新输入和输出。
   有关更多详细说明，请参阅[切除模型的一些部分](../../MO_DG/prepare_model/convert_model/Cutting_Model.md)指南。

您还可以通过使用[嵌入预处理计算](../../MO_DG/prepare_model/Additional_Optimizations.md)一文中介绍的 `--mean_values`、`scales_values`、`--layout` 和其他参数在已转换模型中插入其他输入预处理子图。

模型优化器中的 `--data_type` 压缩参数支持生成 `FP16` 数据类型的 IR。有关更多详细信息，请参阅[将模型压缩为 FP16](../../MO_DG/prepare_model/FP16_Compression.md) 指南。

要获取模型优化器中可用的转换参数的完整列表，请运行以下命令：

```sh
mo --help
```

## CLI 命令的示例

下面提供了不同框架和模型优化器参数的相应示例列表：

1. 为二进制 protobuf 格式的 TensorFlow MobileNet 模型启动模型优化器：
```sh
mo --input_model MobileNet.pb
```
为 SavedModel 格式的 TensorFlow BERT 模型启动模型优化器（三个输入）。以显式方式指定输入形状。其中批次大小和序列长度分别等于 2 和 30：
```sh
mo --saved_model_dir BERT --input mask,word_ids,type_ids --input_shape [2,30],[2,30],[2,30]
```
有关更多信息，请参阅[转换 TensorFlow 模型](../../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md)指南。

2. 为 ONNX OCR 模型启动模型优化器，并以显式方式指定新输出：
```sh
mo --input_model ocr.onnx --output probabilities
```
有关更多信息，请参阅转换 ONNX 模型 (../../MO_DG/prepare_model/convert_model/Convert_Model_From_ONNX.md) 指南。

> **NOTE**: 在将 PyTorch 模型转换为 IR 之前，必须将其导出为 ONNX 格式。请参阅[转换 PyTorch 模型](../../MO_DG/prepare_model/convert_model/Convert_Model_From_PyTorch.md)了解更多信息。

3. 为 PaddlePaddle UNet 模型启动模型优化器，并为其输入应用均值/标度值归一化：
```sh
mo --input_model unet.pdmodel --mean_values [123,117,104] --scale 255
```
有关更多信息，请参阅[转换 PaddlePaddle 模型](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Paddle_zh_CN.md)指南。

4. 为 Apache MXNet SSD Inception V3 模型启动模型优化器，并为输入指定第一通道布局：
```sh
mo --input_model ssd_inception_v3-0000.params --layout NCHW
```
有关更多信息，请参阅[转换 Apache MXNet 模型](../../MO_DG/prepare_model/convert_model/Convert_Model_From_MxNet.md)指南。

5. 为 Caffe AlexNet 模型启动模型优化器，输入通道采用需要逆向转换的 RGB 格式：
```sh
mo --input_model alexnet.caffemodel --reverse_input_channels
```
有关更多信息，请参阅[转换 Caffe 模型](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Caffe.md)指南。

6. 为 Kaldi LibriSpeech nnet2 模型启动模型优化器：
```sh
mo --input_model librispeech_nnet2.mdl --input_shape [1,140]
```
有关更多信息，请参阅[转换 Kaldi 模型](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)指南。

- 要获取特定 TensorFlow、ONNX、PyTorch、Apache MXNet 和 Kaldi 模型的转换方法，请参阅[模型转换教程](../../MO_DG/prepare_model/convert_model/Convert_Model_Tutorials.md)。
- 有关 IR 的更多信息，请参阅 [OpenVINO™ 中的深度学习网络中间表示和操作集](../../MO_DG/IR_and_opsets.md)。
