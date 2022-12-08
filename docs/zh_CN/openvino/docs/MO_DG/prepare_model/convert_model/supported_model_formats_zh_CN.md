# 支持的模型格式{#Supported_Model_Formats_zh_CN}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi
   openvino_docs_MO_DG_prepare_model_convert_model_tutorials

@endsphinxdirective


**OpenVINO™ IR（中间表示）**- OpenVINO™ 的专有格式，受益于它的一系列功能。

**ONNX、PaddlePaddle** - 直接支持的格式，意味着它们无需提前进行任何转换即可与 OpenVINO™ 运行时结合使用。有关如何在 ONNX 和 PaddlePaddle 上运行推理的指南，请参阅如何[将 OpenVINO™ 与您的应用集成](../../../OV_Runtime_UG/integrate_with_your_application.md)。

**TensorFlow、PyTorch、MXNet、Caffe、Kaldi** - 间接支持的格式，意味着需要将它们转换为上面列出的格式之一。使用模型优化器将这些格式转换为 OpenVINO™ IR。在某些情况下，需要将其他转换器用作中间工具。

请参阅以下文章了解有关转换不同格式和模型的详细信息：

* [如何转换 ONNX](./Convert_Model_From_ONNX.md)
* [如何转换 PaddlePaddle](./Convert_Model_From_Paddle_zh_CN.md)
* [如何转换 TensorFlow](./Convert_Model_From_TensorFlow.md)
* [如何转换 PyTorch](./Convert_Model_From_PyTorch.md)
* [如何转换 MXNet](./Convert_Model_From_MxNet.md)
* [如何转换 Caffe](./Convert_Model_From_Caffe.md)
* [如何转换 Kaldi](./Convert_Model_From_Kaldi.md)

* [特定模型的转换示例](./Convert_Model_Tutorials.md)
