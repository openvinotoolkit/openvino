# 支持的模型格式 {#Supported_Model_Formats_zh_CN}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle_zh_CN

@endsphinxdirective


**OpenVINO™ IR（中间表示）**- OpenVINO™ 的专有格式，受益于它的一系列功能。

**ONNX、PaddlePaddle** - 直接支持的格式，意味着它们无需提前进行任何转换即可与 OpenVINO™ 运行时结合使用。有关如何在 ONNX 和 PaddlePaddle 上运行推理的指南，请参阅如何[将 OpenVINO™ 与您的应用集成](@ref openvino_docs_OV_UG_Integrate_OV_with_your_application)。

**TensorFlow、PyTorch、MXNet、Caffe 和 Kaldi** - 间接支持的格式，意味着需要在运行推理前先将它们转换为 OpenVINO IR。此转换通过模型优化器完成，在某些情况下可能涉及中间步骤。

请参阅以下文章了解有关转换不同格式和模型的详细信息：

* [如何转换 ONNX](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX)
* [如何转换 PaddlePaddle](./Convert_Model_From_Paddle_zh_CN.md)
* [如何转换 TensorFlow](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow)
* [如何转换 PyTorch](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch)
* [如何转换 MXNet](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet)
* [如何转换 Caffe](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe)
* [如何转换 Kaldi](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi)
* [特定模型的转换示例](@ref openvino_docs_MO_DG_prepare_model_convert_model_tutorials)
