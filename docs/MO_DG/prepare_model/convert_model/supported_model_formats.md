# Supported Model Formats {#Supported_Model_Formats}

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


**OpenVINO IR (Intermediate Representation)** - the proprietary format of OpenVINO™, benefiting from the full extent of its features.

**ONNX, PaddlePaddle, TensorFlow** - formats supported directly, which means they can be used with OpenVINO Runtime without any prior conversion. For a guide on how to run inference on ONNX, PaddlePaddle, or TensorFlow, see how to [Integrate OpenVINO™ with Your Application](../../../OV_Runtime_UG/integrate_with_your_application.md).

**MXNet, Caffe, Kaldi** - formats supported indirectly, which means they need to be converted to OpenVINO IR before running inference. The conversion is done with Model Optimizer and in some cases may involve intermediate steps.

Refer to the following articles for details on conversion for different formats and models:

* [How to convert ONNX](./Convert_Model_From_ONNX.md)
* [How to convert PaddlePaddle](./Convert_Model_From_Paddle.md)
* [How to convert TensorFlow](./Convert_Model_From_TensorFlow.md)
* [How to convert MXNet](./Convert_Model_From_MxNet.md)
* [How to convert Caffe](./Convert_Model_From_Caffe.md)
* [How to convert Kaldi](./Convert_Model_From_Kaldi.md)

* [Conversion examples for specific models](./Convert_Model_Tutorials.md)
