# Supported Model Formats {#Supported_Model_Formats}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_PyTorch
   openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_TensorFlow
   openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_ONNX
   openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite
   openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_Paddle

**OpenVINO IR (Intermediate Representation)** - the proprietary format of OpenVINO™, benefiting from the full extent of its features. The result of running `ovc` CLI tool or `openvino.save_model` is OpenVINO IR.

**ONNX, PaddlePaddle, TensorFlow, TensorFlow Lite** - file formats can be used with
OpenVINO Runtime  without any prior conversion and can be loaded with `openvino.Core.read_model` and `openvino.Core.compile_model`. For a guide on how to run inference on ONNX, PaddlePaddle, or TensorFlow,
see how to :doc:`Integrate OpenVINO™ with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`.


Refer to the following articles for details on conversion for different formats and models:

* :doc:`How to convert ONNX <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_ONNX>`
* :doc:`How to convert PaddlePaddle <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_Paddle>`
* :doc:`How to convert TensorFlow <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_TensorFlow>`
* :doc:`How to convert TensorFlow Lite <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite>`


@endsphinxdirective
