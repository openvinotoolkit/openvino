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


**OpenVINO IR (Intermediate Representation)** - the proprietary format of OpenVINO™, benefiting from the full extent of its features. The result of running ``ovc`` CLI tool or ``openvino.save_model`` is OpenVINO IR. All other supported formats can be converted to the IR, refer to the following articles for details on conversion:

* :doc:`How to convert PyTorch <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_PyTorch>`
* :doc:`How to convert ONNX <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_ONNX>`
* :doc:`How to convert TensorFlow <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_TensorFlow>`
* :doc:`How to convert TensorFlow Lite <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite>`
* :doc:`How to convert PaddlePaddle <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_Paddle>`

To choose the best workflow for your application, read the :doc:`Model Preparation section <openvino_docs_model_processing_introduction>`

Refer to the list of all supported conversion options in :doc:`Conversion Parameters <openvino_docs_OV_Converter_UG_Conversion_Options>`

Additional Resources
####################

* :doc:`Transition guide from the legacy to new conversion API <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>`

@endsphinxdirective
