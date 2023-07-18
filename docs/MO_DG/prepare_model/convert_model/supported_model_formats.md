# Supported Model Formats {#Supported_Model_Formats}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi
   openvino_docs_MO_DG_prepare_model_convert_model_tutorials

.. meta::
   :description: In OpenVINO, ONNX, PaddlePaddle, TensorFlow and TensorFlow Lite 
                 models do not require any prior conversion, while MxNet, Caffe and Kaldi do.


**OpenVINO IR (Intermediate Representation)** - the proprietary format of OpenVINO™, benefiting from the full extent of its features.

**ONNX, PaddlePaddle, TensorFlow, TensorFlow Lite** - formats supported directly, which means they can be used with
OpenVINO Runtime without any prior conversion. For a guide on how to run inference on ONNX, PaddlePaddle, or TensorFlow,
see how to :doc:`Integrate OpenVINO™ with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`.

**MXNet, Caffe, Kaldi** - legacy formats that need to be converted to OpenVINO IR before running inference. 
The model conversion in some cases may involve intermediate steps. OpenVINO is currently proceeding 
**to deprecate these formats** and **remove their support entirely in the future**.

Refer to the following articles for details on conversion for different formats and models:

* :doc:`How to convert ONNX <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX>`
* :doc:`How to convert PaddlePaddle <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle>`
* :doc:`How to convert TensorFlow <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow>`
* :doc:`How to convert TensorFlow Lite <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite>`
* :doc:`How to convert MXNet <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet>`
* :doc:`How to convert Caffe <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe>`
* :doc:`How to convert Kaldi <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi>`

* :doc:`Conversion examples for specific models <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>`

@endsphinxdirective
