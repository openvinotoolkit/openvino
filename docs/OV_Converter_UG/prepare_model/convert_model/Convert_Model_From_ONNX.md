# Converting an ONNX Model {#openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_ONNX}

@sphinxdirective

.. meta::
   :description: Learn how to convert a model from the
                 ONNX format to the OpenVINO Model.

Introduction to ONNX
####################

`ONNX <https://github.com/onnx/onnx>`__ is a representation format for deep learning models that allows AI developers to easily transfer models between different frameworks.

.. note:: ONNX model file can be loaded by `openvino.Core.read_model` or `openvino.Core.compile_model` methods by OpenVINO runtime API without preparing OpenVINO IR first. Refer to the :doc:`inference example <openvino_docs_OV_UG_Integrate_OV_with_your_application>` for more details. Using ``openvino.convert_model`` is still recommended if model load latency matters for the inference application.

Converting an ONNX Model
########################

This page provides instructions on model conversion from the ONNX format to the OpenVINO IR format. To use model conversion API, install OpenVINO Development Tools by following the :doc:`installation instructions <openvino_docs_install_guides_install_dev_tools>`.

Model conversion process assumes you have an ONNX model that was directly downloaded from a public repository or converted from any framework that supports exporting to the ONNX format.

To convert an ONNX model, run model conversion with the path to the input model ``.onnx`` file:

.. tab-set::

   .. tab-item:: Python
      :sync: py

         import openvino as ov
         ov.convert_model('your_model_file.onnx')

   .. tab-item:: CLI
      :sync: cli

         ovc your_model_file.onnx

External Data Files
#########################

ONNX models may consist of multiple files when the total size of the model exceeds 2GB allowed by Protobuf. According to `ONNX< https://github.com/onnx/onnx/blob/main/docs/ExternalData.md>`, instead of a single file, such a model is represented as one file with `.onnx` extension and multiple separate files with external data which are located in the same directory where the main `.onnx` file is located or in another directory.

OpenVINO model conversion API supports ONNX models with external data representation. In this case, only the main file with `.onnx` extension should be passed as `ovc` or `openvino.convert_model` parameter while other files will be found and loaded automatically during the mode conversion. The resulting OpenVINO model represented as IR in the filesystem will have the usual structure with a single `.xml` file and a single `.bin` file where all the original model weights are copied and packed together.

Supported ONNX Layers
#####################

For the list of supported standard layers, refer to the :doc:`Supported Operations <openvino_resources_supported_operations_frontend>` page.

Additional Resources
####################

**TODO: LINK TO NOTEBOOKS** page for a set of tutorials providing step-by-step instructions for converting specific ONNX models.

@endsphinxdirective
