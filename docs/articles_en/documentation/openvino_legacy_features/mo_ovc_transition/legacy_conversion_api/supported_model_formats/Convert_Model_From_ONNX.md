# Converting an ONNX Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX}

@sphinxdirective

.. meta::
   :description: Learn how to convert a model from the 
                 ONNX format to the OpenVINO Intermediate Representation.

.. note:: ONNX models are supported via FrontEnd API. You may skip conversion to IR and read models directly by OpenVINO runtime API. Refer to the :doc:`inference example <openvino_docs_OV_UG_Integrate_OV_with_your_application>` for more details. Using ``convert_model`` is still necessary in more complex cases, such as new custom inputs/outputs in model pruning, adding pre-processing, or using Python conversion extensions.

Converting an ONNX Model
########################

The model conversion process assumes you have an ONNX model that was directly downloaded from a public repository or converted from any framework that supports exporting to the ONNX format.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      To convert an ONNX model, run ``convert_model()`` method with the path to the ``<INPUT_MODEL>.onnx`` file:

      .. code-block:: py
         :force:

         ov_model = convert_model("<INPUT_MODEL>.onnx")
         compiled_model = core.compile_model(ov_model, "AUTO")

      .. important::

         The ``convert_model()`` method returns ``ov.Model`` that you can optimize, compile, or save to a file for subsequent use.

   .. tab-item:: CLI
      :sync: cli

      You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

      .. code-block:: sh

         mo --input_model <INPUT_MODEL>.onnx


There are no ONNX-specific parameters, so only framework-agnostic parameters are available to convert your model. For details, see the *General Conversion Parameters* section in the :doc:`Converting a Model to Intermediate Representation (IR) <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>` guide.

Supported ONNX Layers
#####################

For the list of supported standard layers, refer to the :doc:`Supported Operations <openvino_resources_supported_operations_frontend>` page.

Additional Resources
####################

See the :doc:`Model Conversion Tutorials <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>` page for a set of tutorials providing step-by-step instructions for converting specific ONNX models. Here are some examples:

* :doc:`Convert ONNX Faster R-CNN Model <openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_Faster_RCNN>`
* :doc:`Convert ONNX GPT-2 Model <openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_GPT2>`
* :doc:`Convert ONNX Mask R-CNN Model <openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_Mask_RCNN>`

@endsphinxdirective

