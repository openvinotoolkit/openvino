# Converting a TensorFlow Lite Model {#openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite}

@sphinxdirective

.. meta::
   :description: Learn how to convert a model from a
                 TensorFlow Lite format to the OpenVINO Model.


To convert an TensorFlow Lite model, run model conversion with the path to the ``.tflite`` model file:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py

         import openvino as ov
         ov.convert_model('your_model_file.tflite')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc your_model_file.tflite

.. note:: TensorFlow Lite model file can be loaded by ``openvino.Core.read_model`` or ``openvino.Core.compile_model`` methods by OpenVINO runtime API without preparing OpenVINO IR first. Refer to the :doc:`inference example <openvino_docs_OV_UG_Integrate_OV_with_your_application>` for more details. Using ``openvino.convert_model`` is still recommended if model load latency matters for the inference application.

Supported TensorFlow Lite Layers
###################################

For the list of supported standard layers, refer to the :doc:`Supported Operations <openvino_resources_supported_operations_frontend>` page.

Supported TensorFlow Lite Models
###################################

More than eighty percent of public TensorFlow Lite models are supported from open sources `TensorFlow Hub <https://tfhub.dev/s?deployment-format=lite&subtype=module,placeholder>`__ and `MediaPipe <https://developers.google.com/mediapipe>`__.
Unsupported models usually have custom TensorFlow Lite operations.

@endsphinxdirective
