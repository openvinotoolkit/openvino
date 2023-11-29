.. {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow_Lite}

Converting a TensorFlow Lite Model
==================================


.. meta::
   :description: Learn how to convert a model from a 
                 TensorFlow Lite format to the OpenVINO Intermediate Representation.


To convert a TensorFlow Lite model, use the ``mo`` script and specify the path to the input ``.tflite`` model file:

.. code-block:: sh

   mo --input_model <INPUT_MODEL>.tflite

TensorFlow Lite models are supported via FrontEnd API. You may skip conversion to IR and read models directly by OpenVINO runtime API. Refer to the :doc:`inference example <openvino_docs_OV_UG_Integrate_OV_with_your_application>` for more details. Using ``convert_model`` is still necessary in more complex cases, such as new custom inputs/outputs in model pruning, adding pre-processing, or using Python conversion extensions.

.. important::

   The ``convert_model()`` method returns ``ov.Model`` that you can optimize, compile, or save to a file for subsequent use.

Supported TensorFlow Lite Layers
###################################

For the list of supported standard layers, refer to the :doc:`Supported Operations <openvino_resources_supported_operations_frontend>` page.

Supported TensorFlow Lite Models
###################################

More than eighty percent of public TensorFlow Lite models are supported from open sources `TensorFlow Hub <https://tfhub.dev/s?deployment-format=lite&subtype=module,placeholder>`__ and `MediaPipe <https://developers.google.com/mediapipe>`__.
Unsupported models usually have custom TensorFlow Lite operations.

