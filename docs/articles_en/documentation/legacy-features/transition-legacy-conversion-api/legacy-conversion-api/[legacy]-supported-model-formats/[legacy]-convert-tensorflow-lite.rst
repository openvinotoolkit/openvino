[LEGACY] Converting a TensorFlow Lite Model
=====================================================


.. meta::
   :description: Learn how to convert a model from a
                 TensorFlow Lite format to the OpenVINO Intermediate Representation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Converting a TensorFlow Lite Model <../../../../../openvino-workflow/model-preparation/convert-model-tensorflow-lite>` article.

To convert a TensorFlow Lite model, use the ``mo`` script and specify the path to the input ``.tflite`` model file:

.. code-block:: sh

   mo --input_model <INPUT_MODEL>.tflite

TensorFlow Lite models are supported via FrontEnd API. You may skip conversion to IR and read models directly by OpenVINO runtime API. Refer to the :doc:`inference example <../../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>` for more details. Using ``convert_model`` is still necessary in more complex cases, such as new custom inputs/outputs in model pruning, adding pre-processing, or using Python conversion extensions.

.. important::

   The ``convert_model()`` method returns ``ov.Model`` that you can optimize, compile, or save to a file for subsequent use.

Supported TensorFlow Lite Layers
###################################

For the list of supported standard layers, refer to the :doc:`Supported Operations <../../../../../about-openvino/compatibility-and-support/supported-operations>` page.

Supported TensorFlow Lite Models
###################################

More than eighty percent of public TensorFlow Lite models are supported from open sources `TensorFlow Hub <https://tfhub.dev/s?deployment-format=lite&subtype=module,placeholder>`__ and `MediaPipe <https://developers.google.com/mediapipe>`__.
Unsupported models usually have custom TensorFlow Lite operations.

