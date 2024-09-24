[LEGACY] Converting an ONNX Model
=============================================

.. meta::
   :description: Learn how to convert a model from the
                 ONNX format to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Converting an ONNX Model <../../../../../openvino-workflow/model-preparation/convert-model-onnx>` article.


.. note:: ONNX models are supported via FrontEnd API. You may skip conversion to IR and read models directly by OpenVINO runtime API. Refer to the :doc:`inference example <../../../../../openvino-workflow/running-inference/integrate-openvino-with-your-application>` for more details. Using ``convert_model`` is still necessary in more complex cases, such as new custom inputs/outputs in model pruning, adding pre-processing, or using Python conversion extensions.

Converting an ONNX Model
########################

The model conversion process assumes you have an ONNX model that was directly downloaded from a public repository or converted from any framework that supports exporting to the ONNX format.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      To convert an ONNX model, run ``convert_model()`` method with the path to the ``<INPUT_MODEL>.onnx`` file:

      .. code-block:: py
         :force:

         import openvino
         from openvino.tools.mo import convert_model

         core = openvino.Core()
         ov_model = convert_model("<INPUT_MODEL>.onnx")
         compiled_model = core.compile_model(ov_model, "AUTO")

      .. important::

         The ``convert_model()`` method returns ``ov.Model`` that you can optimize, compile, or save to a file for subsequent use.

   .. tab-item:: CLI
      :sync: cli

      You can use ``mo`` command-line tool to convert a model to IR. The obtained IR can then be read by ``read_model()`` and inferred.

      .. code-block:: sh

         mo --input_model <INPUT_MODEL>.onnx


There are no ONNX-specific parameters, so only framework-agnostic parameters are available to convert your model. For details, see the *General Conversion Parameters* section in the :doc:`Converting a Model to Intermediate Representation (IR) <../[legacy]-setting-input-shapes>` guide.

Supported ONNX Layers
#####################

For the list of supported standard layers, refer to the :doc:`Supported Operations <../../../../../about-openvino/compatibility-and-support/supported-operations>` page.

Additional Resources
####################

See the :doc:`Model Conversion Tutorials <[legacy]-conversion-tutorials>` page for a set of tutorials providing step-by-step instructions for converting specific ONNX models. Here are some examples:

* :doc:`Convert ONNX Faster R-CNN Model <[legacy]-conversion-tutorials/convert-onnx-faster-r-cnn>`
* :doc:`Convert ONNX GPT-2 Model <[legacy]-conversion-tutorials/convert-onnx-gpt-2>`
* :doc:`Convert ONNX Mask R-CNN Model <[legacy]-conversion-tutorials/convert-onnx-mask-r-cnn>`


