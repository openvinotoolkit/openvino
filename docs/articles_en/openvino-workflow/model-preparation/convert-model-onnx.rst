Converting an ONNX Model
========================


.. meta::
   :description: Learn how to convert a model from the
                 ONNX format to the OpenVINO Model.

Introduction to ONNX
####################

`ONNX <https://github.com/onnx/onnx>`__ is a representation format for deep learning models
that enables AI developers to easily transfer models between different frameworks.

.. note::

   An ONNX model file can be loaded by ``openvino.Core.read_model`` or
   ``openvino.Core.compile_model`` methods by OpenVINO runtime API without the need to
   prepare an OpenVINO IR first. Refer to the
   :doc:`inference example <../running-inference>`
   for more details. Using ``openvino.convert_model`` is still recommended if the model
   load latency is important for the inference application.

Converting an ONNX Model
########################

This page provides instructions on model conversion from the ONNX format to the
OpenVINO IR format.

For model conversion, you need an ONNX model either directly downloaded from
an online database, for example `Hugging Face <https://huggingface.co/models>`__ , or
converted from any framework that supports exporting to the ONNX format.

To convert an ONNX model, run model conversion with the path to the input
model ``.onnx`` file:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py

         import openvino as ov
         ov.convert_model('your_model_file.onnx')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc your_model_file.onnx

External Data Files
###################

ONNX models may consist of multiple files when the model size exceeds 2GB allowed by Protobuf. According to this `ONNX article <https://github.com/onnx/onnx/blob/main/docs/ExternalData.md>`__, instead of a single file, the model is represented as one file with ``.onnx`` extension and multiple separate files with external data. These data files are located in the same directory as the main ``.onnx`` file or in another directory.

OpenVINO model conversion API supports ONNX models with external data representation. In this case, you only need to pass the main file with ``.onnx`` extension as ``ovc`` or ``openvino.convert_model`` parameter. The other files will be found and loaded automatically during the model conversion. The resulting OpenVINO model, represented as an IR in the filesystem, will have the usual structure with a single ``.xml`` file and a single ``.bin`` file, where all the original model weights are copied and packed together.

Supported ONNX Layers
#####################

For the list of supported standard layers, refer to the
:doc:`Supported Operations <../../documentation/compatibility-and-support/supported-operations>`
page.

Additional Resources
####################

Check out more examples of model conversion in
:doc:`interactive Python tutorials <../../get-started/learn-openvino/interactive-tutorials-python>`.

