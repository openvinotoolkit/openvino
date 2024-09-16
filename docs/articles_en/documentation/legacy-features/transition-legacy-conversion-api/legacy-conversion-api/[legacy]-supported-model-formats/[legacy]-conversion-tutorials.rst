[LEGACY] Model Conversion Tutorials
====================================================


.. toctree::
   :maxdepth: 1
   :hidden:

   [legacy]-conversion-tutorials/convert-tensorflow-attention-ocr
   [legacy]-conversion-tutorials/convert-tensorflow-bert
   [legacy]-conversion-tutorials/convert-tensorflow-crnn
   [legacy]-conversion-tutorials/convert-tensorflow-deep-speech
   [legacy]-conversion-tutorials/convert-tensorflow-efficient-det
   [legacy]-conversion-tutorials/convert-tensorflow-face-net
   [legacy]-conversion-tutorials/convert-tensorflow-gnmt
   [legacy]-conversion-tutorials/convert-tensorflow-language-1b
   [legacy]-conversion-tutorials/convert-tensorflow-ncf
   [legacy]-conversion-tutorials/convert-tensorflow-object-detection
   [legacy]-conversion-tutorials/convert-tensorflow-retina-net
   [legacy]-conversion-tutorials/convert-tensorflow-slim-library
   [legacy]-conversion-tutorials/convert-tensorflow-wide-and-deep-family
   [legacy]-conversion-tutorials/convert-tensorflow-xlnet
   [legacy]-conversion-tutorials/convert-tensorflow-yolo
   [legacy]-conversion-tutorials/convert-onnx-faster-r-cnn
   [legacy]-conversion-tutorials/convert-onnx-gpt-2
   [legacy]-conversion-tutorials/convert-onnx-mask-r-cnn
   [legacy]-conversion-tutorials/convert-pytorch-bert-ner
   [legacy]-conversion-tutorials/convert-pytorch-cascade-rcnn-r-101
   [legacy]-conversion-tutorials/convert-pytorch-f3-net
   [legacy]-conversion-tutorials/convert-pytorch-quartz-net
   [legacy]-conversion-tutorials/convert-pytorch-rcan
   [legacy]-conversion-tutorials/convert-pytorch-rnn-t
   [legacy]-conversion-tutorials/convert-pytorch-yolact


.. meta::
   :description: Get to know conversion methods for specific TensorFlow, ONNX, and PyTorch models.


.. danger::

   The code described in the tutorials has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../learn-openvino/interactive-tutorials-python>`.

This section provides a set of tutorials that demonstrate conversion methods for specific
TensorFlow, ONNX, and PyTorch models. Note that these instructions do not cover all use
cases and may not reflect your particular needs.
Before studying the tutorials, try to convert the model out-of-the-box by specifying only the
``--input_model`` parameter in the command line.

.. note::

   Apache MXNet, Caffe, and Kaldi are no longer directly supported by OpenVINO.

You will find a collection of :doc:`Python tutorials <../../../../../learn-openvino/interactive-tutorials-python>` written for running on Jupyter notebooks
that provide an introduction to the OpenVINO™ toolkit and explain how to use the Python API and tools for
optimized deep learning inference.

