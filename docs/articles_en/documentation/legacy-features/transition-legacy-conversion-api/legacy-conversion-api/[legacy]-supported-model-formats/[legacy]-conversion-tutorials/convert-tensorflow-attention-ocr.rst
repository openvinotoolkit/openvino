Converting a TensorFlow Attention OCR Model
===========================================


.. meta::
   :description: Learn how to convert the Attention OCR
                 model from the TensorFlow Attention OCR repository to the
                 OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

This tutorial explains how to convert the Attention OCR (AOCR) model from the `TensorFlow Attention OCR repository <https://github.com/emedvedev/attention-ocr>`__ to the Intermediate Representation (IR).

Extracting a Model from ``aocr`` Library
########################################

To get an AOCR model, download ``aocr`` Python library:

.. code-block:: sh

   pip install git+https://github.com/emedvedev/attention-ocr.git@master#egg=aocr

This library contains a pretrained model and allows training and running AOCR, using the command line. After installation of `aocr`, extract the model:

.. code-block:: sh

   aocr export --format=frozengraph model/path/

Once extracted, the model can be found in ``model/path/`` folder.

Converting the TensorFlow AOCR Model to IR
##########################################

The original AOCR model includes the preprocessing data, which contains:

* Decoding input data to binary format where input data is an image represented as a string.
* Resizing binary image to working resolution.

The resized image is sent to the convolution neural network (CNN). Because model conversion API does not support image decoding, the preprocessing part of the model should be cut off, using the ``input`` command-line parameter.

.. code-block:: sh

   mo \
   --input_model=model/path/frozen_graph.pb \
   --input="map/TensorArrayStack/TensorArrayGatherV3:0[1,32,86,1]" \
   --output "transpose_1,transpose_2" \
   --output_dir path/to/ir/


Where:

* ``map/TensorArrayStack/TensorArrayGatherV3:0[1 32 86 1]`` - name of node producing tensor after preprocessing.
* ``transpose_1`` - name of the node producing tensor with predicted characters.
* ``transpose_2`` - name of the node producing tensor with predicted characters probabilities.

