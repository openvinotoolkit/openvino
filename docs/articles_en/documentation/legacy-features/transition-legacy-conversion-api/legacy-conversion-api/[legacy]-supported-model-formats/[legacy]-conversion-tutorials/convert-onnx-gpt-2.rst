Converting an ONNX GPT-2 Model
==============================


.. meta::
   :description: Learn how to convert a pre-trained GPT-2
                 model from ONNX to the OpenVINO Intermediate Representation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

`Public pre-trained GPT-2 model <https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2>`__ is a large
transformer-based language model with a simple objective: predict the next word, given all of the previous words within some text.

Downloading the Pre-Trained Base GPT-2 Model
############################################

To download the model, go to `this model <https://github.com/onnx/models/blob/main/validated/text/machine_comprehension/gpt-2/model/gpt2-10.onnx>`__, and press **Download**.

To download the model and sample test data, go to `this model <https://github.com/onnx/models/blob/main/validated/text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz>`__, and press **Download**.

Converting an ONNX GPT-2 Model to IR
####################################

Generate the Intermediate Representation of the model GPT-2 by running model conversion with the following parameters:

.. code-block:: sh

    mo --input_model gpt2-10.onnx --input_shape [X,Y,Z] --output_dir <OUTPUT_MODEL_DIR>


