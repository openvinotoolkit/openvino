Converting a TensorFlow RetinaNet Model
=======================================


.. meta::
   :description: Learn how to convert a RetinaNet model
                 from TensorFlow to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python ../../../../../../learn-openvino/interactive-tutorials-python <../../../../../../learn-openvino/interactive-tutorials-python>`.

This tutorial explains how to convert a RetinaNet model to the Intermediate Representation (IR).

`Public RetinaNet model <https://github.com/fizyr/keras-retinanet>`__ does not contain pretrained TensorFlow weights.
To convert this model to the TensorFlow format, follow the `Reproduce Keras to TensorFlow Conversion tutorial <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/retinanet-tf/README.md>`__.

After converting the model to TensorFlow format, run the following command:

.. code-block:: sh

   mo --input "input_1[1,1333,1333,3]" --input_model retinanet_resnet50_coco_best_v2.1.0.pb --transformations_config front/tf/retinanet.json


Where ``transformations_config`` command-line parameter specifies the configuration json file containing model conversion hints for model conversion API.
The json file contains some parameters that need to be changed if you train the model yourself. It also contains information on how to match endpoints
to replace the subgraph nodes. After the model is converted to the OpenVINO IR format, the output nodes will be replaced with DetectionOutput layer.

