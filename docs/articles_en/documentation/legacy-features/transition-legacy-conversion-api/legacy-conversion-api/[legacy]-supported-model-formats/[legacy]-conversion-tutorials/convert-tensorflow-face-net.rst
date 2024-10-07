Converting TensorFlow FaceNet Models
====================================


.. meta::
   :description: Learn how to convert a FaceNet model
                 from TensorFlow to the OpenVINO Intermediate Representation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Supported Model Formats <../../../../../../openvino-workflow/model-preparation>` article.

`Public pre-trained FaceNet models <https://github.com/davidsandberg/facenet#pre-trained-models>`__ contain both training
and inference part of graph. Switch between this two states is manageable with placeholder value.
Intermediate Representation (IR) models are intended for inference, which means that train part is redundant.

There are two inputs in this network: boolean ``phase_train`` which manages state of the graph (train/infer) and
``batch_size`` which is a part of batch joining pattern.

.. image:: ../../../../../../assets/images/FaceNet.svg

Converting a TensorFlow FaceNet Model to the IR
###############################################

To generate a FaceNet OpenVINO model, feed a TensorFlow FaceNet model to model conversion API with the following parameters:

.. code-block:: sh

    mo
   --input_model path_to_model/model_name.pb       \
   --freeze_placeholder_with_value "phase_train->False"


The batch joining pattern transforms to a placeholder with the model default shape if ``--input_shape`` or ``--batch``/``-b`` are not provided. Otherwise, the placeholder shape has custom parameters.

* ``freeze_placeholder_with_value "phase_train->False"`` to switch graph to inference mode
* ``batch`*/*`-b`` is applicable to override original network batch
* ``input_shape`` is applicable with or without ``input``
* other options are applicable

