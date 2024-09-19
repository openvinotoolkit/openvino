Converting TensorFlow EfficientDet Models
=========================================


.. meta::
   :description: Learn how to convert an EfficientDet model
                 from TensorFlow to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

This tutorial explains how to convert EfficientDet public object detection models to the Intermediate Representation (IR).

.. _efficientdet-to-ir:

Converting EfficientDet Model to the IR
#######################################

There are several public versions of EfficientDet model implementation available on GitHub. This tutorial explains how to
convert models from the `repository <https://github.com/google/automl/tree/master/efficientdet>`__  (commit 96e1fee) to the OpenVINO format.

Download and extract the model checkpoint `efficientdet-d4.tar.gz <https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz>`__
referenced in the **"Pretrained EfficientDet Checkpoints"** section of the model repository:

.. code-block:: sh

   wget https://storage.googleapis.com/cloud-tpu-checkpoints/efficientdet/coco2/efficientdet-d4.tar.gz
   tar zxvf efficientdet-d4.tar.gz

Converting an EfficientDet TensorFlow Model to the IR
+++++++++++++++++++++++++++++++++++++++++++++++++++++

To generate the IR of the EfficientDet TensorFlow model, run:

.. code-block:: sh

   mo \
   --input_meta_graph efficientdet-d4/model.meta \
   --input_shape [1,$IMAGE_SIZE,$IMAGE_SIZE,3] \
   --reverse_input_channels


Where ``$IMAGE_SIZE`` is the size that the input image of the original TensorFlow model will be resized to. Different
EfficientDet models were trained with different input image sizes. To determine the right one, refer to the ``efficientdet_model_param_dict``
dictionary in the `hparams_config.py <https://github.com/google/automl/blob/96e1fee/efficientdet/hparams_config.py#L304>`__ file.
The attribute ``image_size`` specifies the shape to be defined for the model conversion.

.. note::

    The color channel order (RGB or BGR) of an input data should match the channel order of the model training dataset. If they are different, perform the ``RGB<->BGR`` conversion specifying the command-line parameter: ``--reverse_input_channels``. Otherwise, inference results may be incorrect. For more information about the parameter, refer to the **When to Reverse Input Channels** section of the :doc:`Converting a Model to Intermediate Representation (IR) <../../[legacy]-setting-input-shapes>` guide.

OpenVINO toolkit provides samples that can be used to infer EfficientDet model.
For more information, refer to the `Open Model Zoo Demos <https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/README.md>`__.

.. important::

   Due to the deprecation of Open Model Zoo, models in the OpenVINO IR format have are now
   published on `Hugging Face <https://huggingface.co/OpenVINO>`__.


Interpreting Results of the TensorFlow Model and the IR
#######################################################

The TensorFlow model produces as output a list of 7-element tuples: ``[image_id, y_min, x_min, y_max, x_max, confidence, class_id]``, where:

* ``image_id`` -- image batch index.
* ``y_min`` -- absolute ``y`` coordinate of the lower left corner of the detected object.
* ``x_min`` -- absolute ``x`` coordinate of the lower left corner of the detected object.
* ``y_max`` -- absolute ``y`` coordinate of the upper right corner of the detected object.
* ``x_max`` -- absolute ``x`` coordinate of the upper right corner of the detected object.
* ``confidence`` -- the confidence of the detected object.
* ``class_id`` -- the id of the detected object class counted from 1.

The output of the IR is a list of 7-element tuples: ``[image_id, class_id, confidence, x_min, y_min, x_max, y_max]``, where:

* ``image_id`` -- image batch index.
* ``class_id`` -- the id of the detected object class counted from 0.
* ``confidence`` -- the confidence of the detected object.
* ``x_min`` -- normalized ``x`` coordinate of the lower left corner of the detected object.
* ``y_min`` -- normalized ``y`` coordinate of the lower left corner of the detected object.
* ``x_max`` -- normalized ``x`` coordinate of the upper right corner of the detected object.
* ``y_max`` -- normalized ``y`` coordinate of the upper right corner of the detected object.

The first element with ``image_id = -1`` means end of data.


