Converting a Keras Model
========================


.. meta::
   :description: Learn how to convert a model from the
                 Keras format to the OpenVINO Model.


This document explains the process of converting Keras 3 models to the OpenVINO Intermediate Representation (IR) format.
For instructions on converting Keras 2 models, refer to :doc:`TensorFlow Model Conversion <convert-model-tensorflow>`.

To convert a Keras 3 model, first export it to a lightweight TensorFlow SavedModel artifact,
and then convert it to an OpenVINO model, using the ``convert_model`` function.
Here is a code example of how to do this:

.. code-block:: py
   :force:

   import keras_hub
   import openvino as ov

   model = keras_hub.models.BertTextClassifier.from_preset(
       "bert_base_en_uncased",
       num_classes=4,
       preprocessor=None,
   )

   # export to SavedModel
   model.export("bert_base")

   # convert to OpenVINO model
   ov_model = ov.convert_model("bert_base")


.. note::

   The resulting OpenVINO IR model can be saved to drive with no additional, Keras-specific steps.
   Use the standard ``ov.save_model(ov_model,'model.xml')`` command. 

Alternatively, a model exported to TensorFlow SavedModel format can also be converted to OpenVINO IR using the ``ovc`` tool. Here is an example:

.. code-block:: sh
   :force:

   ovc bert_base


Run inference in Keras 3 with the OpenVINO backend
##################################################

Starting with release 3.8, Keras provides native integration with the OpenVINO backend for accelerated inference.
This integration enables you to leverage OpenVINO performance optimizations directly within the Keras workflow, enabling faster inference on OpenVINO supported hardware.

To switch to the OpenVINO backend in Keras 3, set the ``KERAS_BACKEND`` environment variable to ``"openvino"``
or specify the backend in the local configuration file at ``~/.keras/keras.json``.
Here is an example of how to infer a model (trained with PyTorch, JAX, or TensorFlow backends) in Keras 3, using the OpenVINO backend:

.. code-block:: py
   :force:

   import os

   os.environ["KERAS_BACKEND"] = "openvino"
   import numpy as np
   import keras
   import keras_hub

   features = {
       "token_ids": np.ones(shape=(2, 12), dtype="int32"),
       "segment_ids": np.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0]] * 2),
       "padding_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * 2),
   }

   # take a model from KerasHub
   bert = keras_hub.models.BertTextClassifier.from_preset(
       "bert_base_en_uncased",
       num_classes=4,
       preprocessor=None,
   )

   predictions = bert.predict(features)

.. note::

   The OpenVINO backend may currently lack support for some operations.
   This will be addressed in upcoming Keras releases as operation coverage is being expanded.
