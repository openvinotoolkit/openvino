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

Export a Model from Any Backend to OpenVINO IR
##############################################

Keras also supports exporting models directly to disk from any backend (TensorFlow, JAX, PyTorch, or OpenVINO) itself directly into the OpenVINO IR format using the ``model.export()`` API.

Here is a minimal example:

.. code-block:: python
   :force:

   import os
   os.environ["KERAS_BACKEND"] = "tensorflow"
   import keras_hub
   import numpy as np
   import openvino as ov

   temp_filepath = "temp_exported_model.xml"

   model = keras_hub.models.BertTextClassifier.from_preset(
      "bert_base_en_uncased",
      num_classes=4,
      preprocessor=None,
   )

   model.export(temp_filepath, format="openvino")

   core = ov.Core()
   ov_model = core.read_model(temp_filepath)
   compiled_model = core.compile_model(ov_model, "CPU")
   inputs = {
      "token_ids:0": np.ones((1, 12), dtype="int32"),
      "segment_ids:0": np.zeros((1, 12), dtype="int32"),
      "padding_mask:0": np.ones((1, 12), dtype="int32"),
   }
   ov_output = compiled_model(inputs)[0]
   print(ov_output)

.. note::

   OpenVINO model input names may include a ``:0`` suffix (for example, ``"token_ids:0"``). 
   Be sure to use the exact input names when running inference.

   To check the input names, use the ``get_any_name()`` method:
   ``[input.get_any_name() for input in ov_model.inputs]``

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
