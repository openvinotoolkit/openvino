Converting a TensorFlow Model
=============================


.. meta::
   :description: Learn how to convert a model from a TensorFlow format to the OpenVINO Model.


This page provides general instructions on how to run model conversion from a TensorFlow
format to the OpenVINO IR format. The instructions are different depending on whether
your model was created with TensorFlow v1.X or TensorFlow v2.X.

TensorFlow models can be obtained from
`Kaggle <https://www.kaggle.com/models?framework=tfLite&subtype=module,placeholder&tfhub-redirect=true>`__
or `Hugging Face <https://huggingface.co/models>`__.

.. note::

   TensorFlow models can be loaded by ``openvino.Core.read_model`` or
   ``openvino.Core.compile_model`` methods by OpenVINO runtime API without preparing
   OpenVINO IR first. Refer to the
   :doc:`inference example <../running-inference/integrate-openvino-with-your-application>`
   for more details. Using ``openvino.convert_model`` is still recommended if model load
   latency matters for the inference application.

.. note::

   ``openvino.convert_model`` uses sharing of model weights by default. That means that
   OpenVINO model will share the same areas in program memory where the original weights
   are located, for this reason the original model cannot be modified (Python object
   cannot be deallocated and original model file cannot be deleted) for the whole
   lifetime of OpenVINO model. Model inference for TensorFlow models can lead to model
   modification, so original TF model should not be inferred during the lifetime of
   OpenVINO model. If it is not desired, set ``share_weights=False`` when calling
   ``openvino.convert_model``.

.. note::

   The examples converting TensorFlow models from a file do not require any version
   of TensorFlow installed on the system, unless the ``tensorflow`` module is imported
   explicitly.

Converting TensorFlow 2 Models
##############################

TensorFlow 2.X officially supports two model formats: SavedModel and Keras H5 (or HDF5).
Below are the instructions on how to convert each of them.

SavedModel Format
+++++++++++++++++

A model in the SavedModel format consists of a directory with a ``saved_model.pb``
file and two subfolders: ``variables`` and ``assets`` inside.
To convert a model, run conversion with the directory as the model argument:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py
         :force:

         import openvino as ov
         ov_model = ov.convert_model('path_to_saved_model_dir')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc path_to_saved_model_dir

Keras H5 Format
+++++++++++++++

If you have a model in HDF5 format, load the model using TensorFlow 2 and serialize it to
SavedModel format. Here is an example of how to do it:

.. code-block:: py
   :force:

   import tensorflow as tf
   model = tf.keras.models.load_model('model.h5')
   tf.saved_model.save(model,'model')

Converting a Keras H5 model with a custom layer to the SavedModel format requires special
considerations. For example, the model with a custom layer ``CustomLayer`` from
``custom_layer.py`` is converted as follows:

.. code-block:: py
   :force:

   import tensorflow as tf
   from custom_layer import CustomLayer
   model = tf.keras.models.load_model('model.h5', custom_objects={'CustomLayer': CustomLayer})
   tf.saved_model.save(model,'model')

Then follow the above instructions for the SavedModel format.

.. note::

   Avoid using any workarounds or hacks to resave TensorFlow 2 models into TensorFlow 1 formats.

Converting TensorFlow 1 Models
###############################

Converting Frozen Model Format
+++++++++++++++++++++++++++++++

To convert a TensorFlow model, run model conversion with the path to the input
model ``*.pb*`` file:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py

         import openvino as ov
         ov_model = ov.convert_model('your_model_file.pb')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc your_model_file.pb


Converting Non-Frozen Model Formats
+++++++++++++++++++++++++++++++++++

There are three ways to store non-frozen TensorFlow models.

1. **SavedModel format**. In this case, a model consists of a special directory with a
   ``.pb`` file and several subfolders: ``variables``, ``assets``, and ``assets.extra``.
   For more information about the SavedModel directory, refer to the
   `README <https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/saved_model#components>`__
   file in the TensorFlow repository. To convert such TensorFlow model, run the conversion
   similarly to other model formats and pass a path to the directory as a model argument:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py

         import openvino as ov
         ov_model = ov.convert_model('path_to_saved_model_dir')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc path_to_saved_model_dir

2. **Checkpoint**. In this case, a model consists of two files: ``inference_graph.pb``
   (or ``inference_graph.pbtxt``) and ``checkpoint_file.ckpt``.
   If you do not have an inference graph file, refer to the
   `Freezing Custom Models in Python <#freezing-custom-models-in-python>`__ section.
   To convert the model with the inference graph in ``.pb`` format, provide paths to both
   files as an argument for ``ovc`` or ``openvino.convert_model``:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py

         import openvino as ov
         ov_model = ov.convert_model(['path_to_inference_graph.pb', 'path_to_checkpoint_file.ckpt'])

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc path_to_inference_graph.pb path_to_checkpoint_file.ckpt

To convert the model with the inference graph in the ``.pbtxt`` format, specify the path
to ``.pbtxt`` file instead of the ``.pb`` file. The conversion API automatically detects
the format of the provided file, there is no need to specify the model file format
explicitly when calling ``ovc`` or ``openvino.convert_model`` in all examples in this document.

3. **MetaGraph**. In this case, a model consists of three or four files stored in the same
   directory: ``model_name.meta``, ``model_name.index``,
   ``model_name.data-00000-of-00001`` (the numbers may vary), and ``checkpoint`` (optional).
   To convert such a TensorFlow model, run the conversion providing a path to ``.meta``
   file as an argument:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. code-block:: py

         import openvino as ov
         ov_model = ov.convert_model('path_to_meta_graph.meta')

   .. tab-item:: CLI
      :sync: cli

      .. code-block:: sh

         ovc path_to_meta_graph.meta


Freezing Custom Models in Python
++++++++++++++++++++++++++++++++

When a model is defined in Python code, you must create an inference graph file. Graphs are
usually built in a form that allows model training. That means all trainable parameters are
represented as variables in the graph. To be able to use such a graph with the model
conversion API, it should be frozen first before passing to the
``openvino.convert_model`` function:

.. code-block:: py
   :force:

   import tensorflow as tf
   from tensorflow.python.framework import graph_io
   frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["name_of_the_output_node"])

   import openvino as ov
   ov_model = ov.convert_model(frozen)

Where:

* ``sess`` is the instance of the TensorFlow Session object where the network topology
  is defined.
* ``["name_of_the_output_node"]`` is the list of output node names in the graph;
  ``frozen`` graph will include only those nodes from the original ``sess.graph_def``
  that are directly or indirectly used to compute given output nodes. The
  ``'name_of_the_output_node'`` is an example of a possible output node name.
  You should derive the names based on your own graph.

Converting TensorFlow Models from Memory Using Python API
############################################################

Model conversion API supports passing TensorFlow/TensorFlow2 models directly from memory.

* ``Trackable``. The object returned by ``hub.load()`` can be converted to
  ``ov.Model`` with ``convert_model()``.

  .. code-block:: py
     :force:

     import tensorflow_hub as hub
     import openvino as ov

     model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
     ov_model = ov.convert_model(model)

* ``tf.function``

  .. code-block:: py
     :force:

     @tf.function(
        input_signature=[tf.TensorSpec(shape=[1, 2, 3], dtype=tf.float32),
                         tf.TensorSpec(shape=[1, 2, 3], dtype=tf.float32)])
     def func(x, y):
        return tf.nn.sigmoid(tf.nn.relu(x + y))

     import openvino as ov
     ov_model = ov.convert_model(func)

* ``tf.keras.Model``

  .. code-block:: py
     :force:

     import openvino as ov
     model = tf.keras.applications.ResNet50(weights="imagenet")
     ov_model = ov.convert_model(model)

* ``tf.keras.layers.Layer``. The ``ov.Model`` converted from ``tf.keras.layers.Layer``
  does not contain original input and output names. So it is recommended to convert the
  model to ``tf.keras.Model`` before conversion or use ``hub.load()`` for
  TensorFlow Hub models.

  .. code-block:: py
     :force:

     import tensorflow_hub as hub
     import openvino as ov

     model = hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/5")
     ov_model = ov.convert_model(model)


* ``tf.Module``. Requires setting shapes in ``input`` parameter.

  .. code-block:: py
     :force:

     import tensorflow as tf
     import openvino as ov

     class MyModule(tf.Module):
        def __init__(self, name=None):
           super().__init__(name=name)
           self.constant1 = tf.constant(5.0, name="var1")
           self.constant2 = tf.constant(1.0, name="var2")
        def __call__(self, x):
           return self.constant1 * x + self.constant2

     model = MyModule(name="simple_module")
     ov_model = ov.convert_model(model, input=[-1])

.. note::

   There is a known bug in ``openvino.convert_model`` on using ``tf.Variable`` nodes in
   the model graph. The results of the conversion of such models are unpredictable. It
   is recommended to save a model with ``tf.Variable`` into TensorFlow Saved Model format
   and load it with ``openvino.convert_model``.

* ``tf.compat.v1.Graph``

  .. code-block:: py
     :force:

     with tf.compat.v1.Session() as sess:
        inp1 = tf.compat.v1.placeholder(tf.float32, [100], 'Input1')
        inp2 = tf.compat.v1.placeholder(tf.float32, [100], 'Input2')
        output = tf.nn.relu(inp1 + inp2, name='Relu')
        tf.compat.v1.global_variables_initializer()
        model = sess.graph

     import openvino as ov
     ov_model = ov.convert_model(model)

* ``tf.compat.v1.GraphDef``

  .. code-block:: py
     :force:

     with tf.compat.v1.Session() as sess:
        inp1 = tf.compat.v1.placeholder(tf.float32, [100], 'Input1')
        inp2 = tf.compat.v1.placeholder(tf.float32, [100], 'Input2')
        output = tf.nn.relu(inp1 + inp2, name='Relu')
        tf.compat.v1.global_variables_initializer()
        model = sess.graph_def

     import openvino as ov
     ov_model = ov.convert_model(model)

* ``tf.compat.v1.session``

  .. code-block:: py
     :force:

     with tf.compat.v1.Session() as sess:
        inp1 = tf.compat.v1.placeholder(tf.float32, [100], 'Input1')
        inp2 = tf.compat.v1.placeholder(tf.float32, [100], 'Input2')
        output = tf.nn.relu(inp1 + inp2, name='Relu')
        tf.compat.v1.global_variables_initializer()

        import openvino as ov
        ov_model = ov.convert_model(sess)

* ``tf.train.checkpoint``

  .. code-block:: py
     :force:

     model = tf.keras.Model(...)
     checkpoint = tf.train.Checkpoint(model)
     save_path = checkpoint.save(save_directory)
     # ...
     checkpoint.restore(save_path)

     import openvino as ov
     ov_model = ov.convert_model(checkpoint)

Supported TensorFlow and TensorFlow 2 Keras Layers
##################################################

For the list of supported standard layers, refer to the
:doc:`Supported Operations <../../about-openvino/compatibility-and-support/supported-operations>`
page.


