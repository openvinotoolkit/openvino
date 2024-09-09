Converting a TensorFlow Neural Collaborative Filtering Model
============================================================


.. meta::
   :description: Learn how to convert a Neural Collaborative
                 Filtering Model from TensorFlow to the OpenVINO Intermediate
                 Representation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

This tutorial explains how to convert Neural Collaborative Filtering (NCF) model to the OpenVINO Intermediate Representation.

`Public TensorFlow NCF model <https://github.com/tensorflow/models/tree/master/official/recommendation>`__ does not contain pre-trained weights. To convert this model to the IR:

1. Use `the instructions <https://github.com/tensorflow/models/tree/master/official/recommendation#train-and-evaluate-model>`__ from this repository to train the model.

2. Freeze the inference graph you get in the previous step in ``model_dir``, following the instructions from the **Freezing Custom Models in Python** section of the :doc:`Converting a TensorFlow Model <../[legacy]-convert-tensorflow>` guide.

   Run the following commands:

   .. code-block:: py
      :force:

       import tensorflow as tf
       from tensorflow.python.framework import graph_io

       sess = tf.compat.v1.Session()
       saver = tf.compat.v1.train.import_meta_graph("/path/to/model/model.meta")
       saver.restore(sess, tf.train.latest_checkpoint('/path/to/model/'))

       frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, \
                                                           ["rating/BiasAdd"])
       graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)

   where ``rating/BiasAdd`` is an output node.

3. Convert the model to the OpenVINO format. If you look at your frozen model, you can see that it has one input that is split into four ``ResourceGather`` layers. (Click image to zoom in.)

   .. image::  ../../../../../../assets/images/NCF_start.svg

   However, as the model conversion API does not support such data feeding, you should skip it. Cut
   the edges incoming in ``ResourceGather`` port 1:

   .. code-block:: sh

       mo --input_model inference_graph.pb                    \
       --input 1:embedding/embedding_lookup,1:embedding_1/embedding_lookup, \
       1:embedding_2/embedding_lookup,1:embedding_3/embedding_lookup        \
       --input_shape [256],[256],[256],[256]                                \
       --output_dir <OUTPUT_MODEL_DIR>

   In the ``input_shape`` parameter, 256 specifies the ``batch_size`` for your model.

Alternatively, you can do steps 2 and 3 in one command line:

.. code-block:: sh

    mo --input_meta_graph /path/to/model/model.meta        \
    --input 1:embedding/embedding_lookup,1:embedding_1/embedding_lookup, \
    1:embedding_2/embedding_lookup,1:embedding_3/embedding_lookup        \
    --input_shape [256],[256],[256],[256] --output rating/BiasAdd        \
    --output_dir <OUTPUT_MODEL_DIR>

