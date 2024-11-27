Converting a TensorFlow XLNet Model
===================================


.. meta::
   :description: Learn how to convert an XLNet model from
                 TensorFlow to the OpenVINO Intermediate Representation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

Pretrained models for XLNet (Bidirectional Encoder Representations from Transformers) are
`publicly available <https://github.com/zihangdai/xlnet>`__.

Supported Models
################

The following models from the pretrained `XLNet model list <https://github.com/zihangdai/xlnet#pre-trained-models>`__ are currently supported:

* `XLNet-Large, Cased <https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip>`__
* `XLNet-Base, Cased <https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip>`__

Downloading the Pretrained Base XLNet Model
###########################################

Download and unzip an archive with the `XLNet-Base, Cased <https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip>`__.

After the archive is unzipped, the directory ``cased_L-12_H-768_A-12`` is created and contains the following files:

* TensorFlow checkpoint (``xlnet_model.ckpt``), containing the pretrained weights (which is actually 3 files)
* sentence piece model (``spiece.model``) used for (de)tokenization
* config file (``xlnet_config.json``), which specifies the hyperparameters of the model

To get pb-file from the archive contents, you need to do the following.

1. Run commands

   .. code-block:: sh

      cd ~
      mkdir XLNet-Base
      cd XLNet-Base
      git clone https://github.com/zihangdai/xlnet
      wget https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip
      unzip cased_L-12_H-768_A-12.zip
      mkdir try_save


2. Save and run the following Python script in `~/XLNet-Base/xlnet`:

   .. note:: The original model repository has been tested with TensorFlow 1.13.1 under Python2.

   .. code-block:: py
      :force:

      from collections import namedtuple

      import tensorflow as tf
      from tensorflow.python.framework import graph_io

      import model_utils
      import xlnet

      LENGTHS = 50
      BATCH = 1
      OUTPUT_DIR = '~/XLNet-Base/try_save/'
      INIT_CKPT_PATH = '~/XLNet-Base/xlnet_cased_L-12_H-768_A-12/xlnet_model.ckpt'
      XLNET_CONFIG_PATH = '~/XLNet-Base/xlnet_cased_L-12_H-768_A-12/xlnet_config.json'

      FLags = namedtuple('FLags', 'use_tpu init_checkpoint')
      FLAGS = FLags(use_tpu=False, init_checkpoint=INIT_CKPT_PATH)

      xlnet_config = xlnet.XLNetConfig(json_path=XLNET_CONFIG_PATH)
      run_config = xlnet.RunConfig(is_training=False, use_tpu=False, use_bfloat16=False, dropout=0.1, dropatt=0.1,)


      sentence_features_input_idx = tf.compat.v1.placeholder(tf.int32, shape=[LENGTHS, BATCH], name='input_ids')
      sentence_features_segment_ids = tf.compat.v1.placeholder(tf.int32, shape=[LENGTHS, BATCH], name='seg_ids')
      sentence_features_input_mask = tf.compat.v1.placeholder(tf.float32, shape=[LENGTHS, BATCH], name='input_mask')

      with tf.compat.v1.Session() as sess:
         xlnet_model = xlnet.XLNetModel(xlnet_config=xlnet_config, run_config=run_config,
                                       input_ids=sentence_features_input_idx,
                                       seg_ids=sentence_features_segment_ids,
                                       input_mask=sentence_features_input_mask)

         sess.run(tf.compat.v1.global_variables_initializer())
         model_utils.init_from_checkpoint(FLAGS, True)

         # Save the variables to disk.
         saver = tf.compat.v1.train.Saver()

         # Saving checkpoint
         save_path = saver.save(sess, OUTPUT_DIR + "model.ckpt")

         # Freezing model
         outputs = ['model/transformer/dropout_2/Identity']
         graph_def_freezed = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)

         # Saving non-frozen and frozen  model to pb
         graph_io.write_graph(sess.graph.as_graph_def(), OUTPUT_DIR, 'model.pb', as_text=False)
         graph_io.write_graph(graph_def_freezed,OUTPUT_DIR, 'model_frozen.pb',
                              as_text=False)

         # Write to tensorboard
         with tf.compat.v1.summary.FileWriter(logdir=OUTPUT_DIR, graph_def=graph_def_freezed) as writer:
            writer.flush()

Downloading the Pretrained Large XLNet Model
############################################

Download and unzip an archive with the `XLNet-Base, Cased <https://storage.googleapis.com/xlnet/released_models/cased_L-12_H-768_A-12.zip>`__.

After unzipping the archive, the directory ``cased_L-12_H-1024_A-16`` is created and contains the following files:

* TensorFlow checkpoint (``xlnet_model.ckpt``) containing the pretrained weights (which is actually 3 files)
* sentence piece model (``spiece.model``) used for (de)tokenization
* config file (``xlnet_config.json``) which specifies the hyperparameters of the model

To get ``pb-file`` from the archive contents, follow the instructions below:

1. Run commands

   .. code-block:: sh

      cd ~
      mkdir XLNet-Large
      cd XLNet-Large
      git clone https://github.com/zihangdai/xlnet
      wget https://storage.googleapis.com/xlnet/released_models/cased_L-24_H-1024_A-16.zip
      unzip cased_L-24_H-1024_A-16.zip
      mkdir try_save


2. Save and run the following Python script in ``~/XLNet-Large/xlnet``:

   .. code-block:: py
      :force:

      from collections import namedtuple

      import tensorflow as tf
      from tensorflow.python.framework import graph_io

      import model_utils
      import xlnet

      LENGTHS = 50
      BATCH = 1
      OUTPUT_DIR = '~/XLNet-Large/try_save'
      INIT_CKPT_PATH = '~/XLNet-Large/cased_L-24_H-1024_A-16/xlnet_model.ckpt'
      XLNET_CONFIG_PATH = '~/XLNet-Large/cased_L-24_H-1024_A-16/xlnet_config.json'

      FLags = namedtuple('FLags', 'use_tpu init_checkpoint')
      FLAGS = FLags(use_tpu=False, init_checkpoint=INIT_CKPT_PATH)

      xlnet_config = xlnet.XLNetConfig(json_path=XLNET_CONFIG_PATH)
      run_config = xlnet.RunConfig(is_training=False, use_tpu=False, use_bfloat16=False, dropout=0.1, dropatt=0.1,)


      sentence_features_input_idx = tf.compat.v1.placeholder(tf.int32, shape=[LENGTHS, BATCH], name='input_ids')
      sentence_features_segment_ids = tf.compat.v1.placeholder(tf.int32, shape=[LENGTHS, BATCH], name='seg_ids')
      sentence_features_input_mask = tf.compat.v1.placeholder(tf.float32, shape=[LENGTHS, BATCH], name='input_mask')

      with tf.compat.v1.Session() as sess:
         xlnet_model = xlnet.XLNetModel(xlnet_config=xlnet_config, run_config=run_config,
                                       input_ids=sentence_features_input_idx,
                                       seg_ids=sentence_features_segment_ids,
                                       input_mask=sentence_features_input_mask)

         sess.run(tf.compat.v1.global_variables_initializer())
         model_utils.init_from_checkpoint(FLAGS, True)

         # Save the variables to disk.
         saver = tf.compat.v1.train.Saver()

         # Saving checkpoint
         save_path = saver.save(sess, OUTPUT_DIR + "model.ckpt")

         # Freezing model
         outputs = ['model/transformer/dropout_2/Identity']
         graph_def_freezed = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), outputs)

         # Saving non-frozen and frozen  model to pb
         graph_io.write_graph(sess.graph.as_graph_def(), OUTPUT_DIR, 'model.pb', as_text=False)
         graph_io.write_graph(graph_def_freezed,OUTPUT_DIR, 'model_frozen.pb',
                              as_text=False)

         # Write to tensorboard
         with tf.compat.v1.summary.FileWriter(logdir=OUTPUT_DIR, graph_def=graph_def_freezed) as writer:
            writer.flush()


The script should save into ``~/XLNet-Large/xlnet``.

Converting a frozen TensorFlow XLNet Model to IR
#################################################

To generate the XLNet Intermediate Representation (IR) of the model, run model conversion with the following parameters:

.. code-block:: sh

   mo --input_model path-to-model/model_frozen.pb \
      --input "input_mask[50,1],input_ids[50,1],seg_ids[50,1]"

