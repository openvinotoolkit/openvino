Converting a TensorFlow BERT Model
==================================


.. meta::
   :description: Learn how to convert a BERT model
                 from TensorFlow to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

Pretrained models for BERT (Bidirectional Encoder Representations from Transformers) are
`publicly available <https://github.com/google-research/bert>`__.

.. _supported_models:

Supported Models
################

The following models from the pretrained `BERT model list <https://github.com/google-research/bert#pre-trained-models>`__ are currently supported:

* ``BERT-Base, Cased``
* ``BERT-Base, Uncased``
* ``BERT-Base, Multilingual Cased``
* ``BERT-Base, Multilingual Uncased``
* ``BERT-Base, Chinese``
* ``BERT-Large, Cased``
* ``BERT-Large, Uncased``

Downloading the Pretrained BERT Model
#####################################

Download and unzip an archive with the `BERT-Base, Multilingual Uncased Model <https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip>`__.

After the archive is unzipped, the directory ``uncased_L-12_H-768_A-12`` is created and contains the following files:

* ``bert_config.json``
* ``bert_model.ckpt.data-00000-of-00001``
* ``bert_model.ckpt.index``
* ``bert_model.ckpt.meta``
* ``vocab.txt``

Pretrained model meta-graph files are ``bert_model.ckpt.*``.

Converting a TensorFlow BERT Model to IR
#########################################

To generate the BERT Intermediate Representation (IR) of the model, run model conversion with the following parameters:

.. code-block:: sh

    mo \
   --input_meta_graph uncased_L-12_H-768_A-12/bert_model.ckpt.meta \
   --output bert/pooler/dense/Tanh                                 \
   --input Placeholder{i32},Placeholder_1{i32},Placeholder_2{i32}


Pretrained models are not suitable for batch reshaping out-of-the-box because of multiple hardcoded shapes in the model.

Converting a Reshapable TensorFlow BERT Model to OpenVINO IR
=============================================================

Follow these steps to make a pretrained TensorFlow BERT model reshapable over batch dimension:

1. Download a pretrained BERT model you want to use from the `Supported Models list <#supported_models>`__.

2. Clone google-research/bert git repository:

   .. code-block:: sh

      https://github.com/google-research/bert.git

3. Go to the root directory of the cloned repository:

   .. code-block:: sh

      cd bert

4. (Optional) Checkout to the commit that the conversion was tested on:

   .. code-block:: sh

      git checkout eedf5716c

5. Download script to load GLUE data:

   * For UNIX-like systems, run the following command:

     .. code-block:: sh

        wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py

   * For Windows systems:

     Download the `Python script <https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py>`__ to the current working directory.

6. Download GLUE data by running:

   .. code-block:: sh

      python3 download_glue_data.py --tasks MRPC

7. Open the file ``modeling.py`` in the text editor and delete lines 923-924. They should look like this:

   .. code-block:: py
      :force:

       if not non_static_indexes:
           return shape

8. Open the file ``run_classifier.py`` and insert the following code after the line 645:

   .. code-block:: py
      :force:

       import os, sys
       import tensorflow as tf
       from tensorflow.python.framework import graph_io
       with tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph()) as sess:
           (assignment_map, initialized_variable_names) = \
               modeling.get_assignment_map_from_checkpoint(tf.compat.v1.trainable_variables(), init_checkpoint)
           tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
           sess.run(tf.compat.v1.global_variables_initializer())
           frozen = tf.compat.v1.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["bert/pooler/dense/Tanh"])
           graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)
       print('BERT frozen model path {}'.format(os.path.join(os.path.dirname(__file__), 'inference_graph.pb')))
       sys.exit(0)

   Lines before the inserted code should look like this:

   .. code-block:: py
      :force:

       (total_loss, per_example_loss, logits, probabilities) = create_model(
           bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
           num_labels, use_one_hot_embeddings)


9. Set environment variables ``BERT_BASE_DIR``, ``BERT_REPO_DIR`` and run the script ``run_classifier.py`` to create ``inference_graph.pb`` file in the root of the cloned BERT repository.

   .. code-block:: sh

      export BERT_BASE_DIR=/path/to/bert/uncased_L-12_H-768_A-12
      export BERT_REPO_DIR=/current/working/directory

      python3 run_classifier.py \
          --task_name=MRPC \
          --do_eval=true \
          --data_dir=$BERT_REPO_DIR/glue_data/MRPC \
          --vocab_file=$BERT_BASE_DIR/vocab.txt \
          --bert_config_file=$BERT_BASE_DIR/bert_config.json \
          --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
          --output_dir=./

   Run model conversion with the following command line parameters to generate reshape-able BERT Intermediate Representation (IR):

   .. code-block:: sh

       mo \
      --input_model inference_graph.pb \
      --input "IteratorGetNext:0{i32}[1,128],IteratorGetNext:1{i32}[1,128],IteratorGetNext:4{i32}[1,128]"

For other applicable parameters, refer to the :doc:`Convert Model from TensorFlow <../[legacy]-convert-tensorflow>` guide.

For more information about reshape abilities, refer to the :doc:`Using Shape Inference <../../../../../../openvino-workflow/running-inference/changing-input-shape>` guide.

