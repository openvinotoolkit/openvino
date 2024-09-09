Converting a TensorFlow Language Model on One Billion Word Benchmark
====================================================================


.. meta::
   :description: Learn how to convert a TensorFlow Language
                 Model on One Billion Word Benchmark to the OpenVINO Intermediate
                 Representation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

Downloading a Pre-trained Language Model on One Billion Word Benchmark
######################################################################

TensorFlow provides a pretrained `Language Model on One Billion Word Benchmark <https://github.com/tensorflow/models/tree/r2.3.0/research/lm_1b>`__.

To download the model for IR conversion, follow the instructions:

1. Create new directory to store the model:

   .. code-block:: sh

      mkdir lm_1b

2. Go to the ``lm_1b`` directory:

   .. code-block:: sh

      cd lm_1b

3. Download the model GraphDef file:

   .. code-block:: sh

      wget http://download.tensorflow.org/models/LM_LSTM_CNN/graph-2016-09-10.pbtxt

4. Create new directory to store 12 checkpoint shared files:

   .. code-block:: sh

      mkdir ckpt

5. Go to the ``ckpt`` directory:

   .. code-block:: sh

      cd ckpt

6. Download 12 checkpoint shared files:

   .. code-block:: sh

       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-base
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-char-embedding
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-lstm
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax0
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax1
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax2
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax3
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax4
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax5
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax6
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax7
       wget http://download.tensorflow.org/models/LM_LSTM_CNN/all_shards-2016-09-10/ckpt-softmax8


Once you have downloaded the pretrained model files, you will have the ``lm_1b`` directory with the following hierarchy:

.. code-block:: sh

    lm_1b/
        graph-2016-09-10.pbtxt
        ckpt/
            ckpt-base
            ckpt-char-embedding
            ckpt-lstm
            ckpt-softmax0
            ckpt-softmax1
            ckpt-softmax2
            ckpt-softmax3
            ckpt-softmax4
            ckpt-softmax5
            ckpt-softmax6
            ckpt-softmax7
            ckpt-softmax8



.. image:: ../../../../../../assets/images/lm_1b.svg

The frozen model still has two variables: ``Variable`` and ``Variable_1``.
It means that the model keeps training those variables at each inference.

At the first inference of this graph, the variables are initialized by initial values.
After executing the ``lstm`` nodes, results of execution are assigned to these two variables.

With each inference of the ``lm_1b`` graph, ``lstm`` initial states data is taken from previous inference
from variables, and states of current inference of ``lstm`` is reassigned to the same variables.

It helps the model to remember the context of the words that it takes as input.

Converting a TensorFlow Language Model on One Billion Word Benchmark to IR
##########################################################################

Model Optimizer assumes that output model is for inference only.
Therefore, you should cut those variables off and resolve keeping cell and hidden states on application level.

There is a certain limitation for the model conversion: the original model cannot be reshaped, so you should keep original shapes.

To generate the ``lm_1b`` Intermediate Representation (IR), provide TensorFlow ``lm_1b`` model to the
Model Optimizer with parameters:

.. code-block:: sh

    mo
    --input_model lm_1b/graph-2016-09-10.pbtxt  \
    --input_checkpoint lm_1b/ckpt               \
    --input_model_is_text                       \
    --input_shape [50],[50],[1,9216],[1,9216]    \
    --output softmax_out,lstm/lstm_0/concat_2,lstm/lstm_1/concat_2 \
    --input char_embedding/EmbeddingLookupUnique/Unique:0,char_embedding/EmbeddingLookupUnique/Unique:1,Variable/read,Variable_1/read

Where:

* ``--input char_embedding/EmbeddingLookupUnique/Unique:0,char_embedding/EmbeddingLookupUnique/Unique:1,Variable/read,Variable_1/read`` and ``--input_shape [50],[50],[1,9216],[1,9216]`` replace the variables with a placeholder.
* ``--output softmax_out,lstm/lstm_0/concat_2,lstm/lstm_1/concat_2`` specifies output node name and names of LSTM cell states.

