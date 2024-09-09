Converting a TensorFlow GNMT Model
==================================


.. meta::
   :description: Learn how to convert a GNMT model
                 from TensorFlow to the OpenVINO Intermediate Representation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

This tutorial explains how to convert Google Neural Machine Translation (GNMT) model to the Intermediate Representation (IR).

There are several public versions of TensorFlow GNMT model implementation available on GitHub. This tutorial explains how to convert the GNMT model from the `TensorFlow Neural Machine Translation (NMT) repository <https://github.com/tensorflow/nmt>`__ to the IR.

Creating a Patch File
#####################

Before converting the model, you need to create a patch file for the repository. The patch modifies the framework code by adding a special command-line argument to the framework options that enables inference graph dumping:

1. Go to a writable directory and create a ``GNMT_inference.patch`` file.
2. Copy the following diff code to the file:

   .. code-block:: py

      diff --git a/nmt/inference.py b/nmt/inference.py
      index 2cbef07..e185490 100644
      --- a/nmt/inference.py
      +++ b/nmt/inference.py
      @@ -17,9 +17,11 @@
      from __future__ import print_function

      import codecs
      +import os
      import time

      import tensorflow as tf
      +from tensorflow.python.framework import graph_io

      from . import attention_model
      from . import gnmt_model
      @@ -105,6 +107,29 @@ def start_sess_and_load_model(infer_model, ckpt_path):
         return sess, loaded_infer_model


      +def inference_dump_graph(ckpt_path, path_to_dump, hparams, scope=None):
      +    model_creator = get_model_creator(hparams)
      +    infer_model = model_helper.create_infer_model(model_creator, hparams, scope)
      +    sess = tf.Session(
      +        graph=infer_model.graph, config=utils.get_config_proto())
      +    with infer_model.graph.as_default():
      +        loaded_infer_model = model_helper.load_model(
      +            infer_model.model, ckpt_path, sess, "infer")
      +    utils.print_out("Dumping inference graph to {}".format(path_to_dump))
      +    loaded_infer_model.saver.save(
      +        sess,
      +        os.path.join(path_to_dump + 'inference_GNMT_graph')
      +        )
      +    utils.print_out("Dumping done!")
      +
      +    output_node_name = 'index_to_string_Lookup'
      +    utils.print_out("Freezing GNMT graph with output node {}...".format(output_node_name))
      +    frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def,
      +                                                          [output_node_name])
      +    graph_io.write_graph(frozen, '.', os.path.join(path_to_dump, 'frozen_GNMT_inference_graph.pb'), as_text=False)
      +    utils.print_out("Freezing done. Freezed model frozen_GNMT_inference_graph.pb saved to {}".format(path_to_dump))
      +
      +
      def inference(ckpt_path,
                     inference_input_file,
                     inference_output_file,
      diff --git a/nmt/nmt.py b/nmt/nmt.py
      index f5823d8..a733748 100644
      --- a/nmt/nmt.py
      +++ b/nmt/nmt.py
      @@ -310,6 +310,13 @@ def add_arguments(parser):
         parser.add_argument("--num_intra_threads", type=int, default=0,
                           help="number of intra_op_parallelism_threads")

      +  # Special argument for inference model dumping without inference
      +  parser.add_argument("--dump_inference_model", type="bool", nargs="?",
      +                      const=True, default=False,
      +                      help="Argument for dump inference graph for specified trained ckpt")
      +
      +  parser.add_argument("--path_to_dump", type=str, default="",
      +                      help="Path to dump inference graph.")

      def create_hparams(flags):
         """Create training hparams."""
      @@ -396,6 +403,9 @@ def create_hparams(flags):
            language_model=flags.language_model,
            num_intra_threads=flags.num_intra_threads,
            num_inter_threads=flags.num_inter_threads,
      +
      +      dump_inference_model=flags.dump_inference_model,
      +      path_to_dump=flags.path_to_dump,
         )


      @@ -613,7 +623,7 @@ def create_or_load_hparams(
         return hparams


      -def run_main(flags, default_hparams, train_fn, inference_fn, target_session=""):
      +def run_main(flags, default_hparams, train_fn, inference_fn, inference_dump, target_session=""):
         """Run main."""
         # Job
         jobid = flags.jobid
      @@ -653,8 +663,26 @@ def run_main(flags, default_hparams, train_fn, inference_fn, target_session=""):
               out_dir, default_hparams, flags.hparams_path,
               save_hparams=(jobid == 0))

      -  ## Train / Decode
      -  if flags.inference_input_file:
      +  #  Dumping inference model
      +  if flags.dump_inference_model:
      +      # Inference indices
      +      hparams.inference_indices = None
      +      if flags.inference_list:
      +          (hparams.inference_indices) = (
      +              [int(token) for token in flags.inference_list.split(",")])
      +
      +      # Ckpt
      +      ckpt = flags.ckpt
      +      if not ckpt:
      +          ckpt = tf.train.latest_checkpoint(out_dir)
      +
      +      # Path to dump graph
      +      assert flags.path_to_dump != "", "Please, specify path_to_dump model."
      +      path_to_dump = flags.path_to_dump
      +      if not tf.gfile.Exists(path_to_dump): tf.gfile.MakeDirs(path_to_dump)
      +
      +      inference_dump(ckpt, path_to_dump, hparams)
      +  elif flags.inference_input_file:
         # Inference output directory
         trans_file = flags.inference_output_file
         assert trans_file
      @@ -693,7 +721,8 @@ def main(unused_argv):
         default_hparams = create_hparams(FLAGS)
         train_fn = train.train
         inference_fn = inference.inference
      -  run_main(FLAGS, default_hparams, train_fn, inference_fn)
      +  inference_dump = inference.inference_dump_graph
      +  run_main(FLAGS, default_hparams, train_fn, inference_fn, inference_dump)


      if __name__ == "__main__":


3. Save and close the file.

Converting a GNMT Model to the IR
#################################

.. note:: Use TensorFlow version 1.13 or lower.

**Step 1**. Clone the GitHub repository and check out the commit:

1. Clone the NMT repository:

   .. code-block:: sh

      git clone https://github.com/tensorflow/nmt.git

2. Check out the necessary commit:

   .. code-block:: sh

      git checkout b278487980832417ad8ac701c672b5c3dc7fa553


**Step 2**. Get a trained model. You have two options:

* Train the model with the GNMT ``wmt16_gnmt_4_layer.json`` or ``wmt16_gnmt_8_layer.json`` configuration file using the NMT framework.
* *Do not use the pre-trained checkpoints provided in the NMT repository, as they are outdated and can be incompatible with the current repository version.*

This tutorial assumes the use of the trained GNMT model from ``wmt16_gnmt_4_layer.json`` config, German to English translation.

**Step 3**. Create an inference graph:

The OpenVINO assumes that a model is used for inference only. Hence, before converting the model into the IR, you need to transform the training graph into the inference graph.
For the GNMT model, the training graph and the inference graph have different decoders: the training graph uses a greedy search decoding algorithm, while the inference graph uses a beam search decoding algorithm.

1. Apply the ``GNMT_inference.patch`` patch to the repository. `Create a Patch File <#Creating-a-Patch-File>`__ instructions if you do not have it:

   .. code-block:: sh

      git apply /path/to/patch/GNMT_inference.patch


2. Run the NMT framework to dump the inference model:

   .. code-block:: sh

      python -m nmt.nmt
         --src=de
         --tgt=en
         --ckpt=/path/to/ckpt/translate.ckpt
         --hparams_path=/path/to/repository/nmt/nmt/standard_hparams/wmt16_gnmt_4_layer.json
         --vocab_prefix=/path/to/vocab/vocab.bpe.32000
         --out_dir=""
         --dump_inference_model
         --infer_mode beam_search
         --path_to_dump /path/to/dump/model/


If you use different checkpoints, use the corresponding values for the ``src``, ``tgt``, ``ckpt``, ``hparams_path``, and ``vocab_prefix`` parameters.
Inference checkpoint ``inference_GNMT_graph`` and frozen inference graph ``frozen_GNMT_inference_graph.pb`` will appear in the ``/path/to/dump/model/`` folder.

To generate ``vocab.bpe.32000``, execute the ``nmt/scripts/wmt16_en_de.sh`` script. If you face an issue of a size mismatch between the checkpoint graph's embedding layer and vocabulary (both src and target), make sure you add the following code to the ``nmt.py`` file to the ``extend_hparams`` function after the line 508 (after initialization of the ``src_vocab_size`` and ``tgt_vocab_size`` variables):

.. code-block:: py
   :force:

   src_vocab_size -= 1
   tgt_vocab_size -= 1


**Step 4**. Convert the model to the IR:

.. code-block:: sh

   mo
   --input_model /path/to/dump/model/frozen_GNMT_inference_graph.pb
   --input "IteratorGetNext:1{i32}[1],IteratorGetNext:0{i32}[1,50],dynamic_seq2seq/hash_table_Lookup_1:0[1]->[2],dynamic_seq2seq/hash_table_Lookup:0[1]->[1]"
   --output dynamic_seq2seq/decoder/decoder/GatherTree
   --output_dir /path/to/output/IR/


Input and output cutting with the ``--input`` and ``--output`` options is required since OpenVINO™ does not support ``IteratorGetNext`` and ``LookupTableFindV2`` operations.

Input cutting:

* ``IteratorGetNext`` operation iterates over a dataset. It is cut by output ports: port 0 contains data tensor with shape ``[batch_size, max_sequence_length]``, port 1 contains ``sequence_length`` for every batch with shape ``[batch_size]``.

* ``LookupTableFindV2`` operations (``dynamic_seq2seq/hash_table_Lookup_1`` and ``dynamic_seq2seq/hash_table_Lookup`` nodes in the graph) are cut with constant values).

Output cutting:

* ``LookupTableFindV2`` operation is cut from the output and the ``dynamic_seq2seq/decoder/decoder/GatherTree`` node is treated as a new exit point.

For more information about model cutting, refer to the :doc:`Cutting Off Parts of a Model <../../[legacy]-cutting-parts-of-a-model>` guide.

Using a GNMT Model
##################

.. note::

   This step assumes you have converted a model to the Intermediate Representation.

Inputs of the model:

* ``IteratorGetNext/placeholder_out_port_0`` input with shape ``[batch_size, max_sequence_length]`` contains ``batch_size`` decoded input sentences. Every sentence is decoded the same way as indices of sentence elements in vocabulary and padded with index of ``eos`` (end of sentence symbol). If the length of the sentence is less than ``max_sequence_length``, remaining elements are filled with index of ``eos`` token.

* ``IteratorGetNext/placeholder_out_port_1`` input with shape ``[batch_size]`` contains sequence lengths for every sentence from the first input. For example, if ``max_sequence_length = 50``, ``batch_size = 1`` and the sentence has only 30 elements, then the input tensor for ``IteratorGetNext/placeholder_out_port_1`` should be ``[30]``.


Outputs of the model:

* ``dynamic_seq2seq/decoder/decoder/GatherTree`` tensor with shape ``[max_sequence_length * 2, batch, beam_size]``,
  that contains ``beam_size`` best translations for every sentence from input (also decoded as indices of words in
  vocabulary).

.. note::
   The shape of this tensor in TensorFlow can be different: instead of ``max_sequence_length * 2``, it can be any value less than that, because OpenVINO does not support dynamic shapes of outputs, while TensorFlow can stop decoding iterations when ``eos`` symbol is generated.

Running GNMT IR
---------------

1. With benchmark app:

   .. code-block:: sh

      benchmark_app -m <path to the generated GNMT IR> -d CPU


2. With OpenVINO Runtime Python API:

   .. note::

      Before running the example, insert a path to your GNMT ``.xml`` and ``.bin`` files into ``MODEL_PATH`` and ``WEIGHTS_PATH``, and fill ``input_data_tensor`` and    ``seq_lengths`` tensors according to your input data.

   .. code-block:: py
      :force:

      from openvino.inference_engine import IENetwork, IECore

      MODEL_PATH = '/path/to/IR/frozen_GNMT_inference_graph.xml'
      WEIGHTS_PATH = '/path/to/IR/frozen_GNMT_inference_graph.bin'

      # Creating network
      net = IENetwork(
         model=MODEL_PATH,
         weights=WEIGHTS_PATH)

      # Creating input data
      input_data = {'IteratorGetNext/placeholder_out_port_0': input_data_tensor,
                  'IteratorGetNext/placeholder_out_port_1': seq_lengths}

      # Creating plugin and loading extensions
      ie = IECore()
      ie.add_extension(extension_path="libcpu_extension.so", device_name="CPU")

      # Loading network
      exec_net = ie.load_network(network=net, device_name="CPU")

      # Run inference
      result_ie = exec_net.infer(input_data)


For more information about Python API, refer to the :doc:`OpenVINO Runtime Python API <../../../../../../api/ie_python_api/api>` guide.

