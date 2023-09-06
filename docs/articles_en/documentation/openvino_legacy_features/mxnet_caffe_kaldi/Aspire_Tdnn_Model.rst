.. {#openvino_docs_MO_DG_prepare_model_convert_model_kaldi_specific_Aspire_Tdnn_Model}

Converting a Kaldi ASpIRE Chain Time Delay Neural Network (TDNN) Model
======================================================================


.. meta::
   :description: Learn how to convert an ASpIRE Chain TDNN
                 model from Kaldi to the OpenVINO Intermediate Representation.


.. warning::

   Note that OpenVINO support for Kaldi is currently being deprecated and will be removed entirely in the future.

At the beginning, you should `download a pre-trained model <https://kaldi-asr.org/models/1/0001_aspire_chain_model.tar.gz>`__
for the ASpIRE Chain Time Delay Neural Network (TDNN) from the Kaldi project official website.

Converting an ASpIRE Chain TDNN Model to IR
###########################################

Generate the Intermediate Representation of the model by running model conversion with the following parameters:

.. code-block:: sh

   mo --input_model exp/chain/tdnn_7b/final.mdl --output output


The IR will have two inputs: ``input`` for data, and ``ivector`` for ivectors.

Example: Running ASpIRE Chain TDNN Model with the Speech Recognition Sample
###########################################################################

.. note::

   Before you continue with this part of the article, get familiar with the
   :doc:`Speech Recognition sample <openvino_inference_engine_samples_speech_sample_README>`.

In this example, the input data contains one utterance from one speaker.

To run the ASpIRE Chain TDNN Model with Speech Recognition sample, You need to prepare environment. Do it by following the steps below :

1. Download a `Kaldi repository <https://github.com/kaldi-asr/kaldi>`__.
2. Build it by following instructions in ``README.md`` from the repository.
3. Download the `model archive <https://kaldi-asr.org/models/1/0001_aspire_chain_model.tar.gz>`__ from Kaldi website.
4. Extract the downloaded model archive to the ``egs/aspire/s5`` folder of the Kaldi repository.

Once everything has been prepared, you can start a proper run:

1. Prepare the model for decoding. Refer to the ``README.txt`` file from the downloaded model archive for instructions.
2. Convert data and ivectors to ``.ark`` format. Refer to the corresponding sections below for instructions.

Preparing Data
++++++++++++++++++++

If you have a ``.wav`` data file, convert it to the ``.ark`` format using the following command:

.. code-block:: sh

   <path_to_kaldi_repo>/src/featbin/compute-mfcc-feats --config=<path_to_kaldi_repo>/egs/aspire/s5/conf/mfcc_hires.conf scp:./wav.scp ark,scp:feats.ark,feats.scp


Add the ``feats.ark`` absolute path to ``feats.scp`` to avoid errors in later commands.

Preparing Ivectors
++++++++++++++++++++

Prepare ivectors for the Speech Recognition sample:

1. Copy the ``feats.scp`` file to the ``egs/aspire/s5/`` directory of the built Kaldi repository and navigate there:

   .. code-block:: sh

      cp feats.scp <path_to_kaldi_repo>/egs/aspire/s5/
      cd <path_to_kaldi_repo>/egs/aspire/s5/


2. Extract ivectors from the data:

   .. code-block:: sh

      ./steps/online/nnet2/extract_ivectors_online.sh --nj 1 --ivector_period <max_frame_count_in_utterance> <data folder> exp/tdnn_7b_chain_online/ivector_extractor <ivector    folder>


   You can simplify the preparation of ivectors for the Speech Recognition sample. To do it, specify the maximum number of frames in utterances as a parameter for    ``--ivector_period`` to get only one ivector per utterance.

   To get the maximum number of frames in utterances, use the following command line:

   .. code-block:: sh

      ../../../src/featbin/feat-to-len scp:feats.scp ark,t: | cut -d' ' -f 2 - | sort -rn | head -1


   As a result, you will find the ``ivector_online.1.ark`` file in ``<ivector folder>``.

3. Go to the ``<ivector folder>``:

   .. code-block:: sh

      cd <ivector folder>


4. Convert the ``ivector_online.1.ark`` file to text format, using the ``copy-feats`` tool. Run the following command:

   .. code-block:: sh

      <path_to_kaldi_repo>/src/featbin/copy-feats --binary=False ark:ivector_online.1.ark ark,t:ivector_online.1.ark.txt


5. For the Speech Recognition sample, the ``.ark`` file must contain an ivector for each frame. Copy the ivector ``frame_count`` times by running the below script in the Python command prompt:

   .. code-block:: py
      :force:

      import subprocess

      subprocess.run(["<path_to_kaldi_repo>/src/featbin/feat-to-len", "scp:<path_to_kaldi_repo>/egs/aspire/s5/feats.scp", "ark,t:feats_length.txt"])

      f = open("ivector_online.1.ark.txt", "r")
      g = open("ivector_online_ie.ark.txt", "w")
      length_file = open("feats_length.txt", "r")
      for line in f:
          if "[" not in line:
              for i in range(frame_count):
                  line = line.replace("]", " ")
                  g.write(line)
          else:
              g.write(line)
              frame_count = int(length_file.read().split(" ")[1])
      g.write("]")
      f.close()
      g.close()
      length_file.close()


6. Create an ``.ark`` file from ``.txt``:

   .. code-block:: sh

      <path_to_kaldi_repo>/src/featbin/copy-feats --binary=True ark,t:ivector_online_ie.ark.txt ark:ivector_online_ie.ark


Running the Speech Recognition Sample
+++++++++++++++++++++++++++++++++++++

Run the Speech Recognition sample with the created ivector ``.ark`` file:

.. code-block:: sh

   speech_sample -i feats.ark,ivector_online_ie.ark -m final.xml -d CPU -o prediction.ark -cw_l 17 -cw_r 12


Results can be decoded as described in "Use of Sample in Kaldi Speech Recognition Pipeline"
in the :doc:`Speech Recognition Sample description <openvino_inference_engine_samples_speech_sample_README>` article.

