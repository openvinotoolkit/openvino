.. {#openvino_sample_automatic_speech_recognition}

[DEPRECATED] Automatic Speech Recognition Sample
====================================================



.. meta::
   :description: Learn how to infer an acoustic model based on Kaldi
                 neural networks and speech feature vectors using Asynchronous
                 Inference Request (Python) API.


.. note::

   This sample is now deprecated and will be removed with OpenVINO 2024.0.
   The sample was mainly designed to demonstrate the features of the GNA plugin
   and the use of models produced by the Kaldi framework. OpenVINO support for
   these components is now deprecated and will be discontinued, making the sample
   redundant.


This sample demonstrates how to do a Synchronous Inference of acoustic model based
on Kaldi neural models and speech feature vectors.

The sample works with Kaldi ARK or Numpy uncompressed NPZ files, so it does not
cover an end-to-end speech recognition scenario (speech to text), requiring additional
preprocessing (feature extraction) to get a feature vector from a speech signal,
as well as postprocessing (decoding) to produce text from scores. Before using the
sample, refer to the following requirements:

- The sample accepts any file format supported by ``core.read_model``.
- The sample has been validated with an acoustic model based on Kaldi neural models
  (see :ref:`Model Preparation <model-preparation-speech>` section)
- To build the sample, use instructions available at :ref:`Build the Sample Applications <build-samples>`
  section in "Get Started with Samples" guide.


How It Works
####################

At startup, the sample application reads command-line parameters, loads a specified
model and input data to the OpenVINO™ Runtime plugin, performs synchronous inference
on all speech utterances stored in the input file, logging each step in a standard output stream.


.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. scrollbox::

         .. doxygensnippet:: samples/python/speech_sample/speech_sample.py
            :language: python


   .. tab-item:: C++
      :sync: cpp

      .. scrollbox::

         .. doxygensnippet:: samples/cpp/speech_sample/main.cpp
            :language: cpp


You can see the explicit description ofeach sample step at
:doc:`Integration Steps <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
section of "Integrate OpenVINO™ Runtime with Your Application" guide.


GNA-specific details
####################

Quantization
++++++++++++++++++++

If the GNA device is selected (for example, using the ``-d`` GNA flag), the GNA
OpenVINO™ Runtime plugin quantizes the model and input feature vector sequence
to integer representation before performing inference.

Several neural model quantization modes:

- *static* - The first utterance in the input file is scanned for dynamic range.
  The scale factor (floating point scalar multiplier) required to scale the maximum
  input value of the first utterance to 16384 (15 bits) is used for all subsequent
  inputs. The model is quantized to accommodate the scaled input dynamic range.
- *user-defined* - The user may specify a scale factor via the ``-sf`` flag that
  will be used for static quantization.

The ``-qb`` flag provides a hint to the GNA plugin regarding the preferred target weight resolution for all layers.
For example, when ``-qb 8`` is specified, the plugin will use 8-bit weights wherever possible in the
model.

.. note::

   It is not always possible to use 8-bit weights due to GNA hardware limitations.
   For example, convolutional layers always use 16-bit weights (GNA hardware version
   1 and 2).  This limitation will be removed in GNA hardware version 3 and higher.

.. _execution-modes:

Execution Modes
++++++++++++++++++++

Several execution modes are supported via the ``-d`` flag:

- ``CPU`` - All calculations are performed on CPU device using CPU Plugin.
- ``GPU`` - All calculations are performed on GPU device using GPU Plugin.
- ``NPU`` - All calculations are performed on NPU device using NPU Plugin.
- ``GNA_AUTO`` - GNA hardware is used if available and the driver is installed. Otherwise, the GNA device is emulated in fast-but-not-bit-exact mode.
- ``GNA_HW`` - GNA hardware is used if available and the driver is installed. Otherwise, an error will occur.
- ``GNA_SW`` - Deprecated. The GNA device is emulated in fast-but-not-bit-exact mode.
- ``GNA_SW_FP32`` - Substitutes parameters and calculations from low precision to floating point (FP32).
- ``GNA_SW_EXACT`` - GNA device is emulated in bit-exact mode.

Loading and Saving Models
+++++++++++++++++++++++++

The GNA plugin supports loading and saving of the GNA-optimized model (non-IR) via the ``-rg`` and ``-wg`` flags.
Thereby, it is possible to avoid the cost of full model quantization at run time.
The GNA plugin also supports export of firmware-compatible embedded model images
for the Intel® Speech Enabling Developer Kit and Amazon Alexa Premium Far-Field
Voice Development Kit via the ``-we`` flag (save only).

In addition to performing inference directly from a GNA model file, these options make it possible to:

- Convert from IR format to GNA format model file (``-m``, ``-wg``)
- Convert from IR format to embedded format model file (``-m``, ``-we``)
- Convert from GNA format to embedded format model file (``-rg``, ``-we``)

Running
####################

Run the application with the ``-h`` option to see the usage message:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         python speech_sample.py -h

      Usage message:

      .. code-block:: console

         usage: speech_sample.py [-h] (-m MODEL | -rg IMPORT_GNA_MODEL) -i INPUT [-o OUTPUT] [-r REFERENCE] [-d DEVICE] [-bs [1-8]]
                                 [-layout LAYOUT] [-qb [8, 16]] [-sf SCALE_FACTOR] [-wg EXPORT_GNA_MODEL]
                                 [-we EXPORT_EMBEDDED_GNA_MODEL] [-we_gen [GNA1, GNA3]]
                                 [--exec_target [GNA_TARGET_2_0, GNA_TARGET_3_0]] [-pc] [-a [CORE, ATOM]] [-iname INPUT_LAYERS]
                                 [-oname OUTPUT_LAYERS] [-cw_l CONTEXT_WINDOW_LEFT] [-cw_r CONTEXT_WINDOW_RIGHT] [-pwl_me PWL_ME]

         optional arguments:
           -m MODEL, --model MODEL
                                 Path to an .xml file with a trained model (required if -rg is missing).
           -rg IMPORT_GNA_MODEL, --import_gna_model IMPORT_GNA_MODEL
                                 Read GNA model from file using path/filename provided (required if -m is missing).

         Options:
           -h, --help            Show this help message and exit.
           -i INPUT, --input INPUT
                                 Required. Path(s) to input file(s).
                                 Usage for a single file/layer: <input_file.ark> or <input_file.npz>.
                                 Example of usage for several files/layers: <layer1>:<port_num1>=<input_file1.ark>,<layer2>:<port_num2>=<input_file2.ark>.
           -o OUTPUT, --output OUTPUT
                                 Optional. Output file name(s) to save scores (inference results).
                                 Usage for a single file/layer: <output_file.ark> or <output_file.npz>.
                                 Example of usage for several files/layers: <layer1>:<port_num1>=<output_file1.ark>,<layer2>:<port_num2>=<output_file2.ark>.
           -r REFERENCE, --reference REFERENCE
                                 Read reference score file(s) and compare inference results with reference scores.
                                 Usage for a single file/layer: <reference_file.ark> or <reference_file.npz>.
                                 Example of usage for several files/layers: <layer1>:<port_num1>=<reference_file1.ark>,<layer2>:<port_num2>=<reference_file2.ark>.
           -d DEVICE, --device DEVICE
                                 Optional. Specify a target device to infer on. CPU, GPU, NPU, GNA_AUTO, GNA_HW, GNA_SW_FP32,
                                 GNA_SW_EXACT and HETERO with combination of GNA as the primary device and CPU as a secondary (e.g.
                                 HETERO:GNA,CPU) are supported. The sample will look for a suitable plugin for device specified.
                                 Default value is CPU.
           -bs [1-8], --batch_size [1-8]
                                 Optional. Batch size 1-8.
           -layout LAYOUT        Optional. Custom layout in format: "input0[value0],input1[value1]" or "[value]" (applied to all
                                 inputs)
           -qb [8, 16], --quantization_bits [8, 16]
                                 Optional. Weight resolution in bits for GNA quantization: 8 or 16 (default 16).
           -sf SCALE_FACTOR, --scale_factor SCALE_FACTOR
                                 Optional. User-specified input scale factor for GNA quantization.
                                 If the model contains multiple inputs, provide scale factors by separating them with commas.
                                 For example: <layer1>:<sf1>,<layer2>:<sf2> or just <sf> to be applied to all inputs.
           -wg EXPORT_GNA_MODEL, --export_gna_model EXPORT_GNA_MODEL
                                 Optional. Write GNA model to file using path/filename provided.
           -we EXPORT_EMBEDDED_GNA_MODEL, --export_embedded_gna_model EXPORT_EMBEDDED_GNA_MODEL
                                 Optional. Write GNA embedded model to file using path/filename provided.
           -we_gen [GNA1, GNA3], --embedded_gna_configuration [GNA1, GNA3]
                                 Optional. GNA generation configuration string for embedded export. Can be GNA1 (default) or GNA3.
           --exec_target [GNA_TARGET_2_0, GNA_TARGET_3_0]
                                 Optional. Specify GNA execution target generation. By default, generation corresponds to the GNA HW
                                 available in the system or the latest fully supported generation by the software. See the GNA
                                 Plugin's GNA_EXEC_TARGET config option description.
           -pc, --performance_counter
                                 Optional. Enables performance report (specify -a to ensure arch accurate results).
           -a [CORE, ATOM], --arch [CORE, ATOM]
                                 Optional. Specify architecture. CORE, ATOM with the combination of -pc.
           -cw_l CONTEXT_WINDOW_LEFT, --context_window_left CONTEXT_WINDOW_LEFT
                                 Optional. Number of frames for left context windows (default is 0). Works only with context window
                                 models. If you use the cw_l or cw_r flag, then batch size argument is ignored.
           -cw_r CONTEXT_WINDOW_RIGHT, --context_window_right CONTEXT_WINDOW_RIGHT
                                 Optional. Number of frames for right context windows (default is 0). Works only with context window
                                 models. If you use the cw_l or cw_r flag, then batch size argument is ignored.
           -pwl_me PWL_ME        Optional. The maximum percent of error for PWL function. The value must be in <0, 100> range. The
                                 default value is 1.0.

   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         speech_sample -h

      Usage message:

      .. code-block:: console

         [ INFO ] OpenVINO Runtime version ......... <version>
         [ INFO ] Build ........... <build>
         [ INFO ]
         [ INFO ] Parsing input parameters

         speech_sample [OPTION]
         Options:

             -h                         Print a usage message.
             -i "<path>"                Required. Path(s) to input file(s). Usage for a single file/layer: <input_file.ark> or <input_file.npz>. Example of usage for several files/layers: <layer1>:<port_num1>=<input_file1.ark>,<layer2>:<port_num2>=<input_file2.ark>.
             -m "<path>"                Required. Path to an .xml file with a trained model (required if -rg is missing).
             -o "<path>"                Optional. Output file name(s) to save scores (inference results). Example of usage for a single file/layer: <output_file.ark> or <output_file.npz>. Example of usage for several files/layers: <layer1>:<port_num1>=<output_file1.ark>,<layer2>:<port_num2>=<output_file2.ark>.
             -d "<device>"              Optional. Specify a target device to infer on. CPU, GPU, NPU, GNA_AUTO, GNA_HW, GNA_HW_WITH_SW_FBACK, GNA_SW_FP32, GNA_SW_EXACT and HETERO with combination of GNA as the primary device and CPU as a secondary (e.g. HETERO:GNA,CPU) are supported. The sample will look for a suitable plugin for device specified.
             -pc                        Optional. Enables per-layer performance report.
             -q "<mode>"                Optional. Input quantization mode for GNA: static (default) or user defined (use with -sf).
             -qb "<integer>"            Optional. Weight resolution in bits for GNA quantization: 8 or 16 (default)
             -sf "<double>"             Optional. User-specified input scale factor for GNA quantization (use with -q user). If the model contains multiple inputs, provide scale factors by separating them with commas. For example: <layer1>:<sf1>,<layer2>:<sf2> or just <sf> to be applied to all inputs.
             -bs "<integer>"            Optional. Batch size 1-8 (default 1)
             -r "<path>"                Optional. Read reference score file(s) and compare inference results with reference scores. Usage for a single file/layer: <reference.ark> or <reference.npz>. Example of usage for several files/layers: <layer1>:<port_num1>=<reference_file1.ark>,<layer2>:<port_num2>=<reference_file2.ark>.
             -rg "<path>"               Read GNA model from file using path/filename provided (required if -m is missing).
             -wg "<path>"               Optional. Write GNA model to file using path/filename provided.
             -we "<path>"               Optional. Write GNA embedded model to file using path/filename provided.
             -cw_l "<integer>"          Optional. Number of frames for left context windows (default is 0). Works only with context window networks. If you use the cw_l or cw_r flag, then batch size argument is ignored.
             -cw_r "<integer>"          Optional. Number of frames for right context windows (default is 0). Works only with context window networks. If you use the cw_r or cw_l flag, then batch size argument is ignored.
             -layout "<string>"         Optional. Prompts how network layouts should be treated by application. For example, "input1[NCHW],input2[NC]" or "[NCHW]" in case of one input size.
             -pwl_me "<double>"         Optional. The maximum percent of error for PWL function.The value must be in <0, 100> range. The default value is 1.0.
             -exec_target "<string>"    Optional. Specify GNA execution target generation. May be one of GNA_TARGET_2_0, GNA_TARGET_3_0. By default, generation corresponds to the GNA HW available in the system or the latest fully supported generation by the software. See the GNA Plugin's GNA_EXEC_TARGET config option description.
             -compile_target "<string>" Optional. Specify GNA compile target generation. May be one of GNA_TARGET_2_0, GNA_TARGET_3_0. By default, generation corresponds to the GNA HW available in the system or the latest fully supported generation by the software. See the GNA Plugin's GNA_COMPILE_TARGET config option description.
             -memory_reuse_off          Optional. Disables memory optimizations for compiled model.

         Available target devices:  CPU  GNA  GPU  NPU



.. _model-preparation-speech:

Model Preparation
####################

You can use the following model conversion command to convert a Kaldi nnet1 or nnet2 model to OpenVINO Intermediate Representation (IR) format:

.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         mo --framework kaldi --input_model wsj_dnn5b.nnet --counts wsj_dnn5b.counts --remove_output_softmax --output_dir <OUTPUT_MODEL_DIR>


   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         mo --framework kaldi --input_model wsj_dnn5b.nnet --counts wsj_dnn5b.counts --remove_output_softmax --output_dir <OUTPUT_MODEL_DIR>


The following pre-trained models are available:

- ``rm_cnn4a_smbr``
- ``rm_lstm4f``
- ``wsj_dnn5b_smbr``

All of them can be downloaded from `the storage <https://storage.openvinotoolkit.org/models_contrib/speech/2021.2>`__ .

Speech Inference
####################

Once the IR has been created, you can do inference on Intel® Processors with the GNA co-processor (or emulation library):


.. tab-set::

   .. tab-item:: Python
      :sync: python

      .. code-block:: console

         python speech_sample.py -m wsj_dnn5b.xml -i dev93_10.ark -r dev93_scores_10.ark -d GNA_AUTO -o result.npz


   .. tab-item:: C++
      :sync: cpp

      .. code-block:: console

         speech_sample -m wsj_dnn5b.xml -i dev93_10.ark -r dev93_scores_10.ark -d GNA_AUTO -o result.ark

      Here, the floating point Kaldi-generated reference neural network scores (``dev93_scores_10.ark``) corresponding to the input feature file (``dev93_10.ark``) are assumed to be available for comparison.

.. note::

   - Before running the sample with a trained model, make sure the model is converted to the intermediate representation (IR) format (\*.xml + \*.bin) using :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`.
   - The sample supports input and output in numpy file format (.npz)
   - When you specify single options multiple times, only the last value will be used. For example, the ``-m`` flag:


     .. tab-set::

        .. tab-item:: Python
           :sync: python

           .. code-block:: console

              python classification_sample_async.py -m model.xml -m model2.xml

        .. tab-item:: C++
           :sync: cpp

           .. code-block:: console

              ./speech_sample -m model.xml -m model2.xml


Sample Output
####################

.. tab-set::

   .. tab-item:: Python
      :sync: python

      The sample application logs each step in a standard output stream.

      .. code-block:: console

         [ INFO ] Creating OpenVINO Runtime Core
         [ INFO ] Reading the model: /models/wsj_dnn5b_smbr_fp32.xml
         [ INFO ] Using scale factor(s) calculated from first utterance
         [ INFO ] For input 0 using scale factor of 2175.4322418
         [ INFO ] Loading the model to the plugin
         [ INFO ] Starting inference in synchronous mode
         [ INFO ]
         [ INFO ] Utterance 0:
         [ INFO ] Total time in Infer (HW and SW): 6326.06ms
         [ INFO ] Frames in utterance: 1294
         [ INFO ] Average Infer time per frame: 4.89ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.7051840
         [ INFO ] avg error: 0.0448388
         [ INFO ] avg rms error: 0.0582387
         [ INFO ] stdev error: 0.0371650
         [ INFO ]
         [ INFO ] Utterance 1:
         [ INFO ] Total time in Infer (HW and SW): 4526.57ms
         [ INFO ] Frames in utterance: 1005
         [ INFO ] Average Infer time per frame: 4.50ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.7575974
         [ INFO ] avg error: 0.0452166
         [ INFO ] avg rms error: 0.0586013
         [ INFO ] stdev error: 0.0372769
         [ INFO ]
         [ INFO ] Utterance 2:
         [ INFO ] Total time in Infer (HW and SW): 6636.56ms
         [ INFO ] Frames in utterance: 1471
         [ INFO ] Average Infer time per frame: 4.51ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.7191710
         [ INFO ] avg error: 0.0472226
         [ INFO ] avg rms error: 0.0612991
         [ INFO ] stdev error: 0.0390846
         [ INFO ]
         [ INFO ] Utterance 3:
         [ INFO ] Total time in Infer (HW and SW): 3927.01ms
         [ INFO ] Frames in utterance: 845
         [ INFO ] Average Infer time per frame: 4.65ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.7436461
         [ INFO ] avg error: 0.0477581
         [ INFO ] avg rms error: 0.0621334
         [ INFO ] stdev error: 0.0397457
         [ INFO ]
         [ INFO ] Utterance 4:
         [ INFO ] Total time in Infer (HW and SW): 3891.49ms
         [ INFO ] Frames in utterance: 855
         [ INFO ] Average Infer time per frame: 4.55ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.7071600
         [ INFO ] avg error: 0.0449147
         [ INFO ] avg rms error: 0.0585048
         [ INFO ] stdev error: 0.0374897
         [ INFO ]
         [ INFO ] Utterance 5:
         [ INFO ] Total time in Infer (HW and SW): 3378.61ms
         [ INFO ] Frames in utterance: 699
         [ INFO ] Average Infer time per frame: 4.83ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.8870468
         [ INFO ] avg error: 0.0479243
         [ INFO ] avg rms error: 0.0625490
         [ INFO ] stdev error: 0.0401951
         [ INFO ]
         [ INFO ] Utterance 6:
         [ INFO ] Total time in Infer (HW and SW): 4034.31ms
         [ INFO ] Frames in utterance: 790
         [ INFO ] Average Infer time per frame: 5.11ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.7648273
         [ INFO ] avg error: 0.0482702
         [ INFO ] avg rms error: 0.0629734
         [ INFO ] stdev error: 0.0404429
         [ INFO ]
         [ INFO ] Utterance 7:
         [ INFO ] Total time in Infer (HW and SW): 2854.04ms
         [ INFO ] Frames in utterance: 622
         [ INFO ] Average Infer time per frame: 4.59ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.7389560
         [ INFO ] avg error: 0.0465543
         [ INFO ] avg rms error: 0.0604941
         [ INFO ] stdev error: 0.0386294
         [ INFO ]
         [ INFO ] Utterance 8:
         [ INFO ] Total time in Infer (HW and SW): 2493.28ms
         [ INFO ] Frames in utterance: 548
         [ INFO ] Average Infer time per frame: 4.55ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.6680136
         [ INFO ] avg error: 0.0439341
         [ INFO ] avg rms error: 0.0574614
         [ INFO ] stdev error: 0.0370353
         [ INFO ]
         [ INFO ] Utterance 9:
         [ INFO ] Total time in Infer (HW and SW): 1654.67ms
         [ INFO ] Frames in utterance: 368
         [ INFO ] Average Infer time per frame: 4.50ms
         [ INFO ]
         [ INFO ] Output blob name: affinetransform14
         [ INFO ] Number scores per frame: 3425
         [ INFO ]
         [ INFO ] max error: 0.6550579
         [ INFO ] avg error: 0.0467643
         [ INFO ] avg rms error: 0.0605045
         [ INFO ] stdev error: 0.0383914
         [ INFO ]
         [ INFO ] Total sample time: 39722.60ms
         [ INFO ] File result.npz was created!
         [ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool


   .. tab-item:: C++
      :sync: cpp

      The sample application logs each step in a standard output stream.

      .. code-block:: console

         [ INFO ] OpenVINO runtime: OpenVINO Runtime version ......... 2022.1.0
         [ INFO ] Build ........... 2022.1.0-6311-a90bb1ff017
         [ INFO ]
         [ INFO ] Parsing input parameters
         [ INFO ] Loading model files:
         [ INFO ] \test_data\models\wsj_dnn5b_smbr_fp32\wsj_dnn5b_smbr_fp32.xml
         [ INFO ] Using scale factor of 2175.43 calculated from first utterance.
         [ INFO ] Model loading time 0.0034 ms
         [ INFO ] Loading model to the device GNA_AUTO
         [ INFO ] Loading model to the device
         [ INFO ] Number scores per frame : 3425
         Utterance 0:
         Total time in Infer (HW and SW):        5687.53 ms
         Frames in utterance:                    1294 frames
         Average Infer time per frame:           4.39531 ms
                  max error: 0.705184
                  avg error: 0.0448388
              avg rms error: 0.0574098
                stdev error: 0.0371649


         End of Utterance 0

         [ INFO ] Number scores per frame : 3425
         Utterance 1:
         Total time in Infer (HW and SW):        4341.34 ms
         Frames in utterance:                    1005 frames
         Average Infer time per frame:           4.31974 ms
                  max error: 0.757597
                  avg error: 0.0452166
              avg rms error: 0.0578436
                stdev error: 0.0372769


         End of Utterance 1

         ...
         End of Utterance X

         [ INFO ] Execution successful


Use of C++ Sample in Kaldi Speech Recognition Pipeline
######################################################

The Wall Street Journal DNN model used in this example was prepared using the
Kaldi s5 recipe and the Kaldi Nnet (nnet1) framework. It is possible to recognize
speech by substituting the ``speech_sample`` for Kaldi's nnet-forward command.
Since the ``speech_sample`` does not yet use pipes, it is necessary to use temporary
files for speaker-transformed feature vectors and scores when running the Kaldi
speech recognition pipeline. The following operations assume that feature extraction
was already performed according to the ``s5`` recipe and that the working directory
within the Kaldi source tree is ``egs/wsj/s5``.

1. Prepare a speaker-transformed feature set, given that the feature transform
   is specified in ``final.feature_transform`` and the feature files are specified in ``feats.scp``:

   .. code-block:: console

      nnet-forward --use-gpu=no final.feature_transform "ark,s,cs:copy-feats scp:feats.scp ark:- |" ark:feat.ark

2. Score the feature set, using the ``speech_sample``:

   .. code-block:: console

      ./speech_sample -d GNA_AUTO -bs 8 -i feat.ark -m wsj_dnn5b.xml -o scores.ark

   OpenVINO™ toolkit Intermediate Representation ``wsj_dnn5b.xml`` file was
   generated in the previous :ref:`Model Preparation <model-preparation-speech>` section.

3. Run the Kaldi decoder to produce n-best text hypotheses and select most likely
   text, given that the WFST (``HCLG.fst``), vocabulary (``words.txt``), and
   TID/PID mapping (``final.mdl``) are specified:

   .. code-block:: console

      latgen-faster-mapped --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.0833 --allow-partial=true    --word-symbol-table=words.txt final.mdl HCLG.fst ark:scores.ark ark:-| lattice-scale --inv-acoustic-scale=13 ark:- ark:- | lattice-best-path    --word-symbol-table=words.txt ark:- ark,t:-  > out.txt &

4. Run the word error rate tool to check accuracy, given that the vocabulary
   (``words.txt``) and reference transcript (``test_filt.txt``) are specified:

   .. code-block:: console

      cat out.txt | utils/int2sym.pl -f 2- words.txt | sed s:\<UNK\>::g | compute-wer --text --mode=present ark:test_filt.txt ark,p:-

   All of the files can be downloaded from `the storage <https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/wsj_dnn5b_smbr>`__


Additional Resources
####################

- :doc:`Integrate the OpenVINO™ Runtime with Your Application <openvino_docs_OV_UG_Integrate_OV_with_your_application>`
- :doc:`Get Started with Samples <openvino_docs_get_started_get_started_demos>`
- :doc:`Using OpenVINO™ Toolkit Samples <openvino_docs_OV_UG_Samples_Overview>`
- :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`
