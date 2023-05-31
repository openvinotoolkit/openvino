# Quantizing for GNA Device {#pot_example_speech_README}

@sphinxdirective

This example demonstrates the use of the :doc:`Post-training Optimization Tool API <pot_compression_api_README>` for the task of quantizing a speech model for :doc:`GNA <openvino_docs_OV_UG_supported_plugins_GNA>` device. Quantization for GNA is different from CPU quantization due to device specifics: GNA supports quantized inputs in INT16 and INT32 (for activations) precision and quantized weights in INT8 and INT16 precision.

This example contains pre-selected quantization options based on the DefaultQuantization algorithm and created for models from `Kaldi <http://kaldi-asr.org/doc/>`__ framework, and its data format.
A custom ``ArkDataLoader`` is created to load the dataset from files with .ark extension for speech analysis task.

How to Prepare the Data
#######################

To run this example, you will need to use the .ark files for each model input from your ``<DATA_FOLDER>``.
For generating data from original formats to .ark, please, follow the `Kaldi data preparation tutorial <https://kaldi-asr.org/doc/data_prep.html>`__.

How to Run the Example
######################

1. Launch :doc:`model conversion API <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>` with the necessary options (for details follow the :doc:`instructions for Kaldi <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi>` to generate Intermediate Representation (IR) files for the model:

   .. code-block:: sh

      mo --input_model <PATH_TO_KALDI_MODEL> [MODEL_CONVERSION_API_PARAMETERS]


2. Launch the example script:

   .. code-block:: sh

      python3 <POT_DIR>/api/examples/speech/gna_example.py -m <PATH_TO_IR_XML> -w <PATH_TO_IR_BIN> -d <DATA_FOLDER> --input_names [LIST_OF_MODEL_INPUTS] --files_for_input [LIST_OF_INPUT_FILES]


   Required parameters:

   - ``-i``, ``--input_names`` option. Defines the list of model inputs;
   - ``-f``, ``--files_for_input`` option. Defines the list of filenames (.ark) mapped with input names. You should define names without extension, for example: FILENAME_1, FILENAME_2 maps with INPUT_1, INPUT_2.

   Optional parameters:

   - ``-p``, ``--preset`` option. Defines preset for quantization: ``performance`` for INT8 weights, ``accuracy`` for INT16 weights;
   - ``-s``, ``--subset_size`` option. Defines subset size for calibration;
   - ``-o``, ``--output`` option. Defines output folder for the quantized model.

3. Validate your INT8 model using ``./speech_example`` from the Inference Engine examples. Follow the :doc:`speech example description link <openvino_inference_engine_samples_speech_sample_README>` for details.

@endsphinxdirective
