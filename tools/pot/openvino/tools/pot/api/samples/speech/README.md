# API usage sample for speech task on GNA {#pot_sample_speech_README}

This sample demonstrates the use of the [Post-training Optimization Tool API](@ref pot_compression_api_README) for the task of quantizing a speech model for [GNA](@ref openvino_docs_OV_UG_supported_plugins_GNA) device. 
Quantization for GNA is different from CPU quantization due to device specific: GNA supports quantized inputs in INT16 and INT32 (for activations) precision and quantized weights in INT8 and INT16 precision.

This sample contains pre-selected quantization options based on the DefaultQuantization algorithm and created for models from [Kaldi](http://kaldi-asr.org/doc/) framework, and its data format.
A custom `ArkDataLoader` is created to load the dataset from files with .ark extension for speech analysis task.

## How to prepare the data

To run this sample, you will need to use the .ark files for each model input from your `<DATA_FOLDER>`.
For generating data from original formats to .ark, please, follow the [Kaldi data preparation tutorial](https://kaldi-asr.org/doc/data_prep.html).

## How to Run the Sample
In the instructions below, the Post-Training Optimization Tool directory `<POT_DIR>` is referred to:
- `<ENV>/lib/python<version>/site-packages/` in the case of PyPI installation, where `<ENV>` is a Python* 
   environment where OpenVINO is installed and `<version>` is a Python* version, e.g. `3.6`.
- `<INSTALL_DIR>/deployment_tools/tools/post_training_optimization_toolkit` in the case of OpenVINO distribution package. 
  `<INSTALL_DIR>` is the directory where Intel&reg; Distribution of OpenVINO&trade; toolkit is installed.

1. To get started, follow the [Installation Guide](@ref pot_InstallationGuide).
2. Launch [Model Optimizer](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide) with the necessary options (for details follow the [instructions for Kaldi](@ref openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi) to generate Intermediate Representation (IR) files for the model:
   ```sh
   python3 <PATH_TO_MODEL_OPTIMIZER>/mo.py --input_model <PATH_TO_KALDI_MODEL> [MODEL_OPTIMIZER_OPTIONS]
   ```
3. Launch the sample script:
   ```sh
   python3 <POT_DIR>/api/samples/speech/gna_sample.py -m <PATH_TO_IR_XML> -w <PATH_TO_IR_BIN> -d <DATA_FOLDER> --input_names [LIST_OF_MODEL_INPUTS] --files_for_input [LIST_OF_INPUT_FILES]
   ```
   Required parameters:
   - `-i`, `--input_names` option. Defines list of model inputs;
   - `-f`, `--files_for_input` option. Defines list of filenames (.ark) mapped with input names. You should define names without extension, for example: FILENAME_1, FILENAME_2 maps with INPUT_1, INPUT_2.  
  
  
   Optional parameters:
    - `-p`, `--preset` option. Defines preset for quantization: `performance` for INT8 weights, `accuracy` for INT16 weights;
    - `-s`, `--subset_size` option. Defines subset size for calibration;
    - `-o`, `--output` option. Defines output folder for quantized model.
4. Validate your INT8 model using `./speech_sample` from the Inference Engine samples. Follow the [speech sample description link](@ref openvino_inference_engine_samples_speech_sample_README) for details.
