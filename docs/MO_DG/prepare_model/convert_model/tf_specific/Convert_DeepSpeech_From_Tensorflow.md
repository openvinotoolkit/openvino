# Convert TensorFlow* DeepSpeech Model to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_DeepSpeech_From_Tensorflow}

[DeepSpeech project](https://github.com/mozilla/DeepSpeech) provides an engine to train speech-to-text models.

## Download the Pretrained DeepSpeech Model

Create a directory where model and metagraph with pretrained weights will be stored:
```
mkdir deepspeech
cd deepspeech
```
[Pretrained English speech-to-text model](https://github.com/mozilla/DeepSpeech/releases/tag/v0.8.2) is publicly available. 
To download the model, please follow the instruction below:

* For UNIX*-like systems, run the following command:
```
wget -O - https://github.com/mozilla/DeepSpeech/archive/v0.8.2.tar.gz | tar xvfz -
wget -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-checkpoint.tar.gz | tar xvfz -
```
* For Windows* systems:
  1. Download the archive with the model: [https://github.com/mozilla/DeepSpeech/archive/v0.8.2.tar.gz](https://github.com/mozilla/DeepSpeech/archive/v0.8.2.tar.gz).
  2. Download the TF metagraph with pretrained weights: [https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-checkpoint.tar.gz](https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-checkpoint.tar.gz).
  3. Unpack it with a file archiver application.

## Freeze the model into a *.pb file

After you unpack the archives below, you have to freeze the model. Please note that this requires 
TensorFlow* version 1 which is not available under python 3.8, so you will need python 3.7 or lesser.
Before freezing deploy a virtual environment and install required packages:
```
virtualenv --python=python3.7 venv-deep-speech
cd DeepSpeech-0.8.2
pip3 install -e .
```
Freeze the model with the following command:
```
python3 DeepSpeech.py --checkpoint_dir ../deepspeech-0.8.2-checkpoint --export_dir ../
```
After that you will get the pretrained frozen model file `output_graph.pb` in the directory `deepspeech` created at 
the beginning. The model contains the preprocessing and main parts. The first preprocessing part performs conversion of input 
spectrogram into a form that's useful for speech recognition (mel). This part of the model is not convertible into 
IR because it contains 2 unsupported operations `AudioSpectrogram` and `Mfcc`. 

The main and most computationally expensive part of the model actually converts the preprocessed audio into text. 
The model contains an input with sequence length. So for now we can convert the model with a fixed input length shape, 
thus the model is not [reshapeable](../../../../IE_DG/ShapeInference.md).

## Convert the main part of DeepSpeech Model into IR

The Model Optimizer assumes that the output model is for inference only. That is why you should cut `previous_state_c` 
and `previous_state_h` variables off and resolve keeping cell and hidden states on the application level.

There are certain limitations for the model conversion:
- Time length (`time_len`) and sequence length (`seq_len`) are equal.
- Original model cannot be reshaped, so you should keep original shapes.

To generate the IR run Model Optimizer with the following parameters:
```sh
python3 {path_to_mo}/mo_tf.py                            \
--input_model output_graph.pb                            \
--freeze_placeholder_with_value "input_lengths->[16]"    \
--input "input_node,previous_state_h,previous_state_c"   \
--input_shape "[1,16,19,26],[1,2048],[1,2048]"           \
--output "cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd_1,cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/GatherNd,logits" \
--disable_nhwc_to_nchw
```

Where:
* `--freeze_placeholder_with_value "input_lengths->[16]"` freezes sequence length
* `--input "input_node,previous_state_h,previous_state_c"` and
`--input_shape "[1,16,19,26],[1,2048],[1,2048]"` replace the variables with a placeholder
* `--output ".../GatherNd_1,.../GatherNd,logits" ` gets data for the next model
execution.

The model contains 2 unconnected components. One part performs conversion of input spectrogram into a form that's useful for speech recognition (mel). This part of the model contains 2 unsupported operations AudioSpectrogram and Mfcc. The second part of the model actually converts the input preprocessed audio to text.
The model contains an input with sequence length. So for now we can convert the model with a fixed input length shape, thus the model is not reshape-able.
