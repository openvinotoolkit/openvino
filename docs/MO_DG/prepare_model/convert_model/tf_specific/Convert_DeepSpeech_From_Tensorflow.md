# Convert TensorFlow* DeepSpeech Model to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_DeepSpeech_From_Tensorflow}

[DeepSpeech project](https://github.com/mozilla/DeepSpeech) provides an engine to train speech-to-text models.

## Download the Pre-Trained DeepSpeech Model

[Pre-trained English speech-to-text model](https://github.com/mozilla/DeepSpeech#getting-the-pre-trained-model)
is publicly available. To download the model, please follow the instruction below:

* For UNIX*-like systems, run the following command:
```
wget -O - https://github.com/mozilla/DeepSpeech/releases/download/v0.3.0/deepspeech-0.3.0-models.tar.gz | tar xvfz -
```
* For Windows* systems:
  1. Download the archive from the DeepSpeech project repository: [https://github.com/mozilla/DeepSpeech/releases/download/v0.3.0/deepspeech-0.3.0-models.tar.gz](https://github.com/mozilla/DeepSpeech/releases/download/v0.3.0/deepspeech-0.3.0-models.tar.gz).
  2. Unpack it with a file archiver application.

After you unpack the archive with the pre-trained model, you will have the new `models` directory with the
following files:
```
alphabet.txt  
lm.binary
output_graph.pb  
output_graph.pbmm  
output_graph.rounded.pb  
output_graph.rounded.pbmm  
trie
```

Pre-trained frozen model file is `output_graph.pb`.

![DeepSpeech model view](../../../img/DeepSpeech.png)

As you can see, the frozen model still has two variables: `previous_state_c` and
`previous_state_h`. It means that the model keeps training those variables at each inference.

At the first inference of this graph, the variables are initialized by zero tensors. After executing the
`lstm_fused_cell` nodes, cell state and hidden state, which are the results of the `BlockLSTM` execution,
are assigned to these two variables.

With each inference of the DeepSpeech graph, initial cell state and hidden state data for `BlockLSTM` is taken
from previous inference from variables. Outputs (cell state and hidden state) of `BlockLSTM` are reassigned
to the same variables.

It helps the model to remember the context of the words that it takes as input.

## Convert the TensorFlow* DeepSpeech Model to IR

The Model Optimizer assumes that the output model is for inference only. That is why you should cut those variables off and
resolve keeping cell and hidden states on the application level.

There are certain limitations for the model conversion:
- Time length (`time_len`) and sequence length (`seq_len`) are equal.
- Original model cannot be reshaped, so you should keep original shapes.

To generate the DeepSpeech Intermediate Representation (IR), provide the TensorFlow DeepSpeech model to the Model Optimizer with the following parameters:
```sh
python3 ./mo_tf.py
--input_model path_to_model/output_graph.pb                         \
--freeze_placeholder_with_value input_lengths->[16]                 \
--input input_node,previous_state_h/read,previous_state_c/read      \
--input_shape [1,16,19,26],[1,2048],[1,2048]                        \
--output raw_logits,lstm_fused_cell/GatherNd,lstm_fused_cell/GatherNd_1 \
--disable_nhwc_to_nchw
```

Where:
* `--freeze_placeholder_with_value input_lengths->[16]` freezes sequence length
* `--input input_node,previous_state_h/read,previous_state_c/read` and
`--input_shape [1,16,19,26],[1,2048],[1,2048]` replace the variables with a placeholder
* `--output raw_logits,lstm_fused_cell/GatherNd,lstm_fused_cell/GatherNd_1` gets data for the next model
execution.
