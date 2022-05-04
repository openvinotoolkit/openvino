# Converting a TensorFlow Attention OCR Model {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_AttentionOCR_From_Tensorflow}

This tutorial explains how to convert the Attention OCR (AOCR) model from the [TensorFlow Attention OCR repository](https://github.com/emedvedev/attention-ocr) to the Intermediate Representation (IR).

## Extracting a Model from *`aocr`* Library

To get an AOCR model, download *`aocr`* Python library:
```
pip install git+https://github.com/emedvedev/attention-ocr.git@master#egg=aocr
```
This library contains a pretrained model and allows training and running AOCR, using the command line. After installation of *`aocr`*, extract the model:
```
aocr export --format=frozengraph model/path/
```
Then, you can find the model in *`model/path/`* folder.

## Converting the TensorFlow AOCR Model to IR

The original AOCR model contains data preprocessing, done as follows:
* Decoding input data to binary format where input data is an image represented as a string.
* Resizing binary image to working resolution.

After that, the resized image is sent to the convolution neural network (CNN). The Model Optimizer does not support image decoding. The preprocessing part of the model should be cut off by using *`--input`* command line parameter.
```sh
mo \
--input_model=model/path/frozen_graph.pb \
--input="map/TensorArrayStack/TensorArrayGatherV3:0[1 32 86 1]" \
--output "transpose_1,transpose_2" \
--output_dir path/to/ir/
```

Where:
* *`map/TensorArrayStack/TensorArrayGatherV3:0[1 32 86 1]`* - name of node producing tensor after preprocessing.
* *`transpose_1`* - name of the node producing tensor with predicted characters.
* *`transpose_2`* - name of the node producing tensor with predicted characters probabilities.