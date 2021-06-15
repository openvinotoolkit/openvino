# Convert TensorFlow* Attention OCR Model to Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_AttentionOCR_From_Tensorflow}

This tutorial explains how to convert the Attention OCR (AOCR) model from the [TensorFlow* Attention OCR repository](https://github.com/emedvedev/attention-ocr) to the Intermediate Representation (IR).

## Extract model from `aocr` library

The easiest way to get AOCR model is to download python library `aocr`:
```
pip install git+https://github.com/emedvedev/attention-ocr.git@master#egg=aocr
```
This library contains pretrain model and allows to train and run AOCR using command line. After installing `aocr` you can extract model:
```
aocr export --format=frozengraph <model_path>
```
After this step you can find model in <model_path> folder.

## Convert the TensorFlow* AOCR Model to IR

The original AOCR contains data preprocessing which consists of folowing steps:
* Decoding input data to binary format where input data is image as string.
* Resizing binary image to working resolution.

After that resized image is sent to CNN. The Model Optimizer does not support decoding images so you should exclude preprocessing from model converting. 
```sh
python3 path/to/model_optimizer/mo_tf.py \
--input_model=<model_path>/frozen_graph.pb \
--input="map/TensorArrayStack/TensorArrayGatherV3:0[1 32 86 1]" \
--output "transpose_1,transpose_2"
--output_dir path/to/ir/
```

Where:
* `map/TensorArrayStack/TensorArrayGatherV3:0[1 32 86 1]` - result of data preprocessing that was cut.
* `transpose_1,transpose_2' - predictions (int array) and their probabilities (float array).