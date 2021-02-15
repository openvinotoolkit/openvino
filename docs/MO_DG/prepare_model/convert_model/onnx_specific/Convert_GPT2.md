# Convert ONNX* GPT-2 Model to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_onnx_specific_Convert_GPT2}

[Public pre-trained GPT-2 model](https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2)  is a large
transformer-based language model with a simple objective: predict the next word, given all of the previous words within some text.

## Download the Pre-Trained Base GPT-2 Model

To download the model, click **Download** on [https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-10.onnx](https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-10.onnx).

To download the model and sample test data, click **Download** on [https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz](https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-10.tar.gz).

## Convert ONNX* GPT-2 Model to IR

To generate the Intermediate Representation (IR) of the model GPT-2, run the Model Optimizer with the following parameters:
```sh
python3 mo.py --input_model gpt2-10.onnx --input_shape [X,Y,Z]
```
