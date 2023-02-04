# Basic Quantization Flow {#basic_qauntization_flow}

## Introduction

The basic quantization flow is the simplest way to apply 8-bit quantization to the model. It is available for models in the following frameworks: PyTorch, TensorFlow 2.x, ONNX, and OpenVINO. The basic quantization flow is based on the following steps:
* Set up an environment and install dependencies.
* Prepare the **calibration dataset** that is used to estimate quantization parameters of the activations within the model.
* Call the quantization API to apply 8-bit quantization to the model.

## Set up an Environment

It is recommended to set up a separate Python environment for quantization with NNCF. To do this, run the following command:
```bash
python3 -m venv nncf_ptq_env
```
Install all the packages required to instantiate the model object, for example, DL framework. After that, install NNCF on top of the environment:
```bash
pip install nncf
```

## Prepare a Calibration Dataset

At this step, create an instance of the `nncf.Dataset` class that represents the calibration dataset. The `nncf.Dataset` class can be a wrapper over the framework dataset object that is used for model training or validation. The class constructor receives the dataset object and the transformation function. For example, if you use PyTorch, you can pass an instance of the `torch.utils.data.DataLoader` object. 

The transformation function is a function that takes a sample from the dataset and returns data that can be passed to the model for inference. For example, this function can take a tuple of a data tensor and labels tensor, and return the former while ignoring the latter. The transformation function is used to avoid modifying the dataset code to make it compatible with the quantization API. The function is applied to each sample from the dataset before passing it to the model for inference. The following code snippet shows how to create an instance of the `nncf.Dataset` class:

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_torch.py dataset

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_onnx.py dataset

@endsphinxtab

@sphinxtab{OpenVINO}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_openvino.py dataset

@endsphinxtab

@sphinxtab{TensorFlow}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_tensorflow.py dataset

@endsphinxtab

@endsphinxtabset

If there is no framework dataset object, you can create your own entity that implements the `Iterable` interface in Python and returns data samples feasible for inference. In this case, a transformation function is not required.


## Run a Quantized Model

Once the dataset is ready and the model object is instantiated, you can apply 8-bit quantization to it:
@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_torch.py quantization

@endsphinxtab

@sphinxtab{ONNX}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_torch.py quantization

@endsphinxtab

@sphinxtab{OpenVINO}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_torch.py quantization

@endsphinxtab

@sphinxtab{TensorFlow}

@snippet docs/optimization_guide/nncf/ptq/code/ptq_tensorflow.py quantization

@endsphinxtab

@endsphinxtabset

> **NOTE**: The `model` is an instance of the `torch.nn.Module` class for PyTorch, `onnx.ModelProto` for ONNX, and `openvino.runtime.Model` for OpenVINO.

After that the model can be exported into th OpenVINO Intermediate Representation if needed and run faster with OpenVINO.

## Tune quantization parameters

`nncf.quantize()` function has several parameters that allow to tune quantization process to get more accurate model. Below is the list of parameters and their description:
* `model_type` - used to specify quantization scheme required for specific type of the model. For example, **Transformer** models (BERT, distillBERT, etc.) require a special quantization scheme to preserve accuracy after quantization.
  ```python
  nncf.quantize(model, dataset, model_type=nncf.ModelType.Transformer)
  ```
* `preset` - defines quantization scheme for the model. Two types of presets are available:
  * `PERFORMANCE` (default) - defines symmetric quantization of weigths and activations
  * `MIXED` - weights are quantized with symmetric quantization and the activations are quantized with asymmetric quantization. This preset is recommended for models with non-ReLU and asymmetric activation funstions, e.g. ELU, PReLU, GELU, etc.
  ```python
    nncf.quantize(model, dataset, preset=nncf.Preset.MIXED)
  ```
* `fast_bias_correction` - enables more accurate bias (error) correction algorithm that can be used to improve accuracy of the model. This parameter is available only for OpenVINO representation. `True` is used by default.
  ```python
    nncf.quantize(model, dataset, fast_bias_correction=False)
  ```
* `subset_size` - defines the number of samples from the calibration dataset that will be used to estimate quantization parameters of activations. The default value is 300.
  ```python
    nncf.quantize(model, dataset, subset_size=1000)
  ```
* `ignored_scope` - this parameter can be used to exclude some layers from quantization process. For example, if you want to exclude the last layer of the model from quantization. Below are some examples of how to use this parameter:
  * Exclude by layer name:
    ```python
    names = ['layer_1', 'layer_2', 'layer_3']
    nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(names=names))
    ```
  * Exclude by layer type:
    ```python
    types = ['Conv2d', 'Linear']
    nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(types=types))
    ```
  * Exclude by regular expression:
    ```python
    regex = '.*layer_.*'
    nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(patterns=regex))
    ```

If the accuracy of the quantized model is not satisfactory, you can try to use the [Quantization with accuracy control](@ref quantization_w_accuracy_control) flow.

## See also

* [Example of basic quantization flow in PyTorch](https://github.com/openvinotoolkit/nncf/tree/develop/examples/post_training_quantization/torch/mobilenet_v2)