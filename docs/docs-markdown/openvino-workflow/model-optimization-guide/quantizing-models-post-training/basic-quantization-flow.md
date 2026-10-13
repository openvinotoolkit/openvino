---
sidebar_position: 1
sidebar_label: "Basic Quantization Flow"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Basic Quantization Flow

## Introduction

The basic quantization flow is the simplest way to apply 8-bit quantization to the model. It is available for models in the following frameworks: OpenVINO, PyTorch, and ONNX. The basic quantization flow is based on the following steps:

- Set up an environment and install dependencies.
- Prepare a representative **calibration dataset** that is used to estimate quantization parameters of the activations within the model, for example, of 300 samples.
- Call the quantization API to apply 8-bit quantization to the model.

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

At this step, create an instance of the `nncf.Dataset` class that represents the calibration dataset. The `nncf.Dataset` class can be a wrapper over the framework dataset object that is used for model training or validation. The class constructor receives the dataset object and an optional transformation function.

The transformation function is a function that takes a sample from the dataset and returns data that can be passed to the model for inference. For example, this function can take a tuple of a data tensor and labels tensor, and return the former while ignoring the latter. The transformation function is used to avoid modifying the dataset code to make it compatible with the quantization API. The function is applied to each sample from the dataset before passing it to the model for inference. The following code snippet shows how to create an instance of the `nncf.Dataset` class:

<Tabs>
  <TabItem value="openvino" label="OpenVINO">
  ```python
  import nncf
  import torch

  calibration_loader = torch.utils.data.DataLoader(...)

  def transform_fn(data_item):
      images, _ = data_item
      return images.numpy()

  calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
  ```
  </TabItem>
  <TabItem value="pytorch" label="PyTorch">
  ```python
  import nncf
  import torch

  calibration_loader = torch.utils.data.DataLoader(...)

  def transform_fn(data_item):
      images, _ = data_item
      return images

  calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
  ```
  </TabItem>
  <TabItem value="onnx" label="ONNX">
  ```python
  import nncf
  import torch

  calibration_loader = torch.utils.data.DataLoader(...)

  def transform_fn(data_item):
      images, _ = data_item
      return {input_name: images.numpy()} # input_name should be taken from the model,
                                          # e.g. model.graph.input[0].name

  calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
  ```
  </TabItem>
  <TabItem value="torch-fx" label="TorchFX">
  ```python
  import nncf
  import torch

  calibration_loader = torch.utils.data.DataLoader(...)

  def transform_fn(data_item):
      images, _ = data_item
      return images

  calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
  ```
  </TabItem>
</Tabs>

If there is no framework dataset object, you can create your own entity that implements the `Iterable` interface in Python, for example the list of images, and returns data samples feasible for inference. In this case, a transformation function is not required.

## Quantize a Model

Once the dataset is ready and the model object is instantiated, you can apply 8-bit quantization to it.
See the [example section](#examples-of-how-to-apply-nncf-post-training-quantization) at the end of this document for examples for each framework.

<Tabs>
  <TabItem value="openvino" label="OpenVINO">
  ```python
  import openvino as ov
  model = ov.Core().read_model("model_path")

  quantized_model = nncf.quantize(model, calibration_dataset)
  ```
  </TabItem>
  <TabItem value="pytorch" label="PyTorch">
  ```python
  import torchvision
  model = torchvision.models.resnet50(pretrained=True)

  quantized_model = nncf.quantize(model, calibration_dataset)
  ```
  </TabItem>
  <TabItem value="onnx" label="ONNX">
  ```python
  import onnx
  model = onnx.load("model_path")

  quantized_model = nncf.quantize(model, calibration_dataset)
  ```
  </TabItem>
  <TabItem value="torch-fx" label="TorchFX">
  ```python
  import torchvision

  input_fp32 = torch.ones((1, 3, 224, 224)) # FP32 model input
  model = torchvision.models.resnet50(pretrained=True)

  exported_model = torch.export.export_for_training(model, args=(input_fp32,)).module()
  quantized_model = nncf.quantize(exported_model, calibration_dataset)
  ```
  </TabItem>
</Tabs>

After that the model can be converted into the OpenVINO Intermediate Representation (IR) if needed, compiled and run with OpenVINO.
If you have not already installed OpenVINO developer tools, install it with `pip install openvino`.

<Tabs>
  <TabItem value="openvino" label="OpenVINO">
  ```python
  # compile the model to transform quantized operations to int8
  model_int8 = ov.compile_model(quantized_model)

  input_fp32 = ... # FP32 model input
  res = model_int8(input_fp32)

  # save the model
  ov.save_model(quantized_model, "quantized_model.xml")
  ```
  </TabItem>
  <TabItem value="pytorch" label="PyTorch">
  ```python
  import openvino as ov

  input_fp32 = ... # FP32 model input

  # convert PyTorch model to OpenVINO model
  ov_quantized_model = ov.convert_model(quantized_model, example_input=input_fp32)

  # compile the model to transform quantized operations to int8
  model_int8 = ov.compile_model(ov_quantized_model)

  res = model_int8(input_fp32)

  # save the model
  ov.save_model(ov_quantized_model, "quantized_model.xml")
  ```
  </TabItem>
  <TabItem value="onnx" label="ONNX">
  ```python
  import openvino as ov

  # use a temporary file to convert ONNX model to OpenVINO model
  quantized_model_path = "quantized_model.onnx"
  onnx.save(quantized_model, quantized_model_path)

  ov_quantized_model = ov.convert_model(quantized_model_path)

  # compile the model to transform quantized operations to int8
  model_int8 = ov.compile_model(ov_quantized_model)

  input_fp32 = ... # FP32 model input
  res = model_int8(input_fp32)

  # save the model
  ov.save_model(ov_quantized_model, "quantized_model.xml")
  ```
  </TabItem>
</Tabs>

TorchFX models can utilize OpenVINO optimizations using [torch.compile(..., backend="openvino")](https://docs.openvino.ai/2026/openvino-workflow/torch-compile.html) functionality:

<Tabs>
  <TabItem value="torch-fx" label="TorchFX">
  ```python
  import openvino.torch

  input_fp32 = ... # FP32 model input

  # compile quantized model using torch.compile API
  compiled_model_int8 = torch.compile(quantized_model, backend="openvino")
  # OpenVINO backend compiles the model during the first call,
  # so the first call is expected to be slower than the following calls
  res = compiled_model_int8(input_fp32)

  # save the model
  exported_program = torch.export.export(quantized_model, args=(input_fp32,))
  torch.export.save(exported_program, 'exported_program.pt2')
  ```
  </TabItem>
</Tabs>

## Tune quantization parameters

`nncf.quantize()` function has several optional parameters that allow tuning the quantization process to get a more accurate model. Below is the list of parameters and their description:

- `model_type` - used to specify quantization scheme required for specific type of the model. `Transformer` is the only supported special quantization scheme to preserve accuracy after quantization of Transformer models (BERT, DistilBERT, etc.). `None` is default, i.e. no specific scheme is defined.
  
  ```bash
  nncf.quantize(model, dataset, model_type=nncf.ModelType.Transformer)
  ```
- `preset` - defines quantization scheme for the model. Two types of presets are available:
  
  - `PERFORMANCE` (default) - defines symmetric quantization of weights and activations
  - `MIXED` - weights are quantized with symmetric quantization and the activations are quantized with asymmetric quantization. This preset is recommended for models with non-ReLU and asymmetric activation functions, e.g. ELU, PReLU, GELU, etc.
    
    ```bash
    nncf.quantize(model, dataset, preset=nncf.QuantizationPreset.MIXED)
    ```
- `fast_bias_correction` - when set to `False`, enables a more accurate bias (error) correction algorithm that can be used to improve the accuracy of the model. This parameter is available only for OpenVINO and ONNX representations. `True` is used by default to minimize quantization time.
  
  ```bash
  nncf.quantize(model, dataset, fast_bias_correction=False)
  ```
- `subset_size` - defines the number of samples from the calibration dataset that will be used to estimate quantization parameters of activations. The default value is 300.
  
  ```bash
  nncf.quantize(model, dataset, subset_size=1000)
  ```
- `ignored_scope` - this parameter can be used to exclude some layers from the quantization process to preserve the model accuracy. For example, when you want to exclude the last layer of the model from quantization. Below are some examples of how to use this parameter:
  
  - Exclude by layer name:
    
    ```bash
    names = ['layer_1', 'layer_2', 'layer_3']
    nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(names=names))
    ```
  - Exclude by layer type:
    
    ```bash
    types = ['Conv2d', 'Linear']
    nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(types=types))
    ```
  - Exclude by regular expression:
    
    ```bash
    regex = '.*layer_.*'
    nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(patterns=regex))
    ```
  - Exclude by subgraphs:
    
    ```bash
    subgraph = nncf.Subgraph(inputs=['layer_1', 'layer_2'], outputs=['layer_3'])
    nncf.quantize(model, dataset, ignored_scope=nncf.IgnoredScope(subgraphs=[subgraph]))
    ```
    
    In this case, all nodes along all simple paths in the graph from input to output nodes will be excluded from the quantization process.
- `target_device` - defines the target device, the specificity of which will be taken into account during optimization. The following values are supported: `ANY` (default), `CPU`, `CPU_SPR`, `GPU`, and `NPU`.
  
  ```bash
  nncf.quantize(model, dataset, target_device=nncf.TargetDevice.CPU)
  ```
- `advanced_parameters` - used to specify advanced quantization parameters for fine-tuning the quantization algorithm. Defined by [nncf.quantization.advanced\_parameters](https://openvinotoolkit.github.io/nncf/autoapi/nncf/quantization/advanced_parameters/index.html) NNCF submodule. `None` is default.

If the accuracy of the quantized model is not satisfactory, you can try to use the [Quantization with accuracy control](quantizing-with-accuracy-control.md) flow.

## Examples of how to apply NNCF post-training quantization:

- [Post-Training Quantization of MobileNet v2 OpenVINO Model](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/mobilenet_v2)
- [Post-Training Quantization of YOLOv8 OpenVINO Model](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/yolov8)
- [Post-Training Quantization of MobileNet v2 PyTorch Model](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/torch/mobilenet_v2)
- [Post-Training Quantization of SSD PyTorch Model](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/torch/ssd300_vgg16)
- [Post-Training Quantization of MobileNet v2 ONNX Model](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/onnx/mobilenet_v2)
