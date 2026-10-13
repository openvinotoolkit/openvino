---
sidebar_position: 2
sidebar_label: "Quantizing with Accuracy Control"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quantizing with Accuracy Control

## Introduction

This is the advanced quantization flow that allows to apply 8-bit quantization to the model with control of accuracy metric. This is achieved by keeping the most impactful operations within the model in the original precision. The flow is based on the [Basic 8-bit quantization](basic-quantization-flow.md) and has the following differences:

- Besides the calibration dataset, a **validation dataset** is required to compute the accuracy metric. Both datasets can refer to the same data in the simplest case.
- **Validation function**, used to compute accuracy metric is required. It can be a function that is already available in the source framework or a custom function.
- Since accuracy validation is run several times during the quantization process, quantization with accuracy control can take more time than the [Basic 8-bit quantization](basic-quantization-flow.md) flow.
- The resulted model can provide smaller performance improvement than the [Basic 8-bit quantization](basic-quantization-flow.md) flow because some of the operations are kept in the original precision.

:::note
Currently, 8-bit quantization with accuracy control is available only for models in OpenVINO and onnx.ModelProto representation.
:::

The steps for the quantization with accuracy control are described below.

## Prepare model

When working with an original model in FP32 precision, it is recommended to use the model as-is, without compressing weights, as the input for the quantization method with accuracy control. This ensures optimal performance relative to a given accuracy drop. Utilizing compression techniques, such as compressing the original model weights to FP16, may significantly increase the number of reverted layers and lead to reduced performance for the quantized model.
If the original model is converted to OpenVINO and saved through `openvino.save_model()` before using it in the quantization method with accuracy control, disable the compression of weights to FP16 by setting `compress_to_fp16=False`. This is necessary because, by default, `openvino.save_model()` saves models in FP16.

## Prepare calibration and validation datasets

This step is similar to the [Basic 8-bit quantization](basic-quantization-flow.md) flow. The only difference is that two datasets, calibration and validation, are required.

<Tabs>
  <TabItem value="openvino" label="OpenVINO">
  ```python
  import nncf
  import torch

  calibration_loader = torch.utils.data.DataLoader(...)

  def transform_fn(data_item):
      images, _ = data_item
      return images

  calibration_dataset = nncf.Dataset(calibration_loader, transform_fn)
  validation_dataset = nncf.Dataset(calibration_loader, transform_fn)
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
  validation_dataset = nncf.Dataset(calibration_loader, transform_fn)
  ```
  </TabItem>
</Tabs>

## Prepare validation function

The validation function takes two arguments: a model object and a validation dataset, and it returns the accuracy metric value. The type of the model object varies for different frameworks. In OpenVINO, it is an `openvino.CompiledModel`. In ONNX, it is an `onnx.ModelProto`.
The following code snippet shows an example of a validation function for OpenVINO and ONNX framework:

<Tabs>
  <TabItem value="openvino" label="OpenVINO">
  ```python
  import numpy as np
  import torch
  from sklearn.metrics import accuracy_score

  import openvino


  def validate(model: openvino.CompiledModel,
               validation_loader: torch.utils.data.DataLoader) -> float:
      predictions = []
      references = []

      output = model.outputs[0]

      for images, target in validation_loader:
          pred = model(images)[output]
          predictions.append(np.argmax(pred, axis=1))
          references.append(target)

      predictions = np.concatenate(predictions, axis=0)
      references = np.concatenate(references, axis=0)
      return accuracy_score(predictions, references)
  ```
  </TabItem>
  <TabItem value="onnx" label="ONNX">
  ```python
  import numpy as np
  import torch
  from sklearn.metrics import accuracy_score

  import onnx
  import onnxruntime


  def validate(model: onnx.ModelProto,
               validation_loader: torch.utils.data.DataLoader) -> float:
      predictions = []
      references = []

      input_name = model.graph.input[0].name
      serialized_model = model.SerializeToString()
      session = onnxruntime.InferenceSession(serialized_model, providers=["CPUExecutionProvider"])
      output_names = [output.name for output in session.get_outputs()]

      for images, target in validation_loader:
          pred = session.run(output_names, input_feed={input_name: images.numpy()})[0]
          predictions.append(np.argmax(pred, axis=1))
          references.append(target)

      predictions = np.concatenate(predictions, axis=0)
      references = np.concatenate(references, axis=0)
      return accuracy_score(predictions, references)
  ```
  </TabItem>
</Tabs>

## Run quantization with accuracy control

`nncf.quantize_with_accuracy_control()` function is used to run the quantization with accuracy control. The following code snippet shows an example of quantization with accuracy control for OpenVINO and ONNX framework:

<Tabs>
  <TabItem value="openvino" label="OpenVINO">
  ```python
  model = ... # openvino.Model object

  quantized_model = nncf.quantize_with_accuracy_control(
      model,
      calibration_dataset=calibration_dataset,
      validation_dataset=validation_dataset,
      validation_fn=validate,
      max_drop=0.01,
      drop_type=nncf.DropType.ABSOLUTE,
  )
  ```
  </TabItem>
  <TabItem value="onnx" label="ONNX">
  ```python
  import onnx

  model = onnx.load("model_path")

  quantized_model = nncf.quantize_with_accuracy_control(
      model,
      calibration_dataset=calibration_dataset,
      validation_dataset=validation_dataset,
      validation_fn=validate,
      max_drop=0.01,
      drop_type=nncf.DropType.ABSOLUTE,
  )
  ```
  </TabItem>
</Tabs>

- `max_drop` defines the accuracy drop threshold. The quantization process stops when the degradation of accuracy metric on the validation dataset is less than the `max_drop`. The default value is 0.01. NNCF will stop the quantization and report an error if the `max_drop` value can't be reached.
- `drop_type` defines how the accuracy drop will be calculated: `ABSOLUTE` (used by default) or `RELATIVE`.

After that the model can be compiled and run with OpenVINO:

<Tabs>
  <TabItem value="openvino" label="OpenVINO">
  ```python
  import openvino as ov

  # compile the model to transform quantized operations to int8
  model_int8 = ov.compile_model(quantized_model)

  input_fp32 = ... # FP32 model input
  res = model_int8(input_fp32)
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
  ```
  </TabItem>
</Tabs>

To save the model in the OpenVINO Intermediate Representation (IR), use `openvino.save_model()`. When dealing with an original model in FP32 precision, it's advisable to preserve FP32 precision in the most impactful model operations that were reverted from INT8 to FP32. To do this, consider using compress\_to\_fp16=False during the saving process. This recommendation is based on the default functionality of `openvino.save_model()`, which saves models in FP16, potentially impacting accuracy through this conversion.

<Tabs>
  <TabItem value="openvino" label="OpenVINO">
  ```python
  # save the model with compress_to_fp16=False to avoid an accuracy drop from compression
  # of unquantized weights to FP16. This is necessary because
  # nncf.quantize_with_accuracy_control(...) keeps the most impactful operations within
  # the model in the original precision to achieve the specified model accuracy
  ov.save_model(quantized_model, "quantized_model.xml", compress_to_fp16=False)
  ```
  </TabItem>
</Tabs>

`nncf.quantize_with_accuracy_control()` API supports all the parameters from [Basic 8-bit quantization](basic-quantization-flow.md) API, to quantize a model with accuracy control and a custom configuration.

If the accuracy or performance of the quantized model is not satisfactory, you can try [Training-time Optimization](../compressing-models-during-training/index.md) as the next step.

## Examples of NNCF post-training quantization with control of accuracy metric:

- [Post-Training Quantization of Anomaly Classification OpenVINO model with control of accuracy metric](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/anomaly_stfpm_quantize_with_accuracy_control)
- [Post-Training Quantization of YOLOv8 OpenVINO Model with control of accuracy metric](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/openvino/yolov8_quantize_with_accuracy_control)
- [Post-Training Quantization of YOLOv8 ONNX Model with control of accuracy metric](https://github.com/openvinotoolkit/nncf/blob/develop/examples/post_training_quantization/onnx/yolov8_quantize_with_accuracy_control)

## See also

- [Optimizing Models at Training Time](../compressing-models-during-training/index.md)
