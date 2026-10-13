---
sidebar_position: 1
sidebar_label: "Quantization-aware Training (QAT)"
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quantization-aware Training (QAT)

## Introduction

Quantization-aware Training is a popular method that allows quantizing a model and applying fine-tuning to restore accuracy
degradation caused by quantization. In fact, this is the most accurate quantization method. This document describes how to
apply QAT from the Neural Network Compression Framework (NNCF) to get 8-bit quantized models. This assumes that you are
knowledgeable in Python programming and familiar with the training code for the model in the source DL framework.

Steps required to apply QAT to the model:

## 1. Apply Post Training Quantization to the Model

Quantize the model using the [Post-Training Quantization](../quantizing-models-post-training/basic-quantization-flow.md) method.

<Tabs>
  <TabItem value="pytorch" label="PyTorch">
  ```python
  model = TorchModel() # instance of torch.nn.Module
  quantized_model = nncf.quantize(model, ...)
  ```
  </TabItem>
</Tabs>

## 2. Fine-tune the Model

This step assumes applying fine-tuning to the model the same way it is done for the
baseline model. For QAT, it is required to train the model for a few epochs with a small
learning rate, for example, 1e-5. Quantized models perform all computations in the
floating-point precision during fine-tuning by modeling quantization errors in both
forward and backward passes.

<Tabs>
  <TabItem value="pytorch" label="PyTorch">
  ```python
  ... # fine-tuning preparations, e.g. dataset, loss, optimization setup, etc.

  # tune quantized model for 5 epochs as the baseline
  for epoch in range(0, 5):
      for i, data in enumerate(train_loader):
          ... # training loop body
  ```
  </TabItem>
</Tabs>

:::note
The precision of weight transitions to INT8 only after converting the model to OpenVINO
Intermediate Representation. You can expect a reduction in the model footprint only for
that format.
:::

These steps outline the basics of applying the QAT method from the NNCF. However, in some cases, it is required to save/load model
checkpoints during training. Since NNCF wraps the original model with its own object, it provides an API for these needs.

## 3. (Optional) Save Checkpoint

To save a model checkpoint, use the following API:

<Tabs>
  <TabItem value="pytorch" label="PyTorch">
  ```python
  checkpoint = {
      'state_dict': model.state_dict(),
      'nncf_config': model.nncf.get_config(),
      ... # the rest of the user-defined objects to save
  }
  torch.save(checkpoint, path_to_checkpoint)
  ```
  </TabItem>
</Tabs>

## 4. (Optional) Restore from Checkpoint

To restore the model from checkpoint, use the following API:

<Tabs>
  <TabItem value="pytorch" label="PyTorch">
  ```python
  resuming_checkpoint = torch.load(path_to_checkpoint)
  nncf_config = resuming_checkpoint['nncf_config']
  quantized_model = nncf.torch.load_from_config(model, nncf_config, example_input)
  state_dict = resuming_checkpoint['state_dict']
  model.load_state_dict(state_dict)
  ```
  </TabItem>
</Tabs>

## Deploying the Quantized Model

You can convert the model to OpenVINO IR, if needed, compile it and run with OpenVINO without
any additional steps.

<Tabs>
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
</Tabs>

For more details, see the corresponding [documentation](../../running-inference/index.md).

## Examples

- [Quantization-aware Training of Resnet18 PyTorch Model](https://github.com/openvinotoolkit/nncf/tree/develop/examples/quantization_aware_training/torch/resnet18)
- [Quantization-aware Training of STFPM PyTorch Model](https://github.com/openvinotoolkit/nncf/tree/develop/examples/quantization_aware_training/torch/anomalib)
