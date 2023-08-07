# Low Precision Optimization Guide

## Introduction
This document provides the best-known methods on how to use low-precision capabilities of the OpenVINO™ toolkit to transform models
to more hardware-friendly representation using such methods as quantization.

Currently, these capabilities are represented by several components:
- Low-precision runtime
- Post-training Optimization Tool (POT)
- [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf)

The first two components are the part of OpenVINO toolkit itself while the latter one is a separate tool build on top of the PyTorch* framework 
and highly aligned with OpenVINO™.

This document covers high level aspects of model optimization flow in OpenVINO™.

## General Information

By low precision we imply the inference of Deep Learning models in the precision which is lower than 32 or 16 bits, such as *FLOAT32* and *FLOAT16*. For example, the most popular
bit-width for the low-precision inference is *INT8* (*UINT8*) because it is possible to get accurate 8-bit models which substantially speed up the inference. 
Such models are represented by the quantized models, i.e. the models that were trained in the floating-point precision and then transformed to integer 
representation with floating/fixed-point quantization operations between the layers. This transformation can be done using post-training methods or 
with additional retraining/fine-tuning. 

Starting from the OpenVINO 2020.1 release all the quantized models are represented using so-called `FakeQuantize` layer which is
a very expressive primitive and is able to represent such operations as `Quantize`, `Dequantize`, `Requantize`, and even more. This operation is
inserted into the model during quantization procedure and is aimed to store quantization parameters for the layers. For more details about this operation
please refer to the following [description](@ref openvino_docs_ops_quantization_FakeQuantize_1).

In order to execute such "fake-quantized" models, OpenVINO has a low-precision runtime which is a part of Inference Engine and consists of a 
generic component translating the model to real integer representation and HW-specific part implemented in the corresponding HW plug-ins. 

## Model Optimization Workflow
We propose a common workflow which aligns with what other DL frameworks have. It contains two main components: post-training quantization and Quantization-Aware Training (QAT). 
The first component is the easiest way to get optimized models where the latter one can be considered as an alternative or an addition when the first does not give
accurate results.

The diagram below shows the optimization flow for the new model with OpenVINO and relative tools.

![](images/low_precision_flow.png)

- **Step 0: Model enabling**. In this step we should ensure that the model trained on the target dataset can be successfully inferred with [OpenVINO™ Runtime](@ref openvino_docs_OV_UG_OV_Runtime_User_Guide) in floating-point precision.
This process involves use of [model conversion API](@ref openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide) tool to convert the model from the source framework 
to the OpenVINO Intermediate Representation (IR) and run it on CPU with Inference Engine. 
  > **NOTE**: This step presumes that the model has the same accuracy as in the original training framework and enabled in the [Accuracy Checker](@ref omz_tools_accuracy_checker) tool or through the custom validation sample.
- **Step 1: Post-training quantization**. As the first step for optimization, we suggest using INT8 quantization from POT where in most cases it is possible to get an accurate quantized model. At this step you do not need model re-training. The only thing required is a representative dataset which is usually several hundreds of images and it is used to collect statistics during the quantization process.
Post-training quantization is also really fast and usually takes several minutes depending on the model size and used HW. And, generally, a regular desktop system is enough to quantize most of [OpenVINO Model Zoo](https://github.com/opencv/open_model_zoo).
For more information on best practices of post-training optimization please refer to the [Post-training Optimization Best practices](BestPractices.md).
- **Step2: Quantization-Aware Training**: If the accuracy of the quantized model does not satisfy accuracy criteria, there is step two which implies QAT using [Neural Network Compression Framework (NNCF)](https://github.com/openvinotoolkit/nncf) for [PyTorch*](https://pytorch.org/) and [TensorFlow*](https://www.tensorflow.org/) models.
At this step, we assume the user has an original training pipeline of the model written on TensorFlow or PyTorch and NNCF is integrated into it.
After this step, you can get an accurate optimized model that can be converted to OpenVINO Intermediate Representation (IR) using model conversion API and inferred with OpenVINO Inference Engine.
