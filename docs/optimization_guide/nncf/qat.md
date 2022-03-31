# Quantization-aware Training (QAT) {#qat_introduction}

## Introduction
Quantization-aware Training is a popular method that allows quantizing a model and apply fine-tuning to restore accuracy degradation caused by quantization. In fact, this is the most accurate quantization method. This document describes how to apply QAT from the Neural Network Compression Framework (NNCF) to get 8-bit quantized models. This assumes that you are knowledgable in Python* programming and familiar with the training code for the model in the source DL framework.

> **NOTE**: Currently NNCF for TensorFlow 2 supports optimization of the models created using Keras [Sequesntial API](https://www.tensorflow.org/guide/keras/sequential_model).

## Using NNCF QAT
Here, we provide the steps that are required to integrate QAT from NNCF into the training script written with PyTorch or TensorFlow 2:

### 1. Import NNCF API
In this step you add NNCF-related imports in the beggining of the file:

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py imports

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py imports

@endsphinxtab

@endsphinxtabset

### 2. Create NNCF configuration
Here, you should define NNCF confiration which consists of model-related parameters (`"input_info"` section) and parameters optimization methods (`"compression"` section). For faster convergions, it is also recommended to register a dataset object specific to the using DL framework. It will be used at the model creation step to initialize quantization parameters.

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py nncf_congig

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py nncf_congig

@endsphinxtab

@endsphinxtabset



## Deploying optimized model


