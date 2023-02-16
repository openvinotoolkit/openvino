# Quantization-aware Training (QAT) {#qat_introduction}

## Introduction
Quantization-aware Training is a popular method that allows quantizing a model and applying fine-tuning to restore accuracy degradation caused by quantization. In fact, this is the most accurate quantization method. This document describes how to apply QAT from the Neural Network Compression Framework (NNCF) to get 8-bit quantized models. This assumes that you are knowledgeable in Python* programming and familiar with the training code for the model in the source DL framework.

## Using NNCF QAT
Here, we provide the steps that are required to integrate QAT from NNCF into the training script written with PyTorch or TensorFlow 2:

> **NOTE**: Currently, NNCF for TensorFlow 2 supports optimization of the models created using Keras [Sequesntial API](https://www.tensorflow.org/guide/keras/sequential_model) or [Functional API](https://www.tensorflow.org/guide/keras/functional).

### 1. Import NNCF API
In this step, you add NNCF-related imports in the beginning of the training script:

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py imports

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py imports

@endsphinxtab

@endsphinxtabset

### 2. Create NNCF configuration
Here, you should define NNCF configuration which consists of model-related parameters (`"input_info"` section) and parameters of optimization methods (`"compression"` section). For faster convergence, it is also recommended to register a dataset object specific to the DL framework. It will be used at the model creation step to initialize quantization parameters.

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py nncf_congig

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py nncf_congig

@endsphinxtab

@endsphinxtabset

### 3. Apply optimization methods
In the next step, you need to wrap the original model object with the `create_compressed_model()` API using the configuration defined in the previous step. This method returns a so-called compression controller and a wrapped model that can be used the same way as the original model. It is worth noting that optimization methods are applied at this step so that the model undergoes a set of corresponding transformations and can contain additional operations required for the optimization. In the case of QAT, the compression controller object is used for model export and, optionally, in distributed training as it will be shown below.

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py wrap_model

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py wrap_model

@endsphinxtab

@endsphinxtabset

### 4. Fine-tune the model
This step assumes that you will apply fine-tuning to the model the same way as it is done for the baseline model. In the case of QAT, it is required to train the model for a few epochs with a small learning rate, for example, 10e-5. In principle, you can skip this step which means that the post-training optimization will be applied to the model.

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py tune_model

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py tune_model

@endsphinxtab

@endsphinxtabset

### 5. Multi-GPU distributed training
In the case of distributed multi-GPU training (not DataParallel), you should call `compression_ctrl.distributed()` before the fine-tuning that will inform optimization methods to do some adjustments to function in the distributed mode.
@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py distributed

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py distributed

@endsphinxtab

@endsphinxtabset

### 6. Export quantized model
When fine-tuning finishes, the quantized model can be exported to the corresponding format for further inference: ONNX in the case of PyTorch and frozen graph - for TensorFlow 2.

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py export

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py export

@endsphinxtab

@endsphinxtabset

> **NOTE**: The precision of weigths gets INT8 only after the step of model conversion to OpenVINO Intermediate Representation. You can expect the model footprint reduction only for that format.
These were the basic steps to applying the QAT method from the NNCF. However, it is required in some cases to save/load model checkpoints during the training. Since NNCF wraps the original model with its own object it provides an API for these needs.

### 7. (Optional) Save checkpoint
To save model checkpoint use the following API:

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py save_checkpoint

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py save_checkpoint

@endsphinxtab

@endsphinxtabset

### 8. (Optional) Restore from checkpoint
To restore the model from checkpoint you should use the following API:

@sphinxtabset

@sphinxtab{PyTorch}

@snippet docs/optimization_guide/nncf/code/qat_torch.py load_checkpoint

@endsphinxtab

@sphinxtab{TensorFlow 2}

@snippet docs/optimization_guide/nncf/code/qat_tf.py load_checkpoint

@endsphinxtab

@endsphinxtabset

For more details on saving/loading checkpoints in the NNCF, see the following [documentation](https://github.com/openvinotoolkit/nncf/blob/develop/docs/Usage.md#saving-and-loading-compressed-models).

## Deploying quantized model
The quantized model can be deployed with OpenVINO in the same way as the baseline model. No extra steps or options are required in this case. For more details, see the corresponding [documentation](../../OV_Runtime_UG/openvino_intro.md).

## Examples
- [Quantizing PyTorch model with NNCF](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/302-pytorch-quantization-aware-training)
- [Quantizing TensorFlow model with NNCF](https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/305-tensorflow-quantization-aware-training)