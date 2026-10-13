---
sidebar_position: 3
sidebar_label: "Training-time Optimization"
---

# Training-time Optimization


Training-time optimization offered by NNCF is based on model compression algorithms executed
alongside the training process. This approach results in the optimal balance between lower
accuracy and higher performance, and better results than post-training quantization. It also
enables you to set the minimum acceptable accuracy value for your optimized model, determining
the optimization efficiency.

With a few lines of code, you can apply NNCF compression to a PyTorch training
script. Once the model is optimized, you may convert it to the
[OpenVINO IR format](../../../documentation/openvino-ir-format/index.md), getting even better
inference results with OpenVINO Runtime. To optimize your model, you will need:

- A PyTorch floating-point model.
- A training pipeline set up in the PyTorch framework.
- Training and validation datasets.
- A [JSON configuration file](https://github.com/openvinotoolkit/nncf/blob/develop/docs/ConfigFile.md)
  specifying which compression methods to use.

<div style={{textAlign: 'center'}}>

![](nncf_workflow.svg)

</div>

## Training-Time Compression Methods

### Quantization

Uniform 8-bit quantization, the method officially supported by NNCF, converts all weights and
activation values in a model from a high-precision format, such as 32-bit floating point, to a
lower-precision format, such as 8-bit integer. During training, it inserts into the model nodes
that simulate the effect of a lower precision. This way, the training algorithm considers
quantization errors part of the overall training loss and tries to minimize their impact.

To learn more, see:

- guide on quantization for [PyTorch](quantization-aware-training.md).
- Jupyter notebook on [Quantization Aware Training with NNCF and PyTorch](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-aware-training).

### Experimental methods

NNCF provides some state-of-the-art compression methods that are still in the experimental
stages of development and are only recommended for expert developers. These include:

- Mixed-precision quantization.
- Sparsity (check out the [Sparsity-Aware Training  notebook](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pytorch-quantization-sparsity-aware-training)).
- Movement Pruning (Movement Sparsity).

To learn [more about these methods](https://github.com/openvinotoolkit/nncf?tab=readme-ov-file#training-time-compression-algorithms),
see developer documentation of the NNCF repository.

## Additional Resources

- [Post-training quantization](../quantizing-models-post-training/index.md)
- [Model Optimization - NNCF](../../model-optimization/index.md)
- [NNCF GitHub repository](https://github.com/openvinotoolkit/nncf)
