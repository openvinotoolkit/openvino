# Optimizing Models at Training Time {#tmo_introduction}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   qat_introduction
   filter_pruning

@endsphinxdirective

## Introduction
Training-time model optimization is a way to get a more efficient and HW-friendly model when applying optimization methods with fine-tuning. OpenVINO&trade; does not have training capabilities but it provides a Neural Network Compression Framework (NNCF) tool that can be used to integrate training-time optimizations supported by OpenVINO in the training scripts created using 3rd party frameworks, such as PyTorch or TensorFlow 2. 

To apply training-time optimization methods you need:
- A floating-point model in the framework representation.
- Training script written with framework API.
- Training and validation datasets.

Figure below shows a common workflow of applying training-time optimizations with NNCF.

![](../../img/nncf_workflow.png)

## Optimization with NNCF
NNCF provides two main optimization methods depending on the user’s needs and requirements:
- **Quantization-aware Training (QAT)** is a recommended method that provides fast and accurate results. Currently, a HW-compatible (CPU, GPU, VPU) QAT for 8-bit inference is available. For details, see [Quantization-aware Training](./qat.md) documentation.
- **Filter Pruning** is used to remove unnecessary or redundant filters from Convolutional Neural Networks. It is usually not used stand-alone but can be stacked with QAT to get additional speedup on top of it. For details, see [Filter Pruning](./filter_pruning.md) documentation.

## Installation
NNCF is open-sourced on [GitHub](https://github.com/openvinotoolkit/nncf) and distributed as a separate package. It is also available on PyPI. We recommend installing it to the Python* environment where the framework is installed.

### Install from PyPI
To install the latest released version via pip manager run the following command:
```
pip install nncf
```

### Install from source
To install the latest NNCF version from source follow the instruction on [GitHub](https://github.com/openvinotoolkit/nncf#installation).

> **NOTE**: NNCF does not have OpenVINO&trade; as an installation requirement. To deploy optimized models you should install OpenVINO&trade; separately.


## See also
- [Post-training Optimization](@ref pot_introduction)