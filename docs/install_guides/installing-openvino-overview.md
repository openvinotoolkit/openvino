# Installing Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_overview}

@sphinxdirective

.. toctree::
   :maxdepth: 3
   :hidden:
   
   OpenVINO Runtime 2022.1 <openvino_docs_install_guides_install_runtime>
   OpenVINO Runtime 2022.1.1 <openvino_docs_install_guides_install_runtime_2022_1_1>
   OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>
   Build from Source <https://github.com/openvinotoolkit/openvino/wiki/BuildingCode>
   Creating a Yocto Image <openvino_docs_install_guides_installing_openvino_yocto>

@endsphinxdirective

Intel® Distribution of OpenVINO™ toolkit is a comprehensive toolkit for developing applications and solutions based on deep learning tasks, such as emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and more. It provides high-performance and rich deployment options from edge to cloud. Some of its advantages are:

* Enables CNN-based and transformer-based deep learning inference on the edge or cloud.
* Supports various execution modes across Intel® technologies: Intel® CPU, Intel® Integrated Graphics, Intel® Discrete Graphics, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.
* Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels.
* Compatible with models from a wide variety of frameworks, including TensorFlow, PyTorch, PaddlePaddle, ONNX, and more.

## Install OpenVINO

The OpenVINO installation package is distributed in two parts: OpenVINO Runtime and OpenVINO Development Tools.

* OpenVINO Runtime contains the core set of libraries for running machine learning model inference on processor devices.
* OpenVINO Development Tools is a set of utilities for working with OpenVINO and OpenVINO models. It includes the following tools:
  - Model Optimizer
  - Post-Training Optimization Tool
  - Benchmark Tool
  - Accuracy Checker and Annotation Converter
  - Model Downloader and other Open Model Zoo tools

### Option 1. Install OpenVINO Development Tools (recommended)

The best way to get started with OpenVINO is to install OpenVINO Development Tools, which will install the development tools in only a few steps. It also installs the OpenVINO Runtime Python package as a dependency. Follow the instructions on the [Install OpenVINO Development Tools](installing-model-dev-tools.md) page to install it.

**Python** <br>
For developers working in Python, OpenVINO Development Tools (which includes OpenVINO Runtime) can easily be installed using PyPI. See the [For Python Developers](installing-model-dev-tools.md#for-python-developers) section of the Install OpenVINO Development Tools page for instructions.

**C++** <br>
For developers working in C++, the core OpenVINO Runtime libraries must be installed separately. Then, OpenVINO Development Tools can be installed using requirements files or PyPI. See the [For C++ Developers](installing-model-dev-tools.md#for-c-developers) section of the Install OpenVINO Development Tools page for instructions.

**If you want to download models from [Open Model Zoo](../model_zoo.md), [convert your own models to OpenVINO IR](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md), or [optimize and tune pre-trained deep learning models](../optimization_guide/model_optimization_guide.md)**, [install OpenVINO Development Tools](installing-model-dev-tools.md), which provides the following tools:

  * Model Optimizer
  * Post-Training Optimization Tool
  * Benchmark Tool
  * Accuracy Checker and Annotation Converter
  * Model Downloader and other Open Model Zoo tools


### Choose Your Installation Method

For Python developers, the easiest way is to [install OpenVINO Development Tools](installing-model-dev-tools.md), which will install both OpenVINO Runtime and OpenVINO Development Tools with a few steps. If you want to install OpenVINO Runtime only, see [Install OpenVINO Runtime from PyPI](installing-openvino-pip.md).

For C++ developers, you may choose one of the following installation options for OpenVINO Runtime on your specific operating system:

* Linux: You can install OpenVINO Runtime using an [Installer](installing-openvino-linux.md), [APT](installing-openvino-apt.md), [YUM](installing-openvino-yum.md), [Anaconda Cloud](installing-openvino-conda.md), or [Docker](installing-openvino-docker-linux.md).
* Windows: You can install OpenVINO Runtime using an [Installer](installing-openvino-windows.md), [Anaconda Cloud](installing-openvino-conda.md), or [Docker](installing-openvino-docker-windows.md).
* macOS: You can install OpenVINO Runtime using an [Installer](installing-openvino-macos.md) or [Anaconda Cloud](installing-openvino-conda.md).
* [Raspbian OS](installing-openvino-raspbian.md).

> **NOTE**: With the introduction of the 2022.1 release, the OpenVINO Development Tools can be installed **only** via PyPI. See [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.
Source files are also available in the [OpenVINO toolkit GitHub repository](https://github.com/openvinotoolkit/openvino/), so you can build your own package for the supported platforms, as described in [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

## Next Steps

- [Install OpenVINO Runtime](installing-openvino-runtime.md)
- [Install OpenVINO Development Tools](installing-model-dev-tools.md)
- [Build from Source](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode)
- [Create a Yocto Image](installing-openvino-yocto.md)
