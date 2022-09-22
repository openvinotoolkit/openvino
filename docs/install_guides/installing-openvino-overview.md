# Installing Intel® Distribution of OpenVINO™ Toolkit {#openvino_docs_install_guides_overview}

@sphinxdirective

.. toctree::
   :maxdepth: 3
   :hidden:
   
   OpenVINO Runtime <openvino_docs_install_guides_install_runtime>
   OpenVINO Development Tools <openvino_docs_install_guides_install_dev_tools>
   Build from Source <https://github.com/openvinotoolkit/openvino/wiki/BuildingCode>
   Creating a Yocto Image <openvino_docs_install_guides_installing_openvino_yocto>

@endsphinxdirective

Intel® Distribution of OpenVINO™ toolkit is a comprehensive toolkit for developing applications and solutions based on deep learning tasks, such as emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and more. It provides high-performance and rich deployment options, from edge to cloud. Some of its advantages are:

* Enables CNN-based and transformer-based deep learning inference on the edge or cloud.
* Supports various execution modes across Intel® technologies: Intel® CPU, Intel® Integrated Graphics, Intel® Discrete Graphics, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.
* Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels.
* Compatible with models from a wide variety of frameworks, including TensorFlow, PyTorch, PaddlePaddle, ONNX, and more.

## Install OpenVINO

The OpenVINO installation package is distributed in two parts: OpenVINO Runtime and OpenVINO Development Tools.

* **OpenVINO Runtime** contains the core set of libraries for running machine learning model inference on processor devices.
* **OpenVINO Development Tools** is a set of utilities for working with OpenVINO and OpenVINO models. It includes the following tools:
  - Model Optimizer
  - Post-Training Optimization Tool
  - Benchmark Tool
  - Accuracy Checker and Annotation Converter
  - Model Downloader and other Open Model Zoo tools

### Option 1. Install OpenVINO Development Tools (recommended)

The best way to get started with OpenVINO is to install OpenVINO Development Tools, which will install the development tools in only a few steps. It also installs the OpenVINO Runtime Python package as a dependency. Follow the instructions on the [Install OpenVINO Development Tools](installing-model-dev-tools.md) page to install it.

**Python** <br>
For developers working in Python, OpenVINO Development Tools can easily be installed using PyPI. See the [For Python Developers](installing-model-dev-tools.md#for-python-developers) section of the Install OpenVINO Development Tools page for instructions.

**C++** <br>
For developers working in C++, the core OpenVINO Runtime libraries must be installed separately. Then, OpenVINO Development Tools can be installed using requirements files or PyPI. See the [For C++ Developers](installing-model-dev-tools.md#for-c-developers) section of the Install OpenVINO Development Tools page for instructions.

### Option 2. Install OpenVINO Runtime only

The OpenVINO Runtime may also be installed on its own without OpenVINO Development Tools. This is recommended for users who already have an optimized model and want to deploy it in an application that uses OpenVINO for inference on their device. To install OpenVINO Runtime only, follow the instructions on the [Install OpenVINO Runtime](installing-openvino-runtime.md) page.

The following methods are available to install OpenVINO Runtime:

* Linux: You can install OpenVINO Runtime using archive files or Docker. See [Install OpenVINO on Linux](installing-openvino-linux-header.md).
* Windows: You can install OpenVINO Runtime using archive files or Docker. See [Install OpenVINO on Windows](installing-openvino-windows-header.md).
* macOS: You can install OpenVINO Runtime using archive files or Docker. See [Install OpenVINO on macOS](installing-openvino-macos-header.md).
* [Rasbpian OS](installing-openvino-raspbian.md)

### Option 3. Build OpenVINO from source

Source files are also available in the OpenVINO Toolkit GitHub repository. If you want to build OpenVINO from source for your platform, follow the [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

## Next Steps
Still unsure if you want to install OpenVINO Toolkit? Check out the [OpenVINO tutorial notebooks](../tutorials.md) to run example applications directly in your web browser without needing to install it locally. Here are some exciting demos you can explore:
- [Monodepth Estimation with OpenVINO](https://docs.openvino.ai/latest/notebooks/201-vision-monodepth-with-output.html)
- [Style Transfer on ONNX Models with OpenVINO](https://docs.openvino.ai/latest/notebooks/212-onnx-style-transfer-with-output.html)
- [OpenVINO API Tutorial](https://docs.openvino.ai/latest/notebooks/002-openvino-api-with-output.html)

Follow these links to install OpenVINO:
- [Install OpenVINO Development Tools](installing-model-dev-tools.md)
- [Install OpenVINO Runtime](installing-openvino-runtime.md)
- [Build from Source](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode)
