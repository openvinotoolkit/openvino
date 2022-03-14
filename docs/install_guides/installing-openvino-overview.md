# Overview {#openvino_docs_install_guides_overview}

Intel® Distribution of OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud by:

* Enabling CNN-based deep learning inference on the edge.
* Supporting heterogeneous execution across Intel® CPU, Intel® Integrated Graphics, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.
* Speeding time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels.

## Installation Options

### Decide What to Install

**If you have already finished your model development and want to deploy your applications on various devices, [install OpenVINO Runtime](installing-openvino-runtime.md)**, which contains a set of libraries for an easy inference integration into your applications and supports heterogeneous execution across Intel® CPU and Intel® GPU hardware.

**If you want to download model from [Open Model Zoo](../model_zoo.md), convert to [OpenVINO IR](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md), [optimize](../optimization_guide/model_optimization_guide.md) and tune pre-trained deep learning models**, [install OpenVINO Development Tools](installing-model-dev-tools.md), which provides the following tools:

  * Model Optimizer
  * Post-Training Optimization Tool
  * Benchmark Tool
  * Accuracy Checker and Annotation Converter
  * Model Downloader and other Open Model Zoo tools


### Choose Your Installation Method

For Python developers, you can [install OpenVINO from PyPI](installing-openvino-pip.md), which contains both OpenVINO Python Runtime and Development Tools and less steps.

For C++ developers, you may choose one of the following installation options to install OpenVINO Runtime on your specific operating system:

* Linux: You can install OpenVINO Runtime using an [Installer](installing-openvino-linux.md), [APT](installing-openvino-apt.md), [YUM](installing-openvino-yum.md), [Anaconda Cloud](installing-openvino-conda.md) or [Docker](installing-openvino-docker-linux.md).
* Windows: You can install OpenVINO Runtime using an [Installer](installing-openvino-windows.md), [Anaconda Cloud](installing-openvino-conda.md) or [Docker](installing-openvino-docker-windows.md).
* macOS: You can install OpenVINO Runtime using an [Installer](installing-openvino-macos.md) or [Anaconda Cloud](installing-openvino-conda.md).
* [Raspbian OS](installing-openvino-raspbian.md).

> **NOTE**: From the 2022.1 release, the OpenVINO Development Tools can **only** be installed via PyPI. See [Install OpenVINO Development Tools](installing-model-dev-tools.md) for detailed steps.

Besides, the open source version is also available in the [OpenVINO™ toolkit GitHub repository](https://github.com/openvinotoolkit/openvino/). You can build it for supported platforms using the [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).
