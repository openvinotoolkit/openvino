# Overview

Intel® Distribution of OpenVINO™ Toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud by:

* Enabling CNN-based deep learning inference on the edge.
* Supporting heterogeneous execution across Intel® CPU, Intel® Integrated Graphics, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs.
* Speeding time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels.

## Installation Options

You can install OpenVINO on major operating systems. For Python developers, you can [Install OpenVINO from PyPI](../installing-openvino-pip.md), which contains less steps and is much simpler.

For C++ developers, you may choose your installation method according to your needs:

* **If you have already finished your model development and want to deploy your applications on various devices, install OpenVINO Runtime**, which contains the Inference Engine to run deep learning models. Inference Engine includes a set of libraries for an easy inference integration into your applications.
  * You can use one of the following ways to install OpenVINO Runtime:
    * Installer for [Linux](../installing-openvino-linux.md), [Windows](../installing-openvino-windows.md) or [macOS](../installing-openvino-macos.md)
    * [APT for Linux](../installing-openvino-apt.md)
    * [YUM for Linux](../installing-openvino-yum.md)
    * Docker for [Linux](../installing-openvino-docker-linux.md) and [Windows](../installing-openvino-docker-windows.md)
    * Install on [Raspbian OS](../installing-openvino-raspbian.md)
  * You can also install OpenVINO Runtime from [Anaconda Cloud](../installing-openvino-conda.md).

* **If you want to develop or optimize your models with OpenVINO and deploy your applications after that, install [OpenVINO Model Development Tools](../installing-model-dev-tools.md)**, which provides the following tools:
  * Model Optimizer
  * Benchmark Tool
  * Accuracy Checker and Annotation Converter
  * Post-Training Optimization Tool
  * Model Downloader and other Open Model Zoo tools
  
  > **NOTE**: From 2022.1 release, the OpenVINO Model Development Tools can only be installed via PyPI. See [Install OpenVINO Model Development Tools](../installing-model-dev-tools.md) for detailed steps.

* The open source version is available in the OpenVINO™ toolkit GitHub repository and you can build it for supported platforms using the Inference Engine Build Instructions.
