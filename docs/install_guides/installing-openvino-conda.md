# Install Intel® Distribution of OpenVINO™ toolkit from Anaconda* Cloud {#openvino_docs_install_guides_installing_openvino_conda}

This guide provides installation steps for Intel® Distribution of OpenVINO™ toolkit distributed through the Anaconda* Cloud.

> **NOTE**: Only runtime packages are available from Anaconda* Cloud.

## Introduction

OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel® hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.

The Intel® Distribution of OpenVINO™ toolkit\*:
- Enables CNN-based deep learning inference on the edge
- Supports heterogeneous execution across Intel® CPU, Intel® Integrated Graphics, Intel® Neural Compute Stick 2, and Intel® Vision Accelerator Design with Intel® Movidius™ VPUs
- Speeds time-to-market via an easy-to-use library of computer vision functions and pre-optimized kernels
The **runtime package** includes the following components installed by default:

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |  
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Inference Engine](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md)                            | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications.                                                                                                                                                                |

## System Requirements

**Software**

 - [Anaconda* distribution](https://www.anaconda.com/products/individual/)

**Operating Systems**

| Supported Operating System                                   | [Python* Version (64-bit)](https://www.python.org/) |
| :------------------------------------------------------------| :---------------------------------------------------|
|   Ubuntu* 18.04 long-term support (LTS), 64-bit              | 3.6, 3.7                                            |
|   Ubuntu* 20.04 long-term support (LTS), 64-bit              | 3.6, 3.7                                            |
|   CentOS* 7.6, 64-bit                                        | 3.6, 3.7                                            |
|   macOS* 10.15.x                                             | 3.6, 3.7                                            |
|   Windows 10*, 64-bit                                        | 3.6, 3.7                                            |

## Install the Runtime Package using the Anaconda* Package Manager

1. Set up the Anaconda* environment: 
   ```sh
   conda create --name py37 python=3.7
   ```
   ```sh
   conda activate py37
   ```
2. Update Anaconda environment to the latest version:
   ```sh
   conda update --all
   ```
3. Install pre-requisites:
    ```sh
   conda install numpy
   ```
4. Install the Intel® Distribution of OpenVINO™ Toolkit:
 - Ubuntu* 20.04 
   ```sh
   conda install openvino-ie4py-ubuntu20 -c intel
   ```
 - Ubuntu* 18.04 
   ```sh
   conda install openvino-ie4py-ubuntu18 -c intel
   ```
 - CentOS* 7.6 
   ```sh
   conda install openvino-ie4py-centos7 -c intel
   ```
 - Windows* 10 and macOS*
   ```sh
   conda install openvino-ie4py -c intel
   ```
5. Verify the package is installed:
   ```sh
   python -c "from openvino.inference_engine import IECore"
   ```
   If installation was successful, you will not see any error messages (no console output).

Now you can start developing your application.

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit).
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org).
- [Model Optimizer Developer Guide](../MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
- [Inference Engine Developer Guide](../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md).
- For more information on Sample Applications, see the [Inference Engine Samples Overview](../IE_DG/Samples_Overview.md).
- Intel® Distribution of OpenVINO™ toolkit Anaconda* home page: [https://anaconda.org/intel/openvino-ie4py](https://anaconda.org/intel/openvino-ie4py)

