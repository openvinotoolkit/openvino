# Intel® Distribution of OpenVINO™ Toolkit Developer Package 

> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the [software license agreements](https://software.intel.com/en-us/license/eula-for-intel-software-development-products) for the Software Package, which may also include notices, disclaimers, or
license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package for additional details.

## Introduction

OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel® hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.

**The Developer Package Includes the Following Components Installed by Default:**

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |  
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) | This tool imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine. <br>Popular frameworks include Caffe\*, TensorFlow\*, MXNet\*, and ONNX\*.                                               |
| Additional Tools                                   | A set of tools to work with your models including [Accuracy Checker utility](https://docs.openvinotoolkit.org/latest/omz_tools_accuracy_checker.html), [Post-Training Optimization Tool](https://docs.openvinotoolkit.org/latest/pot_README.html), [Benchmark Tool](../../inference-engine/samples/benchmark_app/README.md)                                    |

**The Runtime Package Includes the Following Components Installed by Dependency:**

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |  
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Inference Engine](https://pypi.org/project/openvino)               | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications.                                                                                                                                                                                                |


## System Requirements
The complete list of supported hardware is available in the [Release Notes](https://software.intel.com/content/www/us/en/develop/articles/openvino-relnotes.html#inpage-nav-8).

The table below lists the supported operating systems and Python* versions required to run the installation.

| Supported Operating System                                   | [Python* Version (64-bit)](https://www.python.org/) |
| :------------------------------------------------------------| :---------------------------------------------------|
|   Ubuntu* 18.04 long-term support (LTS), 64-bit              | 3.6, 3.7, 3.8                                       |
|   Ubuntu* 20.04 long-term support (LTS), 64-bit              | 3.6, 3.7, 3.8                                       |
|   Red Hat* Enterprise Linux* 8, 64-bit                       | 3.6, 3.8                                            |
|   CentOS* 7, 64-bit                                          | 3.6, 3.7, 3.8                                       |
|   macOS* 10.15.x versions                                    | 3.6, 3.7, 3.8                                       |
|   Windows 10*, 64-bit                                        | 3.6, 3.7, 3.8                                       |

> **NOTE**: This package can be installed on other versions of macOS, Linux and Windows, but only the specific versions above are fully validated.

## Install the Developer Package

### Step 1. Install External Software Dependencies

On Windows* OS you are required to install [Microsoft* Visual C++ Redistributable Package (x64)](https://visualstudio.microsoft.com/downloads/#microsoft-visual-c-redistributable-for-visual-studio-2019) to be able to run OpenVINO™ applications.

### Step 2. Set Up Python Virtual Environment

To avoid dependency conflicts, use a virtual environment. Skip this
   step only if you do want to install all dependencies globally.

Create virtual environment:

On Linux and macOS:
```sh
# Depending on your OS, this step may require installing python3-venv
python3 -m venv openvino_env
```

On Windows: 
```sh
python -m venv openvino_env
```

### Step 3. Activate Virtual Environment

On Linux and macOS:
```sh
source openvino_env/bin/activate
```
On Windows:
```sh
openvino_env\Scripts\activate
```

### Step 4. Set Up and Update pip to the Highest Version

Run the command below:
```sh
python -m pip install --upgrade pip
```

### Step 5. Install the Package

Run the command below: <br>

   ```sh
   pip install openvino-dev
   ```

### Step 6. Verify that the Package is Installed

Run the command below (this may take a few seconds):
```sh
pot -h
```

You will see the help message for Post-Training Optimization Tool if installation finished successfully.

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)
