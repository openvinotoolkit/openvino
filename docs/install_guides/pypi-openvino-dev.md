# Intel® Distribution of OpenVINO™ Toolkit Developer Package 

> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the [software license agreements](https://software.intel.com/en-us/license/eula-for-intel-software-development-products) for the Software Package, which may also include notices, disclaimers, or
license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package for additional details.

## Introduction

OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel® hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.

**Included with the Installation and installed by default:**

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |  
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) | This tool imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine. <br>Popular frameworks include Caffe\*, TensorFlow\*, MXNet\*, and ONNX\*.                                                                              |
| Additional Tools                                   | A set of tools to work with your models including [Accuracy Checker utility](https://docs.openvinotoolkit.org/latest/omz_tools_accuracy_checker_README.html), [Post-Training Optimization Tool](https://docs.openvinotoolkit.org/latest/pot_README.html)  |

**Installed by dependency:**

| Component                                                                                           | Description                                                                                                                                                                                                                                                                                                   |  
|-----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Inference Engine](https://pypi.org/project/openvino)               | This is the engine that runs the deep learning model. It includes a set of libraries for an easy inference integration into your applications.                                                                                                                                                                |


## Supported Operating Systems

* [Python* distribution](https://www.python.org/) 3.6, 3.7, 3.8
* Operating Systems:
  - Ubuntu* 18.04 long-term support (LTS), 64-bit (python 3.6 or 3.7)
  - Ubuntu* 20.04 long-term support (LTS), 64-bit (python 3.6 or 3.7)
  - macOS* 10.15.x versions
  - Windows 10*, 64-bit Pro, Enterprise or Education (1607 Anniversary Update, Build 14393 or higher) editions
  - Windows Server* 2016 or higher
> NOTE: This package can be installed on many versions of Linux, but only the specific versions above are fully validated.

## Install the runtime or developer packages using the PyPI repository

### Step 1. Set up and update pip to the highest version

Run the command below:
```sh
python3 -m pip install --upgrade pip
```

### Step 2. Install the Intel® distribution of OpenVINO™ toolkit

Run the command below: <br>

   ```sh
   pip install openvino-dev
   ```

### Step 3. Verify that the package is installed

Run the command below:
```sh
python3 -c "pot -h"
```
   
Now you are ready to develop and run your application.

## Additional Resources

- Intel® Distribution of OpenVINO™ toolkit home page: [https://software.intel.com/en-us/openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit)
- OpenVINO™ toolkit online documentation: [https://docs.openvinotoolkit.org](https://docs.openvinotoolkit.org)


