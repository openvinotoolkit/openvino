# Intel® Distribution of OpenVINO™ Toolkit Developer Package 
Copyright © 2018-2021 Intel Corporation
> **LEGAL NOTICE**: Your use of this software and any required dependent software (the
“Software Package”) is subject to the terms and conditions of the [software license agreements](https://software.intel.com/content/dam/develop/external/us/en/documents/intel-openvino-license-agreements.pdf) for the Software Package, which may also include notices, disclaimers, or
license terms for third party or open source software included in or with the Software Package, and your use indicates your acceptance of all such terms. Please refer to the “third-party-programs.txt” or other similarly-named text file included with the Software Package for additional details.

>Intel is committed to the respect of human rights and avoiding complicity in human rights abuses, a policy reflected in the [Intel Global Human Rights Principles](https://www.intel.com/content/www/us/en/policy/policy-human-rights.html). Accordingly, by accessing the Intel material on this platform you agree that you will not use the material in a product or application that causes or contributes to a violation of an internationally recognized human right.


## Introduction

OpenVINO™ toolkit is a comprehensive toolkit for quickly developing applications and solutions that solve a variety of tasks including emulation of human vision, automatic speech recognition, natural language processing, recommendation systems, and many others. Based on latest generations of artificial neural networks, including Convolutional Neural Networks (CNNs), recurrent and attention-based networks, the toolkit extends computer vision and non-vision workloads across Intel® hardware, maximizing performance. It accelerates applications with high-performance, AI and deep learning inference deployed from edge to cloud.

The **developer package** includes the following components installed by default:

| Component        | Console Script                                                                   | Description                                                                                                                                                                                                                                                                                                   |  
|------------------|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Model Optimizer](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) | `mo` |**Model Optimizer** imports, converts, and optimizes models that were trained in popular frameworks to a format usable by Intel tools, especially the Inference Engine. <br>Popular frameworks include Caffe\*, TensorFlow\*, MXNet\*, and ONNX\*.                                               |
| [Benchmark Tool](https://docs.openvinotoolkit.org/latest/openvino_inference_engine_tools_benchmark_tool_README.html)| `benchmark_app` | **Benchmark Application** allows you to estimate deep learning inference performance on supported devices for synchronous and asynchronous modes.                                              |
| [Accuracy Checker](https://docs.openvinotoolkit.org/latest/omz_tools_accuracy_checker.html) and <br> [Annotation Converter](https://docs.openvinotoolkit.org/latest/omz_tools_accuracy_checker_annotation_converters.html) | `accuracy_check` <br> `convert_annotation` |**Accuracy Checker**  is a deep learning accuracy validation tool that allows you to collect accuracy metrics against popular datasets. The main advantages of the tool are the flexibility of configuration and an impressive set of supported datasets, preprocessing, postprocessing, and metrics. <br> **Annotation Converter** is a utility for offline conversion of datasets to the format suitable for metric evaluation used in Accuracy Checker.                                            |
| [Post-Training Optimization Tool](https://docs.openvinotoolkit.org/latest/pot_README.html)| `pot` |**Post-Training Optimization Tool** allows you to optimize trained models with advanced capabilities, such as quantization and low-precision optimizations, without the need to retrain or fine-tune models. Optimizations are also available through the [API](https://docs.openvinotoolkit.org/latest/pot_compression_api_README.html).                                            |
| [Model Downloader and other Open Model Zoo tools](https://docs.openvinotoolkit.org/latest/omz_tools_downloader.html)| `omz_downloader` <br> `omz_converter` <br> `omz_quantizer` <br> `omz_info_dumper`| **Model Downloader** is a tool for getting access to the collection of high-quality and extremely fast pre-trained deep learning [public](https://docs.openvinotoolkit.org/latest/omz_models_group_public.html) and [intel](https://docs.openvinotoolkit.org/latest/omz_models_group_intel.html)-trained models. Use these free pre-trained models instead of training your own models to speed up the development and production deployment process. The principle of the tool is as follows: it downloads model files from online sources and, if necessary, patches them with Model Optimizer to make them more usable. A number of additional tools are also provided to automate the process of working with downloaded models:<br> **Model Converter** is a tool for converting the models stored in a format other than the Intermediate Representation (IR) into that format using Model Optimizer. <br> **Model Quantizer** is a tool for automatic quantization of full-precision IR models into low-precision versions using Post-Training Optimization Tool. <br> **Model Information Dumper** is a helper utility for dumping information about the models in a stable machine-readable format.|


**Developer package** also provides the **runtime package** installed as a dependency. The runtime package includes the following components:

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

### Step 1. Set Up Python Virtual Environment

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

### Step 2. Activate Virtual Environment

On Linux and macOS:
```sh
source openvino_env/bin/activate
```
On Windows:
```sh
openvino_env\Scripts\activate
```

### Step 3. Set Up and Update PIP to the Highest Version

Run the command below:
```sh
python -m pip install --upgrade pip
```

### Step 4. Install the Package

To install and configure the components of the development package for working with specific frameworks, use the `pip install openvino-dev[extras]` command, where `extras` is a list of extras from the table below: 

| DL Framework                                                                     | Extra                          |
| :------------------------------------------------------------------------------- | :-------------------------------|
|   [Caffe*](https://caffe.berkeleyvision.org/)                                    | caffe                           |
|   [Caffe2*](https://caffe2.ai/)                                                 | caffe2                          |
|   [Kaldi*](https://kaldi-asr.org/)                                               | kaldi                           |
|   [MXNet*](https://mxnet.apache.org/)                                            | mxnet                           |
|   [ONNX*](https://github.com/microsoft/onnxruntime/)                             | onnx                            |
|   [PyTorch*](https://pytorch.org/)                                               | pytorch                         |
|   [TensorFlow* 1.x](https://www.tensorflow.org/versions#tensorflow_1)            | tensorflow                      |
|   [TensorFlow* 2.x](https://www.tensorflow.org/versions#tensorflow_2)            | tensorflow2                     |

For example, to install and configure the components for working with TensorFlow 2.x, MXNet and Caffe, use the following command:  
   ```sh
   pip install openvino-dev[tensorflow2,mxnet,caffe]
   ```

### Step 5. Verify that the Package Is Installed

- To verify that the **developer package** is properly installed, run the command below (this may take a few seconds):
   ```sh
   mo -h
   ```
   You will see the help message for Model Optimizer if installation finished successfully.

- To verify that Inference Engine from the **runtime package** is available, run the command below:
   ```sh
   python -c "from openvino.inference_engine import IECore"
   ```
   You will not see any error messages if installation finished successfully.

## Troubleshooting

### Error: Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio"

On Windows* some dependencies may require compilation from source when installing. To resolve this issue, you need to install [Build Tools for Visual Studio* 2019](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019) and repeat package installation.

### ImportError: libpython3.7m.so.1.0: cannot open shared object file: No such file or directory

To resolve missing external dependency on Ubuntu* 18.04, execute the following command:
```sh
sudo apt-get install libpython3.7
```

## Additional Resources

- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/en-us/openvino-toolkit)
- [OpenVINO™ toolkit online documentation](https://docs.openvinotoolkit.org)
- [OpenVINO™ Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)
